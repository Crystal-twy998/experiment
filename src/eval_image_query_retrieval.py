"""
Evaluate generated-image query branches for ZS-CIR, including avg/weighted embedding fusion.

Typical usage:

CIRR single/fused image-query retrieval:
python src/eval_image_query_retrieval.py \
  --dataset cirr \
  --split val \
  --dataset_path /nativemm/share/cpfs/tangwenyue/Reasoning/Datasets/CIRR \
  --clip ViT-B-32 \
  --mode_dirs instruction_only=/nativemm/share/cpfs/tangwenyue/Reasoning/Datasets/CIRR/Edited_Images/instruction_only \
              instruction_plus_target=/nativemm/share/cpfs/tangwenyue/Reasoning/Datasets/CIRR/Edited_Images/instruction_plus_target \
              target_only=/nativemm/share/cpfs/tangwenyue/Reasoning/Datasets/CIRR/Edited_Images/target_only_t2i \
  --fusion avg weighted \
  --weights instruction_only=0.7 instruction_plus_target=1.0 target_only=0.5

Notes:
- This script does not generate images. It only reads generated image folders.
- It evaluates each branch independently and also evaluates normalized feature fusion.
- Fused embedding = normalize(sum_b w_b * normalize(f_b)).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import tqdm

import data_utils
import compute_results
from classes import load_clip_model_and_preprocess
from datasets import CIRRDataset, CIRCODataset, FashionIQDataset, EditedImageDataset


def parse_kv_pairs(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_weight_pairs(items: List[str], mode_names: List[str]) -> Dict[str, float]:
    weights = {m: 1.0 for m in mode_names}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected MODE=WEIGHT, got: {item}")
        k, v = item.split("=", 1)
        weights[k.strip()] = float(v)
    for m in mode_names:
        weights.setdefault(m, 1.0)
    return weights


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cirr", "circo", "fashioniq_dress", "fashioniq_toptee", "fashioniq_shirt"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--clip", type=str, default="ViT-B-32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--preprocess-type", type=str, default="targetpad", choices=["clip", "targetpad"])
    parser.add_argument("--task", type=str, default="image_query_eval")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mode_dirs",
        nargs="+",
        required=True,
        help="Generated image folders in MODE=DIR format, e.g. instruction_only=/path/to/dir.",
    )
    parser.add_argument(
        "--fusion",
        nargs="*",
        default=["avg", "weighted"],
        choices=["avg", "weighted"],
        help="Embedding fusion modes to evaluate.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="Fusion weights in MODE=FLOAT format. Used when --fusion weighted is enabled.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing generated images by using zero features; not recommended.")
    return parser


def load_dataset(dataset_name: str, split: str, dataset_path: str, processor):
    if "fashioniq" in dataset_name.lower():
        dress_type = dataset_name.split("_")[-1]
        target_dataset = FashionIQDataset(dataset_path, split, [dress_type], "classic", processor)
        query_dataset = FashionIQDataset(dataset_path, split, [dress_type], "relative", processor)
        compute_fn = compute_results.fiq
        pairing = dress_type
    elif dataset_name.lower() == "cirr":
        ds_split = "test1" if split == "test" else split
        target_dataset = CIRRDataset(dataset_path, ds_split, "classic", processor)
        query_dataset = CIRRDataset(dataset_path, ds_split, "relative", processor)
        compute_fn = compute_results.cirr
        pairing = "default"
    elif dataset_name.lower() == "circo":
        target_dataset = CIRCODataset(dataset_path, split, "classic", processor)
        query_dataset = CIRCODataset(dataset_path, split, "relative", processor)
        compute_fn = compute_results.circo
        pairing = "default"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return target_dataset, query_dataset, compute_fn, pairing


def collect_query_meta(query_dataset, dataset_name: str, batch_size: int = 32) -> Dict[str, List[Any]]:
    loader = torch.utils.data.DataLoader(
        dataset=query_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        collate_fn=data_utils.collate_fn,
        shuffle=False,
    )
    reference_names: List[str] = []
    relative_captions: List[str] = []
    target_names: List[str] = []
    gt_img_ids: List[Any] = []
    query_ids: List[Any] = []

    for batch in tqdm.tqdm(loader, desc="Loading query metadata"):
        reference_names.extend(batch["reference_name"])

        if "fashioniq" not in dataset_name.lower():
            relative_captions.extend(batch["relative_caption"])
        else:
            rel_caps = np.array(batch["relative_captions"]).T.flatten().tolist()
            relative_captions.extend([
                f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}"
                for i in range(0, len(rel_caps), 2)
            ])

        if "target_name" in batch:
            target_names.extend(batch["target_name"])

        gt_key = "gt_img_ids"
        if "group_members" in batch:
            gt_key = "group_members"
        if gt_key in batch:
            gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

        query_key = "query_id"
        if "pair_id" in batch:
            query_key = "pair_id"
        if query_key in batch:
            qvals = batch[query_key]
            if isinstance(qvals, torch.Tensor):
                query_ids.extend(qvals.detach().cpu().tolist())
            else:
                query_ids.extend(list(qvals))

    return {
        "reference_names": reference_names,
        "relative_captions": relative_captions,
        "target_names": target_names,
        "targets": gt_img_ids,
        "query_ids": query_ids,
    }


def expected_generated_path(mode_dir: str, ref_name: str, query_id: Any) -> str:
    if isinstance(query_id, torch.Tensor):
        query_id = query_id.item()
    candidates = [
        os.path.join(mode_dir, f"{ref_name}_edited_{query_id}.png"),
        os.path.join(mode_dir, f"{ref_name}_edited_{str(query_id)}.png"),
        os.path.join(mode_dir, f"{ref_name}_{query_id}.png"),
        os.path.join(mode_dir, f"{ref_name}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


@torch.no_grad()
def extract_image_features(device, args, dataset, clip_model, batch_size: int, num_workers: int):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=data_utils.collate_fn,
        shuffle=False,
    )
    feats, names = [], []
    for batch in tqdm.tqdm(loader, desc=f"Encoding {dataset.__class__.__name__}"):
        images = batch.get("image")
        batch_names = batch.get("image_name")
        if images is None:
            images = batch.get("reference_image")
        if batch_names is None:
            batch_names = batch.get("reference_name")
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            batch_features = clip_model.encode_image(images)
        feats.append(batch_features.detach().cpu())
        if batch_names is not None:
            names.extend(batch_names)
    feats = torch.vstack(feats).float()
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats, names


def evaluate_features(
    compute_fn,
    device,
    features,
    meta: Dict[str, List[Any]],
    index_features,
    index_names,
    dataset_name: str,
    dataset_path: str,
    task: str,
    split: str,
    clip_name: str,
    ways: str,
):
    metrics, sorted_names = compute_fn(
        device=device,
        predicted_features=features,
        reference_names=meta["reference_names"],
        targets=meta["targets"],
        target_names=meta["target_names"],
        index_features=index_features,
        index_names=index_names,
        query_ids=meta["query_ids"],
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        task=task,
        preload_dict={},
        split=split,
        loop=0,
        ways=ways,
        clip=clip_name,
        save_outputs=True,
    )
    return metrics, sorted_names


def dump_branch_summary(output_dir: str, metrics_by_branch: Dict[str, Dict[str, float]], weights: Dict[str, float]):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"image_query_eval_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics_by_branch, "weights": weights}, f, indent=4, ensure_ascii=False)
    print(f"Saved summary to: {path}")


def main():
    args = make_parser().parse_args()
    mode_dirs = parse_kv_pairs(args.mode_dirs)
    mode_names = list(mode_dirs.keys())
    weights = parse_weight_pairs(args.weights, mode_names)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")

    print(f"[Eval] device={device}, dataset={args.dataset}, split={args.split}, clip={args.clip}")
    print("[Eval] mode_dirs:")
    for k, v in mode_dirs.items():
        print(f"  - {k}: {v}")
    print("[Eval] weights:", weights)

    clip_model, processor = load_clip_model_and_preprocess(
        dataset_path=args.dataset_path,
        clip_type=args.clip,
        device=device,
    )
    target_dataset, query_dataset, compute_fn, pairing = load_dataset(args.dataset, args.split, args.dataset_path, processor)
    meta = collect_query_meta(query_dataset, args.dataset, batch_size=args.batch_size)

    print("[Eval] Extracting gallery image features...")
    index_features, index_names = extract_image_features(
        device=device,
        args=args,
        dataset=target_dataset,
        clip_model=clip_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)

    features_by_mode: Dict[str, torch.Tensor] = {}
    paths_by_mode: Dict[str, List[str]] = {}
    metrics_by_branch: Dict[str, Dict[str, float]] = {}

    for mode, mode_dir in mode_dirs.items():
        img_paths = [
            expected_generated_path(mode_dir, ref, meta["query_ids"][i] if i < len(meta["query_ids"]) else i)
            for i, ref in enumerate(meta["reference_names"])
        ]
        missing = [p for p in img_paths if not os.path.exists(p)]
        if missing:
            msg = f"[{mode}] missing {len(missing)} generated images; first examples: {missing[:5]}"
            if not args.allow_missing:
                raise FileNotFoundError(msg)
            print("WARNING:", msg)

        ds = EditedImageDataset(img_paths, processor)
        feats, _ = extract_image_features(
            device=device,
            args=args,
            dataset=ds,
            clip_model=clip_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        features_by_mode[mode] = feats
        paths_by_mode[mode] = img_paths

        print(f"\n[Eval] Single branch: {mode}")
        metrics, _ = evaluate_features(
            compute_fn=compute_fn,
            device=device,
            features=feats,
            meta=meta,
            index_features=index_features,
            index_names=index_names,
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            task=args.task,
            split=args.split,
            clip_name=args.clip,
            ways=f"i2i_{mode}",
        )
        print(metrics)
        metrics_by_branch[f"single/{mode}"] = metrics

    if "avg" in args.fusion:
        print("\n[Eval] Fusion: avg embedding")
        fused = torch.stack([features_by_mode[m] for m in mode_names], dim=0).mean(dim=0)
        fused = torch.nn.functional.normalize(fused.float(), dim=-1)
        metrics, _ = evaluate_features(
            compute_fn=compute_fn,
            device=device,
            features=fused,
            meta=meta,
            index_features=index_features,
            index_names=index_names,
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            task=args.task,
            split=args.split,
            clip_name=args.clip,
            ways="i2i_fused_avg",
        )
        print(metrics)
        metrics_by_branch["fusion/avg"] = metrics

    if "weighted" in args.fusion:
        print("\n[Eval] Fusion: weighted embedding")
        denom = sum(max(weights[m], 0.0) for m in mode_names)
        if denom <= 0:
            raise ValueError(f"Invalid non-positive weight sum: {weights}")
        fused = sum(float(weights[m]) * features_by_mode[m] for m in mode_names) / denom
        fused = torch.nn.functional.normalize(fused.float(), dim=-1)
        metrics, _ = evaluate_features(
            compute_fn=compute_fn,
            device=device,
            features=fused,
            meta=meta,
            index_features=index_features,
            index_names=index_names,
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            task=args.task,
            split=args.split,
            clip_name=args.clip,
            ways="i2i_fused_weighted",
        )
        print(metrics)
        metrics_by_branch["fusion/weighted"] = metrics

    output_dir = args.output_dir or os.path.join(args.dataset_path, "task", args.task)
    dump_branch_summary(output_dir, metrics_by_branch, weights)


if __name__ == "__main__":
    main()
