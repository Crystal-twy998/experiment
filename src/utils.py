import json
import os
from typing import Optional, Tuple, List, Dict, Union

import argparse
import clip
import numpy as np
import openai_api
import pickle
import torch
import tqdm
import datasets
import data_utils
import prompts
import datetime
import pandas as pd
import re
import data_utils
import gc
import csv
import torch.distributed as dist

from check_prompt import CheckModel, ModelHandler
from get_pseudo_targets import VQAModelHandler
import file_utils
from PIL import Image
from bagel_inference import BagelImageEditor

if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


def _print_cuda_mem(tag: str):
    if not torch.cuda.is_available():
        print(f"[CUDA] {tag}: CUDA not available")
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"[CUDA] {tag} | GPU {i}: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")


def _release_bagel_editor_local(bagel_editor):
    if bagel_editor is None:
        print("[BAGEL] No BAGEL editor to release inside utils.")
        return None

    print("[BAGEL] Releasing BAGEL inside utils before verifier...")
    try:
        if hasattr(bagel_editor, "model"):
            del bagel_editor.model
            print("[BAGEL] Deleted bagel_editor.model")
    except Exception as e:
        print(f"[BAGEL] Failed deleting model: {e}")

    try:
        if hasattr(bagel_editor, "inferencer"):
            del bagel_editor.inferencer
            print("[BAGEL] Deleted bagel_editor.inferencer")
    except Exception as e:
        print(f"[BAGEL] Failed deleting inferencer: {e}")

    try:
        del bagel_editor
    except Exception as e:
        print(f"[BAGEL] Failed deleting bagel_editor object: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _print_cuda_mem("after BAGEL release")
    return None


def _dist_info() -> Dict[str, Union[bool, int]]:
    initialized = dist.is_available() and dist.is_initialized()
    if not initialized:
        return {"enabled": False, "rank": 0, "world_size": 1}
    return {
        "enabled": True,
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
    }


def _is_main_process() -> bool:
    return _dist_info()["rank"] == 0


def _sanitize_tag(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "default"


def _clip_prefix(clip_name: Optional[str]) -> str:
    if clip_name is None:
        return ""
    clip_tag = _sanitize_tag(clip_name)
    return f"{clip_tag}_" if clip_tag else ""


def _build_verifier_prefix(clip_name: str, check_model_name: str, model_name: str) -> str:
    clip_tag = _sanitize_tag(clip_name) if clip_name is not None else "clip"
    check_tag = _sanitize_tag(check_model_name)
    model_tag = _sanitize_tag(model_name)
    return f"{clip_tag}_{check_tag}_blip2_t5_{model_tag}"


def _merge_distributed_records(local_records: List[Dict]) -> List[Dict]:
    info = _dist_info()
    if not info["enabled"]:
        return sorted(local_records, key=lambda x: x["index"])

    gathered_records = [None for _ in range(info["world_size"])]
    dist.all_gather_object(gathered_records, local_records)
    merged = {}
    for shard in gathered_records:
        if shard is None:
            continue
        for item in shard:
            merged[item["index"]] = item
    ordered = [merged[idx] for idx in sorted(merged.keys())]
    return ordered


def _local_work_indices(total_num: int, use_dist: bool) -> List[int]:
    info = _dist_info()
    if not (use_dist and info["enabled"]):
        return list(range(total_num))
    return list(range(info["rank"], total_num, info["world_size"]))


def _gather_indexed_values(local_pairs: List[tuple], total_num: int) -> List[Optional[object]]:
    values: List[Optional[object]] = [None for _ in range(total_num)]
    for idx, value in local_pairs:
        values[idx] = value

    info = _dist_info()
    if not info["enabled"]:
        return values

    gathered_pairs = [None for _ in range(info["world_size"])]
    dist.all_gather_object(gathered_pairs, local_pairs)
    for shard in gathered_pairs:
        if shard is None:
            continue
        for idx, value in shard:
            values[idx] = value
    return values


@torch.no_grad()
def extract_image_features(device: torch.device, args: argparse.Namespace, dataset: torch.utils.data.Dataset,
                           clip_model: clip.model.CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 8, preload: str = None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        index_ranks = [] if 'index_ranks' not in extracted_data else extracted_data['index_ranks']
        aux_data = {} if 'aux_data' not in extracted_data else extracted_data['aux_data']
    else:
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_utils.collate_fn
        )

        index_features, index_names, index_ranks, aux_data = [], [], [], []

        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception:
            pass

        index_rank = None
        for batch in tqdm.tqdm(loader):
            images = batch.get('image')
            names = batch.get('image_name')
            if images is None:
                images = batch.get('reference_image')
            if names is None:
                names = batch.get('reference_name')

            images = images.to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                batch_features = clip_model.encode_image(images)
                index_features.append(batch_features.cpu())
                index_names.extend(names)
                if index_rank is not None:
                    index_ranks.extend(index_rank)

        index_features = torch.vstack(index_features)

        if preload is not None:
            os.makedirs(os.path.dirname(preload), exist_ok=True)
            if _is_main_process():
                with open(preload, 'wb') as f:
                    pickle.dump(
                        {
                            'index_features': index_features,
                            'index_names': index_names,
                            'index_ranks': index_ranks,
                            'aux_data': aux_data
                        },
                        f
                    )
                print(f"Save image feathers in {preload}")
            if _dist_info()["enabled"]:
                dist.barrier()

    return index_features, index_names, index_ranks, aux_data


@torch.no_grad()
def generate_editimg_caption_iteration(
    device: torch.device,
    args: argparse.Namespace,
    bagel_editor: BagelImageEditor,
    dataset_name: str,
    llm_prompt_args: str,
    retrieval: str,
    clip_model: clip.model.CLIP,
    query_dataset: torch.utils.data.Dataset,
    target_dataset: torch.utils.data.Dataset,
    preload_dict: Dict[str, Union[str, None]],
    processor,
    LLM_model_name,
    max_check_num,
    Check_LLM_model_name,
    VQA_LLM_model_name,
    dataset_path,
    edit_img_dir,
    compute_results_function,
    index_features,
    index_names,
    openai_key,
    task,
    split,
    preprocess,
    **kwargs
) -> Dict[str, Union[torch.Tensor, list, dict, np.ndarray]]:
    stage_mode = getattr(args, "stage_mode", "initial_only")
    torch.cuda.empty_cache()
    os.makedirs(edit_img_dir, exist_ok=True)

    batch_size = 4
    reload_caption_dict = {}
    reload_img_paths_dict = {}

    if preload_dict["captions"] is None or not os.path.exists(preload_dict["captions"]):
        raise AssertionError("Must generate initial captions before!")
    if preload_dict["img_paths"] is None or not os.path.exists(preload_dict["img_paths"]):
        raise AssertionError("Must generate image path csv before!")

    print(f'Loading precomputed image captions from {preload_dict["captions"]}!')
    print(f'Loading precomputed image paths from {preload_dict["img_paths"]}!')
    print(f"stage_mode = {stage_mode}")

    all_captions, relative_captions = [], []
    ref_img_paths = []
    gt_img_ids, query_ids = [], []
    target_names, reference_names = [], []

    query_loader = torch.utils.data.DataLoader(
        dataset=query_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        collate_fn=data_utils.collate_fn,
        shuffle=False
    )
    query_iterator = tqdm.tqdm(query_loader, position=0, desc="Loading image captions...")

    with open(preload_dict["captions"], "r", encoding="utf-8") as blip_captions:
        reader = csv.reader(blip_captions)
        next(reader)
        reload_caption_dict = {caption[0].lstrip("0"): caption[1] for caption in reader}

    with open(preload_dict["img_paths"], "r", encoding="utf-8") as img_paths:
        reader = csv.reader(img_paths)
        next(reader)
        reload_img_paths_dict = {img_path[0].lstrip("0"): img_path[1] for img_path in reader}

    for batch in query_iterator:
        reference_names.extend(batch["reference_name"])

        if "fashioniq" not in dataset_name.lower():
            relative_captions.extend(batch["relative_caption"])
        else:
            rel_caps = batch["relative_captions"]
            rel_caps = np.array(rel_caps).T.flatten().tolist()
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

        for ref_name in batch["reference_name"]:
            all_captions.append(reload_caption_dict[ref_name])
            ref_img_paths.append(reload_img_paths_dict[ref_name])

    distributed_generate = bool(getattr(args, "distributed_generate", False) and _dist_info()["enabled"])
    local_gen_indices = _local_work_indices(len(reference_names), distributed_generate)
    local_gen_set = set(local_gen_indices)
    if distributed_generate:
        info = _dist_info()
        print(f"[BAGEL][Distributed] rank={info['rank']}/{info['world_size']} handles {len(local_gen_indices)} samples")

    def _require_bagel(resource_name: str, cache_path: Optional[str]):
        if bagel_editor is not None:
            return
        raise AssertionError(
            f"{stage_mode} mode needs BAGEL to create missing {resource_name}, "
            f"but BAGEL is not loaded. Missing cache: {cache_path}"
        )

    # ------------------------------------------------------------------
    # Stage-1 text query generation: reuse modified captions if available,
    # otherwise generate them with BAGEL.
    # ------------------------------------------------------------------
    mods_path = preload_dict.get("mods", None)
    print("modified captions preload path:", mods_path)
    print("modified captions exists:", os.path.exists(mods_path) if mods_path else False)

    modified_captions = None
    if mods_path is not None and mods_path != "" and os.path.exists(mods_path):
        print(f'Loading precomputed caption modifiers from {mods_path}!')
        modified_captions = file_utils.read_modified_captions_file(path=mods_path)
        if not isinstance(modified_captions, list) or len(modified_captions) != len(all_captions):
            print(
                f"[BAGEL] Invalid modified captions cache at {mods_path}: "
                f"expected {len(all_captions)}, got {0 if modified_captions is None else len(modified_captions)}. Regenerating."
            )
            modified_captions = None

    if modified_captions is None:
        _require_bagel("modified captions", mods_path)
        print("No valid modified_captions cache found. Generating with BAGEL...")
        local_modified = LLM_modify_editimg_caption(
            bagel_editor,
            LLM_model_name,
            preload_dict,
            llm_prompt_args,
            [all_captions[i] for i in local_gen_indices],
            [relative_captions[i] for i in local_gen_indices],
            openai_key,
            device
        )
        local_pairs = [(idx, value) for idx, value in zip(local_gen_indices, local_modified)]
        modified_captions = _gather_indexed_values(local_pairs, len(all_captions))
        if any(value is None for value in modified_captions):
            missing = [i for i, value in enumerate(modified_captions) if value is None][:10]
            raise RuntimeError(f"Missing modified captions for indices: {missing}")

        if mods_path is None or mods_path == "":
            mods_path = f"{dataset_path}/task/{task}/modified_captions/{dataset_name}_modified_captions.json"
            preload_dict["mods"] = mods_path

        os.makedirs(os.path.dirname(mods_path), exist_ok=True)
        if _is_main_process():
            file_utils.write_modified_captions_file(
                mods_path,
                reference_names=reference_names,
                modified_captions=modified_captions
            )
            print(f"Saved modified captions to: {mods_path}")
        if _dist_info()["enabled"]:
            dist.barrier()
    else:
        print("[BAGEL] Reusing cached modified captions.")

    # ------------------------------------------------------------------
    # Stage-1 image query generation: reuse edited images if available,
    # otherwise generate only the missing ones with BAGEL.
    # ------------------------------------------------------------------
    expected_edit_img_paths = []
    for i, ref_name in enumerate(reference_names):
        query_id = query_ids[i] if len(query_ids) > i else i
        if isinstance(query_id, torch.Tensor):
            query_id = query_id.item()
        expected_edit_img_paths.append(os.path.join(edit_img_dir, f"{ref_name}_edited_{query_id}.png"))

    edit_meta_path = preload_dict.get("edit_images", None)
    loaded_edit_img_paths = None
    if edit_meta_path is not None and edit_meta_path != "" and os.path.exists(edit_meta_path):
        try:
            print(f'Loading precomputed edited images metadata from {edit_meta_path}!')
            edited_data = pickle.load(open(edit_meta_path, "rb"))
            cached_paths = edited_data.get("all_edit_img_paths", None)
            if isinstance(cached_paths, list) and len(cached_paths) == len(reference_names):
                loaded_edit_img_paths = cached_paths
            else:
                print(
                    f"[BAGEL] Invalid edited image metadata at {edit_meta_path}: "
                    f"expected {len(reference_names)} paths, got "
                    f"{0 if cached_paths is None else len(cached_paths)}. Will repair with BAGEL if needed."
                )
        except Exception as e:
            print(f"[BAGEL] Failed to read edited image metadata {edit_meta_path}: {e}")

    all_edit_img_paths = []
    missing_edit_indices = []
    for i in range(len(reference_names)):
        candidate_paths = []
        if loaded_edit_img_paths is not None and i < len(loaded_edit_img_paths):
            candidate_paths.append(loaded_edit_img_paths[i])
        candidate_paths.append(expected_edit_img_paths[i])

        chosen_path = None
        for candidate_path in candidate_paths:
            if isinstance(candidate_path, str) and candidate_path != "" and os.path.exists(candidate_path):
                chosen_path = candidate_path
                break

        if chosen_path is None:
            chosen_path = expected_edit_img_paths[i]
            missing_edit_indices.append(i)

        all_edit_img_paths.append(chosen_path)

    if len(missing_edit_indices) > 0:
        _require_bagel("edited images", edit_meta_path)
        local_missing_edit_indices = [i for i in missing_edit_indices if i in local_gen_set]
        print(
            f"[BAGEL] Missing edited images: {len(missing_edit_indices)} total, "
            f"{len(local_missing_edit_indices)} on this rank. Generating with BAGEL..."
        )

        edit_iter = tqdm.tqdm(local_missing_edit_indices, total=len(local_missing_edit_indices), desc="Editing images...")
        for i in edit_iter:
            ref_img_path = ref_img_paths[i]
            rel_caption = relative_captions[i]
            save_path = expected_edit_img_paths[i]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if not os.path.exists(save_path):
                edited_out = bagel_editor.edit_image_no_think(ref_img_path, rel_caption)

                if isinstance(edited_out, dict):
                    if "image" not in edited_out:
                        raise ValueError(f"Unexpected BAGEL output dict keys: {edited_out.keys()}")
                    edited_img = edited_out["image"]
                else:
                    edited_img = edited_out

                if hasattr(edited_img, "save"):
                    edited_img.save(save_path)
                else:
                    raise ValueError(f"Edited image output is not PIL.Image, got {type(edited_img)}")

        if _dist_info()["enabled"]:
            dist.barrier()

        for i in missing_edit_indices:
            if os.path.exists(expected_edit_img_paths[i]):
                all_edit_img_paths[i] = expected_edit_img_paths[i]

        unresolved = [i for i in missing_edit_indices if not os.path.exists(all_edit_img_paths[i])]
        if len(unresolved) > 0:
            raise RuntimeError(f"Missing edited images for indices: {unresolved[:10]}")
    else:
        print("[BAGEL] Reusing cached edited images.")

    if edit_meta_path is not None and edit_meta_path != "":
        os.makedirs(os.path.dirname(edit_meta_path), exist_ok=True)
        if _is_main_process():
            with open(edit_meta_path, "wb") as f:
                pickle.dump(
                    {
                        "all_edit_img_paths": all_edit_img_paths,
                        "gt_img_ids": gt_img_ids,
                        "relative_captions": relative_captions,
                        "target_names": target_names,
                        "reference_names": reference_names,
                        "query_ids": query_ids
                    },
                    f
                )
            print(f"Saved edited image metadata to {edit_meta_path}")
        if _dist_info()["enabled"]:
            dist.barrier()

    predicted_txt_features = text_encoding(
        device=device,
        clip_model=clip_model,
        input_captions=modified_captions,
        batch_size=32,
        mode=retrieval
    )
    predicted_txt_features = torch.nn.functional.normalize(predicted_txt_features.float(), dim=-1)

    edited_image_dataset = datasets.EditedImageDataset(all_edit_img_paths, preprocess)
    predicted_img_features, _, _, _ = extract_image_features(
        device=device,
        args=args,
        dataset=edited_image_dataset,
        clip_model=clip_model,
        batch_size=32,
        num_workers=4,
        preload=None
    )
    predicted_img_features = torch.nn.functional.normalize(predicted_img_features.float(), dim=-1)

    clip_name = getattr(args, "clip", None)
    clip_prefix = _clip_prefix(clip_name)
    save_outputs = _is_main_process()

    txt_output_metrics, txt_sorted_index_names = compute_results_function(
        device=device,
        predicted_features=predicted_txt_features,
        reference_names=reference_names,
        targets=gt_img_ids,
        target_names=target_names,
        index_features=index_features,
        index_names=index_names,
        query_ids=query_ids,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        task=task,
        preload_dict=preload_dict,
        split=split,
        loop=0,
        ways="t2i",
        clip=clip_name,
        save_outputs=save_outputs
    )

    img_output_metrics, img_sorted_index_names = compute_results_function(
        device=device,
        predicted_features=predicted_img_features,
        reference_names=reference_names,
        targets=gt_img_ids,
        target_names=target_names,
        index_features=index_features,
        index_names=index_names,
        query_ids=query_ids,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        task=task,
        preload_dict=preload_dict,
        split=split,
        loop=0,
        ways="i2i",
        clip=clip_name,
        save_outputs=save_outputs
    )

    os.makedirs(f"{dataset_path}/task/{task}", exist_ok=True)
    initial_rank_path = ""
    if save_outputs:
        initial_rank_path = f"{dataset_path}/task/{task}/top_rank_{clip_prefix}loop_0_{task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    rank_dump = []
    for i in range(len(reference_names)):
        qid = query_ids[i] if len(query_ids) > i else i
        if isinstance(qid, torch.Tensor):
            qid = qid.item()

        t2i_top = txt_sorted_index_names[i][:50].tolist() if hasattr(txt_sorted_index_names[i], "tolist") else list(txt_sorted_index_names[i][:50])
        i2i_top = img_sorted_index_names[i][:50].tolist() if hasattr(img_sorted_index_names[i], "tolist") else list(img_sorted_index_names[i][:50])

        item = {
            "image_index": reference_names[i],
            "query_id": qid,
            "target_name": target_names[i] if len(target_names) > i else None,
            "relative_caption": relative_captions[i],
            "modified_caption": modified_captions[i],
            "edited_image_path": all_edit_img_paths[i],
            "t2i_top_names": t2i_top,
            "i2i_top_names": i2i_top
        }
        rank_dump.append(item)

    if save_outputs:
        with open(initial_rank_path, "w", encoding="utf-8") as f:
            json.dump(rank_dump, f, indent=4, ensure_ascii=False)
        print(f"Saved initial dual-path rankings to: {initial_rank_path}")

    if stage_mode == "initial_only":
        return {
            "stage": "initial_only",
            "predicted_txt_features": predicted_txt_features,
            "predicted_img_features": predicted_img_features,
            "modified_captions": modified_captions,
            "all_edit_img_paths": all_edit_img_paths,
            "reference_names": reference_names,
            "target_names": target_names,
            "targets": gt_img_ids,
            "query_ids": query_ids,
            "instructions": relative_captions,
            "start_captions": all_captions,
            "txt_sorted_index_names": txt_sorted_index_names,
            "img_sorted_index_names": img_sorted_index_names,
            "txt_output_metrics": txt_output_metrics,
            "img_output_metrics": img_output_metrics,
            "initial_rank_path": initial_rank_path
        }

    elif stage_mode == "qwen_fusion":
        topk_for_vqa = getattr(args, "topk_for_vqa", 10)
        print(f"[Verifier] topk_for_vqa = {topk_for_vqa}")
        _print_cuda_mem("before BAGEL release for verifier")

        bagel_editor = _release_bagel_editor_local(bagel_editor)
        _print_cuda_mem("before Qwen load")

        txt_top_names = [list(x[:topk_for_vqa]) for x in txt_sorted_index_names]
        img_top_names = [list(x[:topk_for_vqa]) for x in img_sorted_index_names]

        txt_top_captions = []
        img_top_captions = []
        txt_top_img_paths = []
        img_top_img_paths = []

        for i in range(len(reference_names)):
            txt_caps_i, img_caps_i = [], []
            txt_paths_i, img_paths_i = [], []

            for name in txt_top_names[i]:
                key = str(name).lstrip("0")
                txt_caps_i.append(reload_caption_dict.get(key, ""))
                txt_paths_i.append(reload_img_paths_dict.get(key, ""))

            for name in img_top_names[i]:
                key = str(name).lstrip("0")
                img_caps_i.append(reload_caption_dict.get(key, ""))
                img_paths_i.append(reload_img_paths_dict.get(key, ""))

            txt_top_captions.append(txt_caps_i)
            img_top_captions.append(img_caps_i)
            txt_top_img_paths.append(txt_paths_i)
            img_top_img_paths.append(img_paths_i)

        txt_check_index = [True for _ in range(len(reference_names))]
        img_check_index = [True for _ in range(len(reference_names))]

        print("[Verifier] Start loading/running Qwen verifier...")
        candidates1, candidates2, ranks1, ranks2, pseudo_targets1, confidences1, pseudo_targets2, confidences2, txt_check_index, img_check_index = get_pseudo_targets(
            Check_LLM_model_name=Check_LLM_model_name,
            openai_key=openai_key,
            dataset_path=dataset_path,
            task=task,
            loop=0,
            reference_names=reference_names,
            model_name=VQA_LLM_model_name,
            txt_top_captions=txt_top_captions,
            img_top_captions=img_top_captions,
            txt_top_img_paths=txt_top_img_paths,
            img_top_img_paths=img_top_img_paths,
            all_captions=all_captions,
            ref_img_paths=ref_img_paths,
            relative_captions=relative_captions,
            txt_check_index=txt_check_index,
            img_check_index=img_check_index,
            clip_name=getattr(args, "clip", None),
            min_pixels=getattr(args, "vqa_min_pixels", 4 * 28 * 28),
            max_pixels=getattr(args, "vqa_max_pixels", 2048 * 28 * 28),
            image_max_size=getattr(args, "vqa_image_max_size", 1024),
            attn_implementation=getattr(args, "vqa_attn_implementation", "sdpa"),
            distributed_vqa=getattr(args, "distributed_vqa", False),
            cleanup_every=getattr(args, "vqa_cleanup_every", 16),
            kwargs_source=args,
            device=device
        )

        print("Qwen verifier finished.")
        _print_cuda_mem("after Qwen verifier")
        print(f"Num text-path candidate sets: {len(candidates1)}")
        print(f"Num image-path candidate sets: {len(candidates2)}")

        return {
            "stage": "qwen_fusion",
            "predicted_txt_features": predicted_txt_features,
            "predicted_img_features": predicted_img_features,
            "modified_captions": modified_captions,
            "all_edit_img_paths": all_edit_img_paths,
            "reference_names": reference_names,
            "target_names": target_names,
            "targets": gt_img_ids,
            "query_ids": query_ids,
            "instructions": relative_captions,
            "start_captions": all_captions,
            "txt_sorted_index_names": txt_sorted_index_names,
            "img_sorted_index_names": img_sorted_index_names,
            "txt_output_metrics": txt_output_metrics,
            "img_output_metrics": img_output_metrics,
            "initial_rank_path": initial_rank_path,
            "candidates1": candidates1,
            "candidates2": candidates2,
            "ranks1": ranks1,
            "ranks2": ranks2,
            "pseudo_targets1": pseudo_targets1,
            "pseudo_targets2": pseudo_targets2,
            "confidences1": confidences1,
            "confidences2": confidences2,
            "txt_check_index": txt_check_index,
            "img_check_index": img_check_index,
            "txt_top_captions": txt_top_captions,
            "img_top_captions": img_top_captions,
            "txt_top_img_paths": txt_top_img_paths,
            "img_top_img_paths": img_top_img_paths,
            "ref_img_paths": ref_img_paths
        }

    else:
        raise ValueError(f"Unknown stage_mode: {stage_mode}")


def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y.%m.%d-%H_%M_%S")


def extract_suggestions(original_suggestion):
    suggestion_split = original_suggestion.split('\n')
    total_suggestion = ""
    suggestion_flag = False
    for line in suggestion_split:
        if suggestion_flag:
            total_suggestion = total_suggestion + line
        elif 'suggestion:' in line or 'Suggestion:' in line:
            suggestion_flag = True
            total_suggestion = total_suggestion + line.split('uggestion:')[1].strip()
    return total_suggestion


def LLM_remodify_editimg_caption(bagel_editor, LLM_model_name, llm_prompt_args, last_captions, last_img_pths,
                                 ref_img_paths, all_captions, relative_captions, t2i_suggestions, i2i_suggestions,
                                 openai_key, device: torch.device, txt_check_index, img_check_index, reference_names,
                                 target_names, query_ids, edited_image_dir, loop, task, dataset_path):
    modified_captions = []
    edit_img_paths = []
    base_prompt = eval(llm_prompt_args)
    input_suggestions1 = []
    input_suggestions2 = []

    path1_scores = []
    path2_scores = []

    if LLM_model_name == "bagel":
        for i in tqdm.trange(len(all_captions), position=1, desc='Remodifying captions with LLM...'):
            t2i_total_suggestions = extract_suggestions(t2i_suggestions[i])
            i2i_total_suggestions = extract_suggestions(i2i_suggestions[i])
            if not t2i_total_suggestions:
                t2i_total_suggestions = t2i_suggestions[i]
            if not i2i_total_suggestions:
                i2i_total_suggestions = i2i_suggestions[i]

            need_text_modify = (
                txt_check_index[i] and
                "Good retrieval, no more loops needed" not in t2i_total_suggestions and
                t2i_total_suggestions != ""
            )

            need_image_modify = (
                img_check_index[i] and
                "Good retrieval, no more loops needed" not in i2i_total_suggestions and
                i2i_total_suggestions != ""
            )

            if need_text_modify:
                cleaned_relative = relative_captions[i].strip('.?, ')
                cleaned_suggestions = t2i_total_suggestions.strip('.?,"\' ')
                instruction = f"{cleaned_relative} and {cleaned_suggestions}."

                input_suggestions1.append(t2i_total_suggestions)
                final_prompt = f"""
                {base_prompt}
                Image Content: {all_captions[i]}.
                Instruction: {instruction}.
                """
                resp = bagel_editor.generate_caption(final_prompt)
                resp = resp.split('\n')
                description = ""
                for line in resp:
                    if line.strip().startswith('Edited Description:'):
                        description = line.split(':')[1].strip()
                        break
                modified_captions.append(description if description else last_captions[i])
                txt_check_index[i] = True
            else:
                input_suggestions1.append("Good retrieval, no more loops needed")
                modified_captions.append(last_captions[i])
                txt_check_index[i] = False

            if need_image_modify:
                input_suggestions2.append(i2i_total_suggestions)

                if not target_names:
                    target_identifier = query_ids[i]
                else:
                    target_identifier = target_names[i]

                tmp_pth = os.path.join(edited_image_dir, f"{reference_names[i]}_edited_{target_identifier}.png")

                if os.path.exists(tmp_pth):
                    edit_img_path = tmp_pth
                else:
                    cleaned_relative = relative_captions[i].strip('.?, ')
                    cleaned_suggestions = i2i_total_suggestions.strip('.?,"\' ')
                    i2i_prompt = f"{cleaned_relative} and {cleaned_suggestions}."
                    edit_result = bagel_editor.edit_image_no_think(ref_img_paths[i], i2i_prompt)
                    edit_result_img = edit_result['image']
                    edit_img_path = file_utils.write_edited_image(
                        edited_image_dir,
                        reference_name=reference_names[i],
                        target_name=target_identifier,
                        edit_result_img=edit_result_img
                    )

                edit_img_paths.append(edit_img_path)
                img_check_index[i] = True
            else:
                input_suggestions2.append("Good retrieval, no more loops needed")

                if not target_names:
                    target_identifier = query_ids[i]
                else:
                    target_identifier = target_names[i]

                expected_img_path = os.path.join(edited_image_dir, f"{reference_names[i]}_edited_{target_identifier}.png")

                if os.path.exists(expected_img_path):
                    edit_img_paths.append(expected_img_path)
                else:
                    source_path = last_img_pths[i]
                    if os.path.exists(source_path):
                        import shutil
                        shutil.copy2(source_path, expected_img_path)
                        print(f"Copied image from {source_path} to {expected_img_path}")
                        edit_img_paths.append(expected_img_path)
                    else:
                        edit_img_paths.append(source_path)

                img_check_index[i] = False

        return modified_captions, edit_img_paths, txt_check_index, img_check_index, input_suggestions1, input_suggestions2, path1_scores, path2_scores


def LLM_modify_editimg_caption(bagel_editor, LLM_model_name, preload_dict, llm_prompt_args,
                               all_captions, relative_captions, openai_key, device: torch.device):
    modified_captions = []
    base_prompt = eval(llm_prompt_args)
    if LLM_model_name == "bagel":
        for i in tqdm.trange(len(all_captions), position=1, desc='Modifying captions with LLM...'):
            instruction = relative_captions[i]
            img_caption = all_captions[i]
            final_prompt = base_prompt + '\n' + "Image Content: " + img_caption
            final_prompt = final_prompt + '\n' + 'Instruction: ' + instruction

            resp = bagel_editor.generate_caption(final_prompt)
            resp = resp.split('\n')
            description = ""
            aug = False
            for line in resp:
                if line.strip().startswith('Edited Description:'):
                    description = line.split(':')[1].strip()
                    if description == "":
                        modified_captions.append(relative_captions[i])
                    else:
                        modified_captions.append(description)
                    aug = True
                    break
            if not aug:
                modified_captions.append(relative_captions[i])
        return modified_captions


def check_prompt(dataset_path, task, loop, reference_names, model_name, txt_top_captions, img_top_captions,
                 all_captions, ref_img_paths, relative_captions, openai_key, txt_check_index, img_check_index,
                 device: torch.device):
    model_handler = ModelHandler(model_type=model_name, device=device, openai_key=openai_key)

    t2i_suggestions = []
    i2i_suggestions = []
    for i in tqdm.trange(len(relative_captions), position=1, desc='Generate suggestions with LLM...'):
        t2i_suggestion = ''
        i2i_suggestion = ''
        if txt_check_index[i] is True:
            t2i_suggestion = model_handler.chat_function(
                "t2i", all_captions[i], relative_captions[i], txt_top_captions[i], img_top_captions[i], device, max_length=10000
            )
        if img_check_index[i] is True:
            i2i_suggestion = model_handler.chat_function(
                "i2i", all_captions[i], relative_captions[i], txt_top_captions[i], img_top_captions[i], device, max_length=10000
            )

        if t2i_suggestion is None:
            t2i_suggestion = ''
        if i2i_suggestion is None:
            i2i_suggestion = ''

        t2i_suggestions.append(t2i_suggestion)
        i2i_suggestions.append(i2i_suggestion)
    return t2i_suggestions, i2i_suggestions



def _collect_model_path_overrides(kwargs_source) -> Dict[str, str]:
    if kwargs_source is None:
        return {}
    override_keys = [
        "qwen2_5_vl_7b_path",
        "qwen2_vl_7b_path",
        "qwen2_5_vl_3b_path",
        "qwen2_5_vl_32b_path",
        "qwen2_5_vl_72b_path",
        "qwen3_vl_8b_path",
    ]
    overrides = {}
    for key in override_keys:
        value = getattr(kwargs_source, key, None)
        if value:
            overrides[key] = value
    return overrides



def get_pseudo_targets(Check_LLM_model_name, openai_key, dataset_path, task, loop, reference_names, model_name,
                       txt_top_captions, img_top_captions, txt_top_img_paths, img_top_img_paths, all_captions,
                       ref_img_paths, relative_captions, txt_check_index, img_check_index, device: torch.device,
                       clip_name: Optional[str] = None,
                       min_pixels: Optional[int] = None,
                       max_pixels: Optional[int] = None,
                       image_max_size: Optional[int] = None,
                       attn_implementation: Optional[str] = None,
                       distributed_vqa: bool = False,
                       cleanup_every: int = 16,
                       kwargs_source = None):
    info = _dist_info()
    use_dist = bool(distributed_vqa and info["enabled"])
    rank = info["rank"]
    world_size = info["world_size"]

    if use_dist:
        print(f"[Qwen][Distributed] rank={rank}/{world_size} on device={device}")
    else:
        print(f"[Qwen] Building VQAModelHandler on device={device}")

    model_handler = VQAModelHandler(
        model_type=model_name,
        device=device,
        openai_key=openai_key,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_max_size=image_max_size,
        attn_implementation=attn_implementation,
        model_path_overrides=_collect_model_path_overrides(kwargs_source)
    )
    print("[Qwen] VQAModelHandler ready.")

    threshold = 0.7
    num_queries = len(relative_captions)
    local_indices = list(range(rank, num_queries, world_size)) if use_dist else list(range(num_queries))
    local_records = []

    progress_desc = f'Scoring candidates with VQA (rank {rank})' if use_dist else 'Scoring candidates with VQA...'
    for local_step, i in enumerate(tqdm.tqdm(local_indices, position=1, desc=progress_desc)):
        txt_scores = []
        for j, candidate_path in enumerate(txt_top_img_paths[i]):
            confidence = model_handler.chat_function(
                ref_img_paths[i],
                relative_captions[i],
                candidate_path,
                device
            )
            txt_scores.append((j + 1, candidate_path, float(confidence)))

        img_scores = []
        for j, candidate_path in enumerate(img_top_img_paths[i]):
            confidence = model_handler.chat_function(
                ref_img_paths[i],
                relative_captions[i],
                candidate_path,
                device
            )
            img_scores.append((j + 1, candidate_path, float(confidence)))

        if len(txt_scores) > 0:
            best_txt = max(txt_scores, key=lambda x: x[2])
            rank1, pseudo_target1, confidence1 = best_txt[0], best_txt[1], float(best_txt[2])
        else:
            rank1, pseudo_target1, confidence1 = None, None, 0.0

        if len(img_scores) > 0:
            best_img = max(img_scores, key=lambda x: x[2])
            rank2, pseudo_target2, confidence2 = best_img[0], best_img[1], float(best_img[2])
        else:
            rank2, pseudo_target2, confidence2 = None, None, 0.0

        local_records.append({
            "index": i,
            "reference_name": reference_names[i],
            "txt_scores": txt_scores,
            "img_scores": img_scores,
            "rank1": rank1,
            "rank2": rank2,
            "pseudo_target1": pseudo_target1,
            "confidence1": confidence1,
            "pseudo_target2": pseudo_target2,
            "confidence2": confidence2,
            "txt_check_flag": not (confidence1 > threshold),
            "img_check_flag": not (confidence2 > threshold),
        })

        if cleanup_every > 0 and ((local_step + 1) % cleanup_every == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ordered_records = _merge_distributed_records(local_records)
    if len(ordered_records) != num_queries:
        raise RuntimeError(f"Expected {num_queries} VQA records, got {len(ordered_records)}")

    candidates1 = [[] for _ in range(num_queries)]
    candidates2 = [[] for _ in range(num_queries)]
    ranks1 = [None for _ in range(num_queries)]
    ranks2 = [None for _ in range(num_queries)]
    pseudo_targets1 = [None for _ in range(num_queries)]
    pseudo_targets2 = [None for _ in range(num_queries)]
    confidences1 = [0.0 for _ in range(num_queries)]
    confidences2 = [0.0 for _ in range(num_queries)]

    merged_txt_check_index = list(txt_check_index)
    merged_img_check_index = list(img_check_index)

    for item in ordered_records:
        i = item["index"]
        candidates1[i] = item["txt_scores"]
        candidates2[i] = item["img_scores"]
        ranks1[i] = item["rank1"]
        ranks2[i] = item["rank2"]
        pseudo_targets1[i] = item["pseudo_target1"]
        pseudo_targets2[i] = item["pseudo_target2"]
        confidences1[i] = float(item["confidence1"])
        confidences2[i] = float(item["confidence2"])
        merged_txt_check_index[i] = item["txt_check_flag"]
        merged_img_check_index[i] = item["img_check_flag"]

    if (not use_dist) or rank == 0:
        pseudo_dir = f'{dataset_path}/task/{task}/pseudo_targets'
        os.makedirs(pseudo_dir, exist_ok=True)
        prefix = _build_verifier_prefix(clip_name or "clip", Check_LLM_model_name, model_name)
        candidates_path = f'{pseudo_dir}/{prefix}_candidates_loop_{loop}_{task}.json'
        pseudo_targets_path = f'{pseudo_dir}/{prefix}_pseudo_targets_loop_{loop}_{task}.json'

        for output_path in [candidates_path, pseudo_targets_path]:
            if os.path.exists(output_path):
                os.remove(output_path)

        for i in range(num_queries):
            file_utils.write_candidates_file(
                candidates_path,
                reference_name=reference_names[i],
                txt_scores=candidates1[i],
                img_scores=candidates2[i]
            )
            file_utils.write_a_pseudo_target_file(
                pseudo_targets_path,
                reference_name=reference_names[i],
                rank1=ranks1[i],
                rank2=ranks2[i],
                pseudo_target1=pseudo_targets1[i],
                confidence1=confidences1[i],
                pseudo_target2=pseudo_targets2[i],
                confidence2=confidences2[i]
            )

    if hasattr(model_handler, "release"):
        model_handler.release()

    if use_dist:
        dist.barrier()

    return (
        candidates1,
        candidates2,
        ranks1,
        ranks2,
        pseudo_targets1,
        confidences1,
        pseudo_targets2,
        confidences2,
        merged_txt_check_index,
        merged_img_check_index
    )


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_recall(indices, targets):
    if len(targets.size()) == 1:
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0:
            return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:
        recall = []
        for preds, gt in zip(indices, targets):
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))
        return torch.Tensor(recall).float().mean()


def text_encoding(device, clip_model, input_captions, batch_size=32, mode='default'):
    n_iter = int(np.ceil(len(input_captions) / batch_size))
    predicted_features = []

    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i * batch_size:(i + 1) * batch_size]

        if hasattr(clip_model, 'tokenizer'):
            tokenized_input_captions = clip_model.tokenizer(captions_to_use, context_length=77).to(device)
        else:
            tokenized_input_captions = clip.tokenize(captions_to_use, context_length=77, truncate=True).to(device)

        clip_text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)

    return torch.nn.functional.normalize(predicted_features, dim=-1)


prompt_ensemble = [
    'A bad photo of a {}',
    'A photo of many {}',
    'A sculpture of a {}',
    'A photo of the hard to see {}',
    'A low resolution photo of the {}',
    'A rendering of a {}',
    'Graffiti of a {}',
    'A bad photo of the {}',
    'A cropped photo of the {}',
    'A tattoo of a {}',
    'The embroidered {}',
    'A photo of a hard to see {}',
    'A bright photo of a {}',
    'A photo of a clean {}',
    'A photo of a dirty {}',
    'A dark photo of the {}',
    'A drawing of a {}',
    'A photo of my {}',
    'The plastic {}',
    'A photo of the cool {}',
    'A close-up photo of a {}',
    'A black and white photo of the {}',
    'A painting of the {}',
    'A painting of a {}',
    'A pixelated photo of the {}',
    'A sculpture of the {}',
    'A bright photo of the {}',
    'A cropped photo of a {}',
    'A plastic {}',
    'A photo of the dirty {}',
    'A jpeg corrupted photo of a {}',
    'A blurry photo of the {}',
    'A photo of the {}',
    'A good photo of the {}',
    'A rendering of the {}',
    'A {} in a video game',
    'A photo of one {}',
    'A doodle of a {}',
    'A close-up photo of the {}',
    'A photo of a {}',
    'The origami {}',
    'The {} in a video game',
    'A sketch of a {}',
    'A doodle of the {}',
    'A origami {}',
    'A low resolution photo of a {}',
    'The toy {}',
    'A rendition of the {}',
    'A photo of the clean {}',
    'A photo of a large {}',
    'A rendition of a {}',
    'A photo of a nice {}',
    'A photo of a weird {}',
    'A blurry photo of a {}',
    'A cartoon {}',
    'Art of a {}',
    'A sketch of the {}',
    'A embroidered {}',
    'A pixelated photo of a {}',
    'Itap of the {}',
    'A jpeg corrupted photo of the {}',
    'A good photo of a {}',
    'A plushie {}',
    'A photo of the nice {}',
    'A photo of the small {}',
    'A photo of the weird {}',
    'The cartoon {}',
    'Art of the {}',
    'A drawing of the {}',
    'A photo of the large {}',
    'A black and white photo of a {}',
    'The plushie {}',
    'A dark photo of a {}',
    'Itap of a {}',
    'Graffiti of the {}',
    'A toy {}',
    'Itap of my {}',
    'A photo of a cool {}',
    'A photo of a small {}',
    'A tattoo of the {}',
]
