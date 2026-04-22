from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch

import utils as base_utils
import compute_results_ipcir_qwen
from stage1_pooling import build_ipcir_stage1_pool


def _to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _default_lambda(dataset_name: str) -> float:
    ds = (dataset_name or "").lower()
    if ds == "circo":
        return 0.3
    if ds == "cirr":
        return 0.0
    if "fashioniq" in ds:
        return 0.8
    return 0.3


def _read_lookup_csv(csv_path: str, value_key: str) -> Dict[str, str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: Dict[str, str] = {}
        for row in reader:
            image_id = str(row["image_id"]).lstrip("0")
            out[image_id] = row[value_key]
    return out


def _get_value_by_name(lookup: Dict[str, str], name: str) -> str:
    key = str(name).lstrip("0")
    return lookup.get(key, "")


def _build_candidate_side_info(
    candidate_names: Sequence[Sequence[str]],
    caption_lookup: Dict[str, str],
    path_lookup: Dict[str, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    all_caps: List[List[str]] = []
    all_paths: List[List[str]] = []
    for names in candidate_names:
        caps = [_get_value_by_name(caption_lookup, n) for n in names]
        paths = [_get_value_by_name(path_lookup, n) for n in names]
        all_caps.append(caps)
        all_paths.append(paths)
    return all_caps, all_paths


def _build_ref_img_paths(reference_names: Sequence[str], path_lookup: Dict[str, str]) -> List[str]:
    return [_get_value_by_name(path_lookup, name) for name in reference_names]


def _is_number_like(x: Any) -> bool:
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, str):
        try:
            float(x)
            return True
        except Exception:
            return False
    return False


def _extract_vqa_score(item: Any) -> float:
    if isinstance(item, dict):
        for key in ("score", "confidence", "prob", "value"):
            if key in item and _is_number_like(item[key]):
                return float(item[key])
        for value in item.values():
            if _is_number_like(value):
                return float(value)
        raise TypeError(f"Cannot extract numeric VQA score from dict: {item!r}")

    if isinstance(item, (list, tuple)):
        # In this repo VQA candidate records are typically
        # (rank_index, candidate_path, confidence). Prefer the last field.
        for value in reversed(item):
            if _is_number_like(value):
                return float(value)
        raise TypeError(f"Cannot extract numeric VQA score from sequence: {item!r}")

    if _is_number_like(item):
        return float(item)

    raise TypeError(f"Cannot extract numeric VQA score from object: {item!r}")


def _build_verifier_score_maps(
    candidate_names: Sequence[Sequence[str]],
    raw_scores: Sequence[Any],
) -> List[Dict[str, float]]:
    all_maps: List[Dict[str, float]] = []
    for names, raw in zip(candidate_names, raw_scores):
        names_list = [str(x) for x in names]
        if isinstance(raw, dict):
            score_map = {}
            for name in names_list:
                if name in raw:
                    try:
                        score_map[name] = float(_extract_vqa_score(raw[name]))
                    except Exception:
                        continue
            all_maps.append(score_map)
            continue

        if raw is None:
            all_maps.append({})
            continue

        values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        usable = min(len(names_list), len(values))
        score_map = {}
        for idx in range(usable):
            try:
                score_map[names_list[idx]] = float(_extract_vqa_score(values[idx]))
            except Exception:
                continue
        all_maps.append(score_map)

    while len(all_maps) < len(candidate_names):
        all_maps.append({})
    return all_maps


def _empty_stage2_outputs(num_queries: int) -> Dict[str, Any]:
    return {
        "candidates1": [[] for _ in range(num_queries)],
        "candidates2": [[] for _ in range(num_queries)],
        "ranks1": [{} for _ in range(num_queries)],
        "ranks2": [{} for _ in range(num_queries)],
        "pseudo_targets1": [None for _ in range(num_queries)],
        "confidences1": [0.0 for _ in range(num_queries)],
        "pseudo_targets2": [None for _ in range(num_queries)],
        "confidences2": [0.0 for _ in range(num_queries)],
        "txt_check_index": [True for _ in range(num_queries)],
        "img_check_index": [True for _ in range(num_queries)],
        "rerank_candidates1_names": [[] for _ in range(num_queries)],
        "rerank_candidates2_names": [[] for _ in range(num_queries)],
    }


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _get_time() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y.%m.%d-%H_%M_%S")


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _ranking_to_dict(rankings: Sequence[Sequence[str]], query_ids: Sequence[Any] | None) -> Dict[str, List[str]]:
    if query_ids is None:
        query_ids = list(range(len(rankings)))
    out: Dict[str, List[str]] = {}
    for qid, rank in zip(query_ids, rankings):
        out[str(qid)] = [str(x) for x in rank]
    return out


def _task_dir(dataset_path: str, task: str) -> str:
    return os.path.join(dataset_path, "task", task)


def _save_stage1_metric_artifact(dataset_path: str, task: str, clip: str, dataset: str, metrics: Dict[str, float]) -> str:
    save_path = os.path.join(
        _task_dir(dataset_path, task),
        f"result_{clip}_{dataset}_merged_loop_0_{_get_time()}.json",
    )
    _write_json(save_path, metrics)
    print(f"[Artifact] saved stage1 merged metrics to: {save_path}")
    return save_path


def _save_stage1_rank_artifact(
    dataset_path: str,
    task: str,
    clip: str,
    dataset: str,
    rankings: Sequence[Sequence[str]],
    query_ids: Sequence[Any] | None,
) -> str:
    save_path = os.path.join(
        _task_dir(dataset_path, task),
        f"top_rank_{clip}_{dataset}_merged_loop_0_{_get_time()}.json",
    )
    _write_json(save_path, _ranking_to_dict(rankings, query_ids))
    print(f"[Artifact] saved stage1 merged rankings to: {save_path}")
    return save_path


def _compute_stage1_metrics_and_labels(dataset_name: str, stage1_out: Dict[str, Any], kwargs: Dict[str, Any]):
    input_kwargs = dict(kwargs)
    input_kwargs.update(stage1_out)
    ds = (dataset_name or "").lower()
    if ds == "cirr":
        return compute_results_ipcir_qwen.cirr_stage1_pool(**input_kwargs)
    if ds == "circo":
        return compute_results_ipcir_qwen.circo_stage1_pool(**input_kwargs)
    if "fashioniq" in ds:
        return compute_results_ipcir_qwen.fiq_stage1_pool(**input_kwargs)
    return None, None


def generate_editimg_caption_iteration(**kwargs):
    """
    1) Reuse the original code to generate:
       - text branch (target captions)
       - proxy-image branch (edited image)
       - original t2i / i2i metrics and json artifacts
    2) Build a strict IP-CIR merged stage-1 pool:
       - f_s = f_t - f_o
       - f_RP = f_p + scaled(f_q) + scaled(f_s)
       - S_f = lambda * S_t + (1 - lambda) * (S_t * S_p)
    3) Save merged stage-1 artifacts immediately, before VQA.
       This guarantees that initial_only and qwen_fusion both get the same
       stage1 merged files, and that stage2 failures do not affect stage1 files.
    4) Take the merged pool topK for verifier reranking when stage_mode=qwen_fusion.
    """
    args = kwargs["args"]
    requested_stage_mode = getattr(args, "stage_mode", "initial_only")

    setattr(args, "stage_mode", "initial_only")
    stage1_out = base_utils.generate_editimg_caption_iteration(**kwargs)
    setattr(args, "stage_mode", requested_stage_mode)

    query_caption_features = base_utils.text_encoding(
        device=kwargs["device"],
        clip_model=kwargs["clip_model"],
        input_captions=stage1_out["start_captions"],
        batch_size=32,
        mode=kwargs["retrieval"],
    )
    query_caption_features = torch.nn.functional.normalize(query_caption_features.float(), dim=-1)

    rerank_pool_size = int(getattr(args, "rerank_pool_size", getattr(args, "topk_for_vqa", 50)))
    prior_topk = int(getattr(args, "ipcir_prior_topk", max(rerank_pool_size, 100)))
    lambda_weight = float(getattr(args, "ipcir_lambda", _default_lambda(kwargs["dataset_name"])))

    pool_result = build_ipcir_stage1_pool(
        reference_names=stage1_out["reference_names"],
        target_caption_features=stage1_out["predicted_txt_features"],
        proxy_image_features=stage1_out["predicted_img_features"],
        query_caption_features=query_caption_features,
        index_features=kwargs["index_features"],
        index_names=kwargs["index_names"],
        lambda_weight=lambda_weight,
        prior_topk=prior_topk,
    )

    stage1_out.update(
        {
            "query_caption_features": query_caption_features,
            "stage1_pool_names": pool_result.merged_names,
            "stage1_pool_score_maps": pool_result.merged_score_maps,
            "stage1_txt_score_maps": pool_result.text_score_maps,
            "stage1_img_score_maps": pool_result.proxy_score_maps,
            "stage1_lambda": lambda_weight,
        }
    )

    # Save stage1 merged artifacts right here, before any VQA reranking.
    stage1_metrics, stage1_labels = _compute_stage1_metrics_and_labels(kwargs["dataset_name"], stage1_out, kwargs)
    stage1_out["stage1_output_metrics"] = stage1_metrics
    stage1_out["stage1_output_labels"] = stage1_labels
    stage1_out["stage1_metric_artifact_path"] = None
    stage1_out["stage1_rank_artifact_path"] = None

    is_test_split = str(kwargs.get("split", "")).lower().startswith("test")
    if _is_main_process():
        if stage1_metrics is not None:
            stage1_out["stage1_metric_artifact_path"] = _save_stage1_metric_artifact(
                dataset_path=kwargs["dataset_path"],
                task=kwargs["task"],
                clip=kwargs["clip"],
                dataset=kwargs["dataset_name"],
                metrics=stage1_metrics,
            )
        if stage1_labels is not None and not is_test_split:
            stage1_out["stage1_rank_artifact_path"] = _save_stage1_rank_artifact(
                dataset_path=kwargs["dataset_path"],
                task=kwargs["task"],
                clip=kwargs["clip"],
                dataset=kwargs["dataset_name"],
                rankings=stage1_labels,
                query_ids=stage1_out.get("query_ids"),
            )

    if requested_stage_mode == "initial_only":
        return stage1_out

    if requested_stage_mode != "qwen_fusion":
        raise ValueError(f"Unsupported stage_mode in patched wrapper: {requested_stage_mode}")

    distributed_vqa = bool(getattr(args, "distributed_vqa", False))
    if torch.distributed.is_available() and torch.distributed.is_initialized() and (not distributed_vqa) and (not _is_main_process()):
        print("[VQA] Distributed run detected with distributed_vqa=False; skip duplicated verifier work on non-main rank.")
        stage1_out.update(_empty_stage2_outputs(len(stage1_out["reference_names"])))
        return stage1_out

    preload_dict = kwargs["preload_dict"]
    caption_lookup = _read_lookup_csv(preload_dict["captions"], value_key="generated_text")
    path_lookup = _read_lookup_csv(preload_dict["img_paths"], value_key="image_path")

    verifier_mode = str(getattr(args, "verifier_candidate_source", "merged_only")).lower()
    merged_top_names = [list(names[:rerank_pool_size]) for names in stage1_out["stage1_pool_names"]]

    if verifier_mode == "merged_twice":
        rerank_candidates1_names = merged_top_names
        rerank_candidates2_names = [list(x) for x in merged_top_names]
    elif verifier_mode == "merged_plus_i2i":
        rerank_candidates1_names = merged_top_names
        rerank_candidates2_names = [
            list(x[:rerank_pool_size]) if hasattr(x, "tolist") else list(x[:rerank_pool_size])
            for x in stage1_out["img_sorted_index_names"]
        ]
    else:
        rerank_candidates1_names = merged_top_names
        rerank_candidates2_names = [[] for _ in merged_top_names]

    txt_top_captions, txt_top_img_paths = _build_candidate_side_info(
        rerank_candidates1_names, caption_lookup, path_lookup
    )
    img_top_captions, img_top_img_paths = _build_candidate_side_info(
        rerank_candidates2_names, caption_lookup, path_lookup
    )

    txt_check_index = [True for _ in range(len(stage1_out["reference_names"]))]
    img_check_index = [True for _ in range(len(stage1_out["reference_names"]))]
    ref_img_paths = _build_ref_img_paths(stage1_out["reference_names"], path_lookup)

    (
        candidates1,
        candidates2,
        ranks1,
        ranks2,
        pseudo_targets1,
        confidences1,
        pseudo_targets2,
        confidences2,
        txt_check_index,
        img_check_index,
    ) = base_utils.get_pseudo_targets(
        Check_LLM_model_name=kwargs["Check_LLM_model_name"],
        openai_key=kwargs["openai_key"],
        dataset_path=kwargs["dataset_path"],
        task=kwargs["task"],
        loop=0,
        reference_names=stage1_out["reference_names"],
        model_name=kwargs["VQA_LLM_model_name"],
        txt_top_captions=txt_top_captions,
        img_top_captions=img_top_captions,
        txt_top_img_paths=txt_top_img_paths,
        img_top_img_paths=img_top_img_paths,
        all_captions=stage1_out["start_captions"],
        ref_img_paths=ref_img_paths,
        relative_captions=stage1_out["instructions"],
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
        device=kwargs["device"],
    )

    verifier_score_maps1 = _build_verifier_score_maps(rerank_candidates1_names, candidates1)
    verifier_score_maps2 = _build_verifier_score_maps(rerank_candidates2_names, candidates2)

    stage1_out.update(
        {
            # Use gallery image names as candidate ids so stage-2 reranking actually
            # matches the stage-1 pool. Raw VQA outputs are kept separately for debugging.
            "candidates1": rerank_candidates1_names,
            "candidates2": rerank_candidates2_names,
            "ranks1": verifier_score_maps1,
            "ranks2": verifier_score_maps2,
            "raw_candidates1": candidates1,
            "raw_candidates2": candidates2,
            "pseudo_targets1": pseudo_targets1,
            "confidences1": confidences1,
            "pseudo_targets2": pseudo_targets2,
            "confidences2": confidences2,
            "txt_check_index": txt_check_index,
            "img_check_index": img_check_index,
            "rerank_candidates1_names": rerank_candidates1_names,
            "rerank_candidates2_names": rerank_candidates2_names,
        }
    )
    return stage1_out
