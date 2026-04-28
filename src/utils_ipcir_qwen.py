from __future__ import annotations

import csv
import datetime
import functools
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

import compute_results_ipcir_qwen
import utils as base_utils
from stage1_pooling import build_ipcir_stage1_pool


def _to_str(x: Any) -> str:
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


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_time() -> str:
    return datetime.datetime.now().strftime("%Y.%m.%d-%H_%M_%S")


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _task_dir(dataset_path: str, task: str) -> str:
    return os.path.join(dataset_path, "task", task)


def stage1_image_mode_tag(image_generation_mode: str) -> str:
    """Return the cache/folder tag used by stage-1 generated images.

    The old code used "target_only" for a mode that was still image-conditioned.
    The corrected target_only branch is pure text-to-image, so it gets a separate
    tag to avoid silently reusing old reference-conditioned caches.
    """
    if str(image_generation_mode) == "target_only":
        return "target_only_t2i"
    if hasattr(base_utils, "_sanitize_tag"):
        return base_utils._sanitize_tag(str(image_generation_mode))
    return str(image_generation_mode).replace("/", "-").replace(" ", "_")


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
        # Raw verifier records are often (rank_index, candidate_path, confidence).
        # Prefer the last numeric field.
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
            score_map: Dict[str, float] = {}
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
        score_map: Dict[str, float] = {}
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


def _dedup_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in records:
        key = r.get("query_id") or r.get("image_index") or len(out)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _gather_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Artifact-only helper.

    Do not call dist.all_gather_object here. These records are only for
    human-checkable debug JSON files. After long VQA reranking, different ranks
    may enter/skip artifact saving at different times, and a collective here can
    trigger NCCL ALLGATHER watchdog timeout. Rank0 writes its local artifacts.
    """
    return _dedup_records(records)


def _save_metric_artifact(dataset_path: str, task: str, clip: str, dataset: str, tag: str, metrics: Dict[str, float]) -> str:
    save_path = os.path.join(_task_dir(dataset_path, task), f"result_{clip}_{dataset}_{tag}_{_get_time()}.json")
    _write_json(save_path, metrics)
    print(f"[Artifact] saved metrics to: {save_path}")
    return save_path


def _save_record_artifact(dataset_path: str, task: str, clip: str, dataset: str, tag: str, records: Sequence[Dict[str, Any]]) -> str:
    save_path = os.path.join(_task_dir(dataset_path, task), f"top_rank_{clip}_{dataset}_{tag}_{_get_time()}.json")
    _write_json(save_path, list(records))
    print(f"[Artifact] saved top-rank records to: {save_path}")
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


def _build_stage_records(stage1_out: Dict[str, Any], rankings: Sequence[Sequence[str]], topk: int = 50) -> List[Dict[str, Any]]:
    return compute_results_ipcir_qwen.build_top_rank_records(
        rankings=rankings,
        query_ids=stage1_out.get("query_ids"),
        reference_names=stage1_out.get("reference_names"),
        target_names=stage1_out.get("target_names"),
        topk=topk,
    )


def _safe_get_stage_sequence(stage1_out: Dict[str, Any], key: str) -> List[List[str]]:
    values = stage1_out.get(key)
    if values is None:
        return []
    out: List[List[str]] = []
    for item in values:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().tolist()
        if hasattr(item, "tolist") and not isinstance(item, (list, tuple)):
            item = item.tolist()
        if isinstance(item, (list, tuple)):
            out.append([_to_str(x) for x in item])
        else:
            out.append([_to_str(item)])
    return out


def _unique_keep_order(names: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for name in names:
        s = _to_str(name)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    elif hasattr(x, "tolist") and not isinstance(x, (list, tuple, set, str, bytes)):
        try:
            x = x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _build_submission_dict(rankings: Sequence[Sequence[Any]], query_ids: Sequence[Any], topk: int = 50) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for qid, rank in zip(query_ids, rankings):
        out[_to_str(qid)] = _unique_keep_order(rank)[: int(topk)]
    return out


def _filter_cirr_reference(rankings: Sequence[Sequence[str]], reference_names: Sequence[Any]) -> List[List[str]]:
    if not reference_names or len(reference_names) != len(rankings):
        return [[_to_str(x) for x in rank] for rank in rankings]
    filtered: List[List[str]] = []
    for ref_name, rank in zip(reference_names, rankings):
        ref = _to_str(ref_name)
        filtered.append([_to_str(x) for x in rank if _to_str(x) != ref])
    return filtered


def _save_raw_branch_top_rank_artifacts(kwargs: Dict[str, Any], stage1_out: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Save inspectable top-50 records for raw t2i/i2i rankings on every split."""
    if not _is_main_process():
        return {}

    saved: Dict[str, Optional[str]] = {"t2i_rank_artifact_path": None, "i2i_rank_artifact_path": None}
    reference_names = stage1_out.get("reference_names")
    target_names = stage1_out.get("target_names")
    query_ids = stage1_out.get("query_ids")

    for branch, key in (("t2i", "txt_sorted_index_names"), ("i2i", "img_sorted_index_names")):
        rankings = _safe_get_stage_sequence(stage1_out, key)
        if not rankings:
            print(f"[Artifact] skip raw {branch} top-rank artifact because {key} is missing.")
            continue
        records = compute_results_ipcir_qwen.build_top_rank_records(
            rankings=rankings,
            query_ids=query_ids,
            reference_names=reference_names,
            target_names=target_names,
            topk=50,
        )
        path = _save_record_artifact(
            dataset_path=kwargs["dataset_path"],
            task=kwargs["task"],
            clip=kwargs["clip"],
            dataset=kwargs["dataset_name"],
            tag=f"{branch}_loop_0",
            records=records,
        )
        saved[f"{branch}_rank_artifact_path"] = path
    return saved


def _clip_prefix_for_submission(clip: Any) -> str:
    clip = "" if clip is None else str(clip).strip()
    return f"{clip}_" if clip else ""


def _query_id_to_key(qid: Any) -> str:
    """Match the original compute_results.py behavior for official submissions."""
    if isinstance(qid, torch.Tensor):
        qid = qid.detach().cpu().item()
    try:
        return str(int(qid))
    except Exception:
        return _to_str(qid)


def _target_only_t2i_dispatcher(bagel_editor: Any, args: Any):
    """Return a function with edit_image_no_think's signature but pure T2I behavior."""

    @functools.wraps(bagel_editor.edit_image_no_think)
    def _dispatch(_image_path: str, prompt: str, *unused_args: Any, **unused_kwargs: Any) -> Dict[str, Any]:
        if not hasattr(bagel_editor, "text_to_image_no_think"):
            raise AttributeError(
                "BAGEL editor has no text_to_image_no_think(...). "
                "Please replace src/bagel_inference.py with the T2I-enabled version."
            )
        return bagel_editor.text_to_image_no_think(
            prompt=prompt,
            image_size=int(getattr(args, "t2i_image_size", 1024)),
            cfg_text_scale=float(getattr(args, "t2i_cfg_text_scale", 4.0)),
            cfg_img_scale=float(getattr(args, "t2i_cfg_img_scale", 1.0)),
            cfg_interval=list(getattr(args, "t2i_cfg_interval", [0.0, 1.0])),
            timestep_shift=float(getattr(args, "t2i_timestep_shift", 3.0)),
            num_timesteps=int(getattr(args, "t2i_num_timesteps", 50)),
            cfg_renorm_min=float(getattr(args, "t2i_cfg_renorm_min", 0.0)),
            cfg_renorm_type=str(getattr(args, "t2i_cfg_renorm_type", "text_channel")),
        )

    return _dispatch


@contextmanager
def _patched_target_only_t2i_if_needed(args: Any, bagel_editor: Any):
    """Patch only the base stage-1 call when target_only should be pure T2I.

    utils_ipcir_qwen delegates the fast caption/image generation stage to
    base_utils.generate_editimg_caption_iteration(...). The original base utils
    always calls bagel_editor.edit_image_no_think(ref_img_path, prompt). For
    target_only, we temporarily reroute that call to text_to_image_no_think and
    also patch base_utils._sanitize_tag so metadata uses target_only_t2i.
    """
    image_generation_mode = getattr(args, "image_generation_mode", "instruction_plus_target")
    if image_generation_mode != "target_only":
        yield
        return

    old_edit_method = None
    if bagel_editor is not None and hasattr(bagel_editor, "edit_image_no_think"):
        old_edit_method = bagel_editor.edit_image_no_think
        bagel_editor.edit_image_no_think = _target_only_t2i_dispatcher(bagel_editor, args)

    old_sanitize = getattr(base_utils, "_sanitize_tag", None)

    def _sanitize_tag_t2i(value: Any) -> str:
        if str(value) == "target_only":
            return "target_only_t2i"
        if old_sanitize is not None:
            return old_sanitize(value)
        return str(value).replace("/", "-").replace(" ", "_")

    if old_sanitize is not None:
        base_utils._sanitize_tag = _sanitize_tag_t2i

    try:
        print("[Stage-1 Image] image_generation_mode=target_only -> pure BAGEL text_to_image_no_think; reference image is ignored.")
        yield
    finally:
        if old_edit_method is not None:
            bagel_editor.edit_image_no_think = old_edit_method
        if old_sanitize is not None:
            base_utils._sanitize_tag = old_sanitize


def _run_base_stage1_with_correct_target_only_dispatch(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    args = kwargs["args"]
    bagel_editor = kwargs.get("bagel_editor", None)
    with _patched_target_only_t2i_if_needed(args=args, bagel_editor=bagel_editor):
        return base_utils.generate_editimg_caption_iteration(**kwargs)


def generate_editimg_caption_iteration(**kwargs):
    """IP-CIR wrapper.

    Fixed behavior:
    1. Run fast t2i/i2i stage through the original utility.
    2. For image_generation_mode == target_only, force true text-to-image generation.
    3. Build the IP-CIR merged pool for every split/dataset.
    4. Save merged submissions for CIRR/CIRCO test.
    5. Save merged top-rank records for train/val/test, always with top-50 names.
    6. Use merged top-100 candidates for Qwen verifier by default.
    """
    args = kwargs["args"]
    requested_stage_mode = getattr(args, "stage_mode", "initial_only")

    # Force original pipeline to stop after fast t2i/i2i, then restore the requested mode.
    setattr(args, "stage_mode", "initial_only")
    stage1_out = _run_base_stage1_with_correct_target_only_dispatch(kwargs)
    setattr(args, "stage_mode", requested_stage_mode)

    # Save inspectable top-50 records for raw t2i/i2i. Do NOT rewrite official raw
    # t2i/i2i submission files here: base_utils already saves them, and rewriting
    # them can introduce inconsistent subset formatting.
    raw_rank_paths = _save_raw_branch_top_rank_artifacts(kwargs, stage1_out)
    stage1_out.update(raw_rank_paths)

    query_caption_features = base_utils.text_encoding(
        device=kwargs["device"],
        clip_model=kwargs["clip_model"],
        input_captions=stage1_out["start_captions"],
        batch_size=32,
        mode=kwargs["retrieval"],
    )
    query_caption_features = torch.nn.functional.normalize(query_caption_features.float(), dim=-1)

    # User-requested default: rerank merged top-100 candidates.
    rerank_pool_size = int(getattr(args, "rerank_pool_size", getattr(args, "topk_for_vqa", 100)))
    rerank_pool_size = max(rerank_pool_size, 100)
    prior_topk = int(getattr(args, "ipcir_prior_topk", max(rerank_pool_size, 100)))
    prior_topk = max(prior_topk, rerank_pool_size, 100)
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

    # Compute merged metrics and labels. For CIRR/CIRCO test, this also writes merged submission files.
    stage1_metrics, stage1_labels = _compute_stage1_metrics_and_labels(kwargs["dataset_name"], stage1_out, kwargs)
    if stage1_labels is None:
        stage1_labels = stage1_out["stage1_pool_names"]

    stage1_out["stage1_output_metrics"] = stage1_metrics
    stage1_out["stage1_output_labels"] = stage1_labels
    stage1_out["stage1_metric_artifact_path"] = None
    stage1_out["stage1_rank_artifact_path"] = None

    # Save top-rank debug records for every split, including test. Always save top-50.
    local_records = _build_stage_records(stage1_out, stage1_labels, topk=50)
    gathered_records = _gather_records(local_records)
    if _is_main_process():
        if stage1_metrics is not None:
            stage1_out["stage1_metric_artifact_path"] = _save_metric_artifact(
                dataset_path=kwargs["dataset_path"],
                task=kwargs["task"],
                clip=kwargs["clip"],
                dataset=kwargs["dataset_name"],
                tag="merged_loop_0",
                metrics=stage1_metrics,
            )
        stage1_out["stage1_rank_artifact_path"] = _save_record_artifact(
            dataset_path=kwargs["dataset_path"],
            task=kwargs["task"],
            clip=kwargs["clip"],
            dataset=kwargs["dataset_name"],
            tag="merged_loop_0",
            records=gathered_records,
        )

    if requested_stage_mode == "initial_only":
        return stage1_out
    if requested_stage_mode != "qwen_fusion":
        raise ValueError(f"Unsupported stage_mode in IP-CIR wrapper: {requested_stage_mode}")

    distributed_vqa = bool(getattr(args, "distributed_vqa", False))
    if _dist_is_ready() and (not distributed_vqa) and (not _is_main_process()):
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
        rerank_candidates2_names = [list(x[:rerank_pool_size]) for x in stage1_out["img_sorted_index_names"]]
    else:
        # Default: only score the IP-CIR merged candidates once.
        rerank_candidates1_names = merged_top_names
        rerank_candidates2_names = [[] for _ in merged_top_names]

    txt_top_captions, txt_top_img_paths = _build_candidate_side_info(rerank_candidates1_names, caption_lookup, path_lookup)
    img_top_captions, img_top_img_paths = _build_candidate_side_info(rerank_candidates2_names, caption_lookup, path_lookup)
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
            # Use gallery image names as candidate ids so stage-2 reranking matches stage-1 pool.
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
