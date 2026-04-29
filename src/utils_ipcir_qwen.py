from __future__ import annotations

import copy
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



def _coerce_mode_list(value: Any) -> List[str]:
    """Parse image_generation_modes from JSON config or CLI."""
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        raw = []
        for item in value:
            if isinstance(item, str):
                raw.extend(item.replace(",", " ").split())
            else:
                raw.append(str(item))
    else:
        raw = [str(value)]
    valid = {"instruction_only", "instruction_plus_target", "target_only"}
    out: List[str] = []
    seen = set()
    for mode in raw:
        mode = str(mode).strip()
        if not mode:
            continue
        if mode not in valid:
            raise ValueError(f"Unsupported image generation mode in fusion: {mode!r}. Valid modes: {sorted(valid)}")
        if mode not in seen:
            seen.add(mode)
            out.append(mode)
    return out


def _image_fusion_enabled(args: Any) -> bool:
    mode = str(getattr(args, "image_fusion_mode", "none") or "none").lower()
    return mode not in {"", "none", "single", "off", "false", "0"}


def _canonical_fusion_mode(args: Any) -> str:
    mode = str(getattr(args, "image_fusion_mode", "none") or "none").lower()
    if mode in {"weighted", "weighted_avg", "wavg"}:
        return "weighted"
    if mode in {"avg", "mean", "average"}:
        return "avg"
    if mode in {"none", "single", "off", "false", "0", ""}:
        return "none"
    raise ValueError(f"Unsupported image_fusion_mode={mode!r}; use none, avg, or weighted.")


def _parse_fusion_weights(args: Any, modes: Sequence[str]) -> Dict[str, float]:
    """Parse weights for weighted image-feature fusion."""
    default = {mode: 1.0 for mode in modes}
    raw = getattr(args, "image_fusion_weights", None)
    if raw is None or raw == "":
        return default
    if isinstance(raw, dict):
        for key, value in raw.items():
            key = str(key).strip()
            if key in default:
                default[key] = float(value)
        return default
    if isinstance(raw, str):
        items = raw.replace(",", " ").split()
    elif isinstance(raw, (list, tuple, set)):
        items = []
        for item in raw:
            if isinstance(item, str):
                items.extend(item.replace(",", " ").split())
            elif isinstance(item, dict):
                for key, value in item.items():
                    if str(key) in default:
                        default[str(key)] = float(value)
    else:
        return default
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key in default:
            default[key] = float(value)
    return default


def _infer_edit_root_from_kwargs(kwargs: Dict[str, Any]) -> str:
    args = kwargs["args"]
    edit_root = getattr(args, "edit_img_dir", None)
    if edit_root:
        return str(edit_root)
    current = str(kwargs.get("edit_img_dir", ""))
    if current:
        return os.path.dirname(current)
    return os.path.join(kwargs["dataset_path"], "Edited_Images")


def _make_mode_preload_dict(kwargs: Dict[str, Any], mode: str) -> Dict[str, Any]:
    args = kwargs["args"]
    mode_tag = stage1_image_mode_tag(mode)
    preload_dict = dict(kwargs.get("preload_dict", {}))
    if preload_dict.get("edit_images", None) is not None:
        edit_meta_file = str(getattr(args, "preload_edited_images_file", "edited_images.pkl"))
        meta_root, meta_ext = os.path.splitext(edit_meta_file)
        if meta_ext == "":
            meta_ext = ".pkl"

        meta_dir = os.path.join(kwargs["dataset_path"], "preload", "edited_images")
        roots = [meta_root]
        # Practical fallback: previous single-branch runs may have used
        # CIRCO_edited_images.pkl, while target-only T2I experiments may have used
        # CIRCO_edited_images_t2i.pkl. Try both root namespaces before regenerating.
        if meta_root.endswith("_t2i"):
            roots.append(meta_root[: -len("_t2i")])
        else:
            roots.append(meta_root + "_t2i")

        candidates = []
        for root in roots:
            path = os.path.join(meta_dir, f"{root}_{mode_tag}{meta_ext}")
            if path not in candidates:
                candidates.append(path)

        chosen = candidates[0]
        for path in candidates:
            if os.path.exists(path):
                chosen = path
                break
        preload_dict["edit_images"] = chosen
    return preload_dict


def _make_mode_kwargs(kwargs: Dict[str, Any], mode: str) -> Dict[str, Any]:
    mode_tag = stage1_image_mode_tag(mode)
    mode_kwargs = dict(kwargs)
    mode_args = copy.copy(kwargs["args"])
    setattr(mode_args, "image_generation_mode", mode)
    setattr(mode_args, "stage_mode", "initial_only")
    mode_kwargs["args"] = mode_args
    mode_kwargs["edit_img_dir"] = os.path.join(_infer_edit_root_from_kwargs(kwargs), mode_tag)
    mode_kwargs["preload_dict"] = _make_mode_preload_dict(kwargs, mode)
    mode_kwargs["save_outputs"] = False
    os.makedirs(mode_kwargs["edit_img_dir"], exist_ok=True)
    if mode_kwargs["preload_dict"].get("edit_images") is not None:
        os.makedirs(os.path.dirname(mode_kwargs["preload_dict"]["edit_images"]), exist_ok=True)
    return mode_kwargs


def _normalize_feature_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1)


def _fuse_proxy_image_features(features_by_mode: Dict[str, torch.Tensor], modes: Sequence[str], fusion_mode: str, weights: Dict[str, float]) -> torch.Tensor:
    """Fuse multiple proxy-image CLIP embeddings.

    Every branch is L2-normalized before averaging, and the fused vector is
    L2-normalized again. This keeps cosine retrieval and IP-CIR behavior stable.
    """
    if not modes:
        raise ValueError("No image generation modes were provided for fusion.")
    ref_shape = None
    normed = []
    used_weights = []
    for mode in modes:
        if mode not in features_by_mode:
            raise KeyError(f"Missing predicted_img_features for mode={mode!r}")
        feat = features_by_mode[mode]
        if ref_shape is None:
            ref_shape = tuple(feat.shape)
        elif tuple(feat.shape) != ref_shape:
            raise ValueError(f"Feature shape mismatch for {mode}: got {tuple(feat.shape)}, expected {ref_shape}")
        normed.append(_normalize_feature_rows(feat))
        used_weights.append(float(weights.get(mode, 1.0)))
    stack = torch.stack(normed, dim=0)
    if fusion_mode == "avg":
        fused = stack.mean(dim=0)
    elif fusion_mode == "weighted":
        w = torch.tensor(used_weights, dtype=stack.dtype, device=stack.device).view(-1, 1, 1)
        fused = (stack * w).sum(dim=0) / torch.clamp(w.sum(), min=1e-8)
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
    return _normalize_feature_rows(fused)



def _compute_i2i_metrics_for_proxy(
    kwargs: Dict[str, Any],
    stage1_out: Dict[str, Any],
    features: torch.Tensor,
    tag: str,
    save_outputs: bool = True,
) -> Tuple[Any, Any]:
    """Evaluate a proxy-image feature matrix as one I2I branch.

    This intentionally mirrors eval_image_query_retrieval.py: features are passed
    directly to the dataset-specific compute_results function with ways=tag.
    """
    compute_fn = kwargs.get("compute_results_function", None)
    if compute_fn is None:
        return None, None
    metric_kwargs = dict(kwargs)
    metric_kwargs.update(stage1_out)
    metric_kwargs.update(
        {
            "predicted_features": features,
            "loop": 0,
            "ways": tag,
            "clip": kwargs.get("clip", getattr(kwargs.get("args"), "clip", None)),
            "save_outputs": bool(save_outputs and _is_main_process()),
        }
    )
    return compute_fn(**metric_kwargs)


def _query_id_to_filename_value(qid: Any) -> str:
    if isinstance(qid, torch.Tensor):
        qid = qid.detach().cpu().item()
    try:
        if float(qid).is_integer():
            return str(int(qid))
    except Exception:
        pass
    return _to_str(qid)


def _expected_mode_image_paths(kwargs: Dict[str, Any], stage1_out: Dict[str, Any], mode: str) -> List[str]:
    mode_dir = os.path.join(_infer_edit_root_from_kwargs(kwargs), stage1_image_mode_tag(mode))
    refs = list(stage1_out.get("reference_names", []))
    qids = list(stage1_out.get("query_ids", []))
    paths: List[str] = []
    for i, ref in enumerate(refs):
        qid = qids[i] if i < len(qids) else i
        paths.append(os.path.join(mode_dir, f"{_to_str(ref)}_edited_{_query_id_to_filename_value(qid)}.png"))
    return paths


def _load_mode_image_paths_from_metadata(kwargs: Dict[str, Any], stage1_out: Dict[str, Any], mode: str) -> List[str]:
    mode_preload = _make_mode_preload_dict(kwargs, mode)
    meta_path = mode_preload.get("edit_images")
    expected_len = len(stage1_out.get("reference_names", []))

    candidates: List[str] = []
    if meta_path:
        candidates.append(str(meta_path))

    args = kwargs["args"]
    meta_dir = os.path.join(kwargs["dataset_path"], "preload", "edited_images")
    root, ext = os.path.splitext(str(getattr(args, "preload_edited_images_file", "edited_images.pkl")))
    if ext == "":
        ext = ".pkl"
    tag = stage1_image_mode_tag(mode)
    extra_roots = [root, root + "_t2i"]
    if root.endswith("_t2i"):
        extra_roots.append(root[: -len("_t2i")])
    for r in extra_roots:
        cand = os.path.join(meta_dir, f"{r}_{tag}{ext}")
        if cand not in candidates:
            candidates.append(cand)

    for path in candidates:
        if path and os.path.exists(path):
            try:
                import pickle
                data = pickle.load(open(path, "rb"))
                img_paths = data.get("all_edit_img_paths", None)
                if isinstance(img_paths, list) and len(img_paths) == expected_len:
                    missing = [p for p in img_paths if not isinstance(p, str) or not os.path.exists(p)]
                    if not missing:
                        print(f"[Image Fusion] Loaded image paths for mode={mode} from metadata: {path}")
                        return img_paths
                    print(f"[Image Fusion] Metadata exists but has {len(missing)} missing images for mode={mode}: {missing[:3]}")
                else:
                    print(
                        f"[Image Fusion] Invalid metadata length for mode={mode}: {path}; "
                        f"expected {expected_len}, got {0 if img_paths is None else len(img_paths)}"
                    )
            except Exception as e:
                print(f"[Image Fusion] Failed to read metadata for mode={mode}: {path}; error={e}")

    fallback_paths = _expected_mode_image_paths(kwargs, stage1_out, mode)
    missing = [p for p in fallback_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"[Image Fusion] Missing generated images for mode={mode}. "
            f"Tried metadata candidates={candidates}. First missing files={missing[:5]}. "
            f"Please run the branch once with image_generation_mode={mode}, or check edit_img_dir/preload_edited_images_file."
        )
    print(f"[Image Fusion] Using folder fallback image paths for mode={mode}: {os.path.dirname(fallback_paths[0])}")
    return fallback_paths


@torch.no_grad()
def _encode_generated_image_paths(kwargs: Dict[str, Any], img_paths: Sequence[str], mode: str) -> torch.Tensor:
    if not img_paths:
        raise ValueError(f"No image paths provided for mode={mode}")
    preprocess = kwargs.get("preprocess", None)
    if preprocess is None:
        preprocess = kwargs.get("processor", None)
    if preprocess is None:
        raise ValueError("Cannot encode generated images: missing preprocess/processor in kwargs.")
    dataset = base_utils.datasets.EditedImageDataset(list(img_paths), preprocess)
    feats, _, _, _ = base_utils.extract_image_features(
        device=kwargs["device"],
        args=kwargs["args"],
        dataset=dataset,
        clip_model=kwargs["clip_model"],
        batch_size=32,
        num_workers=4,
        preload=None,
    )
    return _normalize_feature_rows(feats)


def _save_top_rank_for_labels(
    kwargs: Dict[str, Any],
    stage1_out: Dict[str, Any],
    tag: str,
    labels: Any,
    topk: int = 50,
) -> Optional[str]:
    if not _is_main_process() or labels is None:
        return None
    records = _build_stage_records(stage1_out, labels, topk=topk)
    return _save_record_artifact(
        dataset_path=kwargs["dataset_path"],
        task=kwargs["task"],
        clip=kwargs["clip"],
        dataset=kwargs["dataset_name"],
        tag=tag,
        records=_gather_records(records),
    )


def _save_text_top_rank_artifact(kwargs: Dict[str, Any], stage1_out: Dict[str, Any]) -> Dict[str, Optional[str]]:
    if not _is_main_process():
        return {"t2i_rank_artifact_path": None}
    rankings = _safe_get_stage_sequence(stage1_out, "txt_sorted_index_names")
    if not rankings:
        return {"t2i_rank_artifact_path": None}
    records = compute_results_ipcir_qwen.build_top_rank_records(
        rankings=rankings,
        query_ids=stage1_out.get("query_ids"),
        reference_names=stage1_out.get("reference_names"),
        target_names=stage1_out.get("target_names"),
        topk=50,
    )
    path = _save_record_artifact(
        dataset_path=kwargs["dataset_path"],
        task=kwargs["task"],
        clip=kwargs["clip"],
        dataset=kwargs["dataset_name"],
        tag="t2i_loop_0",
        records=records,
    )
    return {"t2i_rank_artifact_path": path}


def _run_stage1_with_optional_image_fusion(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    args = kwargs["args"]
    requested_stage_mode = getattr(args, "stage_mode", "initial_only")
    fusion_mode = _canonical_fusion_mode(args)

    if fusion_mode == "none":
        setattr(args, "stage_mode", "initial_only")
        stage1_out = _run_base_stage1_with_correct_target_only_dispatch(kwargs)
        setattr(args, "stage_mode", requested_stage_mode)
        stage1_out["image_fusion_mode"] = "none"
        return stage1_out

    modes = _coerce_mode_list(getattr(args, "image_generation_modes", None))
    if len(modes) == 0:
        modes = [str(getattr(args, "image_generation_mode", "instruction_plus_target"))]
    if len(modes) < 2:
        print(f"[Image Fusion] only one mode provided ({modes}); fallback to single-branch behavior.")
        setattr(args, "stage_mode", "initial_only")
        stage1_out = _run_base_stage1_with_correct_target_only_dispatch(kwargs)
        setattr(args, "stage_mode", requested_stage_mode)
        stage1_out["image_fusion_mode"] = "none"
        return stage1_out

    base_mode = str(getattr(args, "image_generation_mode", modes[0]))
    if base_mode not in modes:
        base_mode = modes[0]
        setattr(args, "image_generation_mode", base_mode)

    weights = _parse_fusion_weights(args, modes)
    print(f"[Image Fusion] enabled: mode={fusion_mode}, branches={modes}, base_mode={base_mode}, weights={weights}")

    original_mode = getattr(args, "image_generation_mode", base_mode)
    setattr(args, "image_generation_mode", base_mode)
    setattr(args, "stage_mode", "initial_only")
    stage1_out = _run_base_stage1_with_correct_target_only_dispatch(kwargs)
    setattr(args, "image_generation_mode", original_mode)
    setattr(args, "stage_mode", requested_stage_mode)

    features_by_mode: Dict[str, torch.Tensor] = {}
    paths_by_mode: Dict[str, List[str]] = {}
    metrics_by_mode: Dict[str, Any] = {}
    labels_by_mode: Dict[str, Any] = {}
    rank_artifacts_by_mode: Dict[str, Optional[str]] = {}

    for mode in modes:
        mode_tag = stage1_image_mode_tag(mode)
        branch_tag = f"i2i_{mode_tag}"
        print(f"[Image Fusion] Evaluating raw image branch: mode={mode}, tag={branch_tag}")

        if mode == base_mode:
            img_paths = list(stage1_out.get("all_edit_img_paths", []))
            feats = _normalize_feature_rows(stage1_out["predicted_img_features"])
        else:
            img_paths = _load_mode_image_paths_from_metadata(kwargs, stage1_out, mode)
            feats = _encode_generated_image_paths(kwargs, img_paths, mode)

        features_by_mode[mode] = feats
        paths_by_mode[mode] = img_paths
        metrics, labels = _compute_i2i_metrics_for_proxy(
            kwargs=kwargs,
            stage1_out=stage1_out,
            features=feats,
            tag=branch_tag,
            save_outputs=True,
        )
        metrics_by_mode[mode] = metrics
        labels_by_mode[mode] = labels
        rank_artifacts_by_mode[mode] = _save_top_rank_for_labels(kwargs, stage1_out, f"{branch_tag}_loop_0", labels, topk=50)
        if metrics is not None:
            print(f"[Image Fusion] Raw branch metrics [{branch_tag}]: {metrics}")

    fused_features = _fuse_proxy_image_features(features_by_mode, modes, fusion_mode, weights)
    fused_tag = f"i2i_fused_{fusion_mode}"
    print(f"[Image Fusion] Evaluating fused image branch: tag={fused_tag}")
    fused_metrics, fused_labels = _compute_i2i_metrics_for_proxy(
        kwargs=kwargs,
        stage1_out=stage1_out,
        features=fused_features,
        tag=fused_tag,
        save_outputs=True,
    )
    fused_rank_artifact = _save_top_rank_for_labels(kwargs, stage1_out, f"{fused_tag}_loop_0", fused_labels, topk=50)
    if fused_metrics is not None:
        print(f"[Image Fusion] Fused branch metrics [{fused_tag}]: {fused_metrics}")

    stage1_out["predicted_img_features_by_mode"] = features_by_mode
    stage1_out["all_edit_img_paths_by_mode"] = paths_by_mode
    stage1_out["img_output_metrics_by_mode"] = metrics_by_mode
    stage1_out["img_sorted_index_names_by_mode"] = labels_by_mode
    stage1_out["i2i_rank_artifact_paths_by_mode"] = rank_artifacts_by_mode

    stage1_out["predicted_img_features_single_branch"] = stage1_out.get("predicted_img_features")
    stage1_out["img_output_metrics_single_branch"] = stage1_out.get("img_output_metrics")
    stage1_out["img_sorted_index_names_single_branch"] = stage1_out.get("img_sorted_index_names")

    stage1_out["predicted_img_features"] = fused_features
    if fused_metrics is not None:
        stage1_out["img_output_metrics"] = fused_metrics
    if fused_labels is not None:
        stage1_out["img_sorted_index_names"] = fused_labels

    stage1_out["image_fusion_mode"] = fusion_mode
    stage1_out["image_generation_modes"] = list(modes)
    stage1_out["image_fusion_weights"] = {m: float(weights.get(m, 1.0)) for m in modes}
    stage1_out["image_fusion_tag"] = fused_tag
    stage1_out["image_fused_i2i_rank_artifact_path"] = fused_rank_artifact
    return stage1_out

def generate_editimg_caption_iteration(**kwargs):
    """IP-CIR wrapper.

    The only new behavior is optional feature-level fusion of multiple generated-image
    branches before IP-CIR. Downstream IP-CIR pooling and Qwen reranking are unchanged.
    """
    args = kwargs["args"]
    requested_stage_mode = getattr(args, "stage_mode", "initial_only")

    stage1_out = _run_stage1_with_optional_image_fusion(kwargs)
    setattr(args, "stage_mode", requested_stage_mode)

    if _canonical_fusion_mode(args) == "none":
        raw_rank_paths = _save_raw_branch_top_rank_artifacts(kwargs, stage1_out)
    else:
        raw_rank_paths = _save_text_top_rank_artifact(kwargs, stage1_out)
    stage1_out.update(raw_rank_paths)

    query_caption_features = base_utils.text_encoding(
        device=kwargs["device"],
        clip_model=kwargs["clip_model"],
        input_captions=stage1_out["start_captions"],
        batch_size=32,
        mode=kwargs["retrieval"],
    )
    query_caption_features = torch.nn.functional.normalize(query_caption_features.float(), dim=-1)

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

    stage1_metrics, stage1_labels = _compute_stage1_metrics_and_labels(kwargs["dataset_name"], stage1_out, kwargs)
    if stage1_labels is None:
        stage1_labels = stage1_out["stage1_pool_names"]

    stage1_out["stage1_output_metrics"] = stage1_metrics
    stage1_out["stage1_output_labels"] = stage1_labels
    stage1_out["stage1_metric_artifact_path"] = None
    stage1_out["stage1_rank_artifact_path"] = None

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
