from __future__ import annotations

import copy
import csv
import datetime
import functools
import gc
import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

import compute_results_ipcir_qwen
import utils as base_utils
from stage1_pooling import build_ipcir_stage1_pool
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _to_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


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


def _write_json_atomic(path: str, data: Any) -> None:
    """Atomically write JSON for distributed readers.

    Rank0 writes to <path>.tmp first, fsyncs it, then os.replace(...) to
    <path>.  A small <path>.done sentinel is written last. Non-main ranks
    should wait for the .done file rather than just checking that <path>
    exists. This avoids JSONDecodeError from reading half-written files.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    done_path = f"{path}.done"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("done\n")
        f.flush()
        os.fsync(f.fileno())


def _write_json_atomic_no_done(path: str, data: Any) -> None:
    """Atomic JSON write without a .done sentinel for normal artifacts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _multitext_t2i_dir(dataset_path: str, task: str) -> str:
    return os.path.join(_task_dir(dataset_path, task), "multitext_t2i")


def _ipmerge_dir(dataset_path: str, task: str) -> str:
    return os.path.join(_task_dir(dataset_path, task), "ipmerge")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _task_dir(dataset_path: str, task: str) -> str:
    return os.path.join(dataset_path, "task", task)


def _default_lambda(dataset_name: str) -> float:
    ds = (dataset_name or "").lower()
    if ds == "circo":
        return 0.3
    if ds == "cirr":
        return 0.0
    if "fashioniq" in ds:
        return 0.8
    return 0.3


def stage1_image_mode_tag(image_generation_mode: str) -> str:
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


def _get_value_by_name(lookup: Dict[str, str], name: Any) -> str:
    key = str(name).lstrip("0")
    return lookup.get(key, "")


def _normalize_feature_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def _safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


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


# -----------------------------------------------------------------------------
# Original target-only proxy image dispatch
# -----------------------------------------------------------------------------


def _target_only_t2i_dispatcher(bagel_editor: Any, args: Any):
    @functools.wraps(bagel_editor.edit_image_no_think)
    def _dispatch(_image_path: str, prompt: str, *unused_args: Any, **unused_kwargs: Any) -> Dict[str, Any]:
        if not hasattr(bagel_editor, "text_to_image_no_think"):
            raise AttributeError(
                "BAGEL editor has no text_to_image_no_think(...). "
                "Please replace src/bagel_inference.py with the multitext version."
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
        print("[Stage-1 Image] image_generation_mode=target_only -> pure BAGEL text_to_image_no_think.")
        yield
    finally:
        if old_edit_method is not None:
            bagel_editor.edit_image_no_think = old_edit_method
        if old_sanitize is not None:
            base_utils._sanitize_tag = old_sanitize


def _run_base_stage1(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    args = kwargs["args"]
    bagel_editor = kwargs.get("bagel_editor", None)
    with _patched_target_only_t2i_if_needed(args=args, bagel_editor=bagel_editor):
        return base_utils.generate_editimg_caption_iteration(**kwargs)


# -----------------------------------------------------------------------------
# Multi-text query generation
# -----------------------------------------------------------------------------


def _build_vision_direct_prompt(relative_caption: str) -> str:
    return (
        "You are given a reference image and an edit instruction.\n"
        "Describe the target image after applying the instruction to the reference image.\n"
        "Preserve unchanged visible details and apply only the requested modification.\n"
        "Return one concise target image description.\n\n"
        f"Instruction: {relative_caption}"
    )


def _build_vision_prompted_prompt(relative_caption: str) -> str:
    return (
        "You are a composed image retrieval assistant.\n\n"
        "Given a reference image and a modification instruction, generate a target image description for retrieval.\n"
        "Requirements:\n"
        "- preserve all visible attributes not mentioned in the instruction;\n"
        "- apply the instruction exactly;\n"
        "- avoid unsupported objects, colors, styles, brands, counts, or background changes;\n"
        "- be concise and visually grounded;\n"
        "- return only the final target image description, with no explanation.\n\n"
        f"Instruction: {relative_caption}"
    )


def _clean_generated_text(text: Any) -> str:
    s = _to_str(text).strip()
    # Strip common JSON wrappers if the model ignored the instruction.
    if s.startswith("{") and "target_text" in s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and obj.get("target_text"):
                return str(obj["target_text"]).strip()
        except Exception:
            pass
    s = s.replace("\n", " ").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def _get_multi_text_query_path(kwargs: Dict[str, Any]) -> str:
    args = kwargs["args"]
    explicit = str(getattr(args, "multi_text_queries_path", "") or "").strip()
    if explicit:
        return explicit
    dataset = str(kwargs["dataset_name"])
    split = str(kwargs.get("split", getattr(args, "split", "val")))
    filename = f"{dataset}_{split}_multi_text_queries.json"
    return os.path.join(_task_dir(kwargs["dataset_path"], kwargs["task"]), "modified_captions", filename)


def _wait_for_file(path: str, timeout_s: int = 7200, poll_s: int = 5) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for file: {path}")


def _simplify_multi_text_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return the compact query-cache schema.

    Keep this intentionally close to the original circo_modified_captions.json:
      {
        "image_index": "...",
        "modified_caption": "...",
        "modified_caption_img_direct": "...",
        "modified_caption_img_prompt": "..."
      }

    Older rich-format cache files are accepted and mapped into this compact form.
    """
    image_index = (
        item.get("image_index")
        or item.get("reference_name")
        or item.get("ref_name")
        or item.get("query_id")
        or ""
    )
    return {
        "image_index": _to_str(image_index),
        "modified_caption": _to_str(item.get("modified_caption", item.get("caption_query", ""))),
        "modified_caption_img_direct": _to_str(
            item.get("modified_caption_img_direct", item.get("vision_direct_query", ""))
        ),
        "modified_caption_img_prompt": _to_str(
            item.get("modified_caption_img_prompt", item.get("vision_prompted_query", ""))
        ),
    }


def _load_multi_text_items(path: str) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, dict) and "items" in data:
        raw_items = list(data["items"])
    elif isinstance(data, list):
        raw_items = list(data)
    else:
        raise ValueError(f"Unsupported multi-text query JSON format: {path}")
    items = [_simplify_multi_text_item(x) for x in raw_items]
    return items


def _save_multi_text_items(path: str, kwargs: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    # Deliberately save a compact list, not a large metadata-heavy object.
    # This keeps the file similar to circo_modified_captions.json and avoids
    # distributed readers parsing huge JSON payloads.
    compact_items = [_simplify_multi_text_item(x) for x in items]
    _write_json_atomic(path, compact_items)
    print(f"[MultiText] saved compact multi-text query cache: {path}", flush=True)

def _build_ref_img_paths(reference_names: Sequence[Any], path_lookup: Dict[str, str]) -> List[str]:
    return [_get_value_by_name(path_lookup, name) for name in reference_names]


def _get_caption_queries_from_stage1(stage1_out: Dict[str, Any]) -> List[str]:
    for key in ("modified_captions", "new_captions", "target_captions"):
        value = stage1_out.get(key)
        if isinstance(value, list) and value:
            return [_to_str(x) for x in value]
    # Fallback: if base stage did not expose text strings, use start captions.
    value = stage1_out.get("start_captions", [])
    return [_to_str(x) for x in value]


def _ensure_multi_text_queries(kwargs: Dict[str, Any], stage1_out: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate/load the compact three-query cache.

    Three query texts:
      1. modified_caption: original ref-caption + mod-text target caption.
      2. modified_caption_img_direct: ref-image + mod-text target caption.
      3. modified_caption_img_prompt: ref-image + mod-text + stricter retrieval prompt target caption.
    """
    args = kwargs["args"]
    path = _get_multi_text_query_path(kwargs)
    print(f"[MultiText] expected compact query cache path: {path}", flush=True)
    if os.path.exists(path):
        print(f"[MultiText] loading compact query cache: {path}", flush=True)
        items = _load_multi_text_items(path)
        print(f"[MultiText] loaded {len(items)} compact multi-text query records.", flush=True)
        return path, items

    if _dist_is_ready() and not _is_main_process():
        print(f"[MultiText] non-main rank waits for compact query cache: {path}.done", flush=True)
        _wait_for_file(path + ".done", timeout_s=int(getattr(args, "multi_text_query_cache_timeout_s", 7200)), poll_s=5)
        return path, _load_multi_text_items(path)

    bagel_editor = kwargs.get("bagel_editor", None)
    if bagel_editor is None:
        raise RuntimeError(
            "Multi-text query cache is missing and bagel_editor is None. "
            "The compact cache must be generated once by rank0, or BAGEL must be loaded. "
            f"Expected cache: {path}"
        )
    if not hasattr(bagel_editor, "generate_caption_from_image"):
        raise AttributeError(
            "bagel_editor.generate_caption_from_image is missing. "
            "Replace src/bagel_inference.py with the multitext version."
        )

    preload_dict = kwargs["preload_dict"]
    path_lookup = _read_lookup_csv(preload_dict["img_paths"], value_key="image_path")
    reference_names = list(stage1_out.get("reference_names", []))
    instructions = list(stage1_out.get("instructions", []))
    caption_queries = _get_caption_queries_from_stage1(stage1_out)
    ref_img_paths = _build_ref_img_paths(reference_names, path_lookup)

    max_think = int(getattr(args, "vision_caption_max_think_token_n", 1000))
    do_sample = bool(getattr(args, "vision_caption_do_sample", False))

    items: List[Dict[str, Any]] = []
    progress_iter = tqdm(
        enumerate(reference_names),
        total=len(reference_names),
        desc="[MultiText] generating compact vision text",
        unit="query",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
        disable=not _is_main_process(),
    )
    for i, ref_name in progress_iter:
        instruction = _to_str(instructions[i]) if i < len(instructions) else ""
        ref_img_path = ref_img_paths[i]
        if _is_main_process():
            progress_iter.set_postfix_str(f"ref={ref_name}", refresh=False)
        if not ref_img_path or not os.path.exists(ref_img_path):
            raise FileNotFoundError(f"Missing reference image path for {ref_name}: {ref_img_path}")

        direct_text = bagel_editor.generate_caption_from_image(
            image_path=ref_img_path,
            prompt=_build_vision_direct_prompt(instruction),
            max_think_token_n=max_think,
            do_sample=do_sample,
        )
        prompted_text = bagel_editor.generate_caption_from_image(
            image_path=ref_img_path,
            prompt=_build_vision_prompted_prompt(instruction),
            max_think_token_n=max_think,
            do_sample=do_sample,
        )
        items.append(
            {
                "image_index": _to_str(ref_name),
                "modified_caption": _to_str(caption_queries[i]) if i < len(caption_queries) else "",
                "modified_caption_img_direct": _clean_generated_text(direct_text),
                "modified_caption_img_prompt": _clean_generated_text(prompted_text),
            }
        )

    _save_multi_text_items(path, kwargs, items)
    if items:
        sample = items[0]
        print("[MultiText] sample compact query record:", json.dumps(sample, ensure_ascii=False)[:500], flush=True)
    return path, items




def _release_bagel_editor_from_kwargs(kwargs: Dict[str, Any], reason: str = "") -> None:
    """Release BAGEL as soon as generation is finished.

    In the multi-text path, rank0 may load BAGEL only to create
    *_multi_text_queries.json. If we keep the 7B model resident while the same
    rank enters CLIP text encoding, GPU0 can look like it is stuck at
    "Encoding captions..." or can OOM before any multi-text rank artifacts are
    saved. The base Experiment releases BAGEL only after this function returns,
    which is too late for this path.
    """
    bagel_editor = kwargs.get("bagel_editor", None)
    if bagel_editor is None:
        return
    rank = int(os.environ.get("RANK", "0"))
    msg = f"[BAGEL][Release] rank={rank} releasing BAGEL before CLIP text encoding"
    if reason:
        msg += f" ({reason})"
    print(msg, flush=True)
    try:
        if hasattr(bagel_editor, "model"):
            del bagel_editor.model
            print(f"[BAGEL][Release] rank={rank} deleted model", flush=True)
    except Exception as e:
        print(f"[BAGEL][Release] rank={rank} failed deleting model: {e}", flush=True)
    try:
        if hasattr(bagel_editor, "inferencer"):
            del bagel_editor.inferencer
            print(f"[BAGEL][Release] rank={rank} deleted inferencer", flush=True)
    except Exception as e:
        print(f"[BAGEL][Release] rank={rank} failed deleting inferencer: {e}", flush=True)
    kwargs["bagel_editor"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[BAGEL][Release] rank={rank} CUDA after release: allocated={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)

# -----------------------------------------------------------------------------
# Multi-text scoring / ranking
# -----------------------------------------------------------------------------


def _encode_texts(kwargs: Dict[str, Any], texts: Sequence[str], tag: str = "") -> torch.Tensor:
    args = kwargs["args"]
    batch_size = int(getattr(args, "multi_text_encoding_batch_size", 32))
    rank = int(os.environ.get("RANK", "0"))
    tag_msg = f" for {tag}" if tag else ""
    print(
        f"[MultiText][Encode] rank={rank} start CLIP text encoding{tag_msg}: "
        f"num_texts={len(texts)} batch_size={batch_size} device={kwargs['device']}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.inference_mode():
        feats = base_utils.text_encoding(
            device=kwargs["device"],
            clip_model=kwargs["clip_model"],
            input_captions=list(texts),
            batch_size=batch_size,
            mode=kwargs["retrieval"],
        )
    feats = _normalize_feature_rows(feats)
    print(f"[MultiText][Encode] rank={rank} finished CLIP text encoding{tag_msg}: shape={tuple(feats.shape)}", flush=True)
    return feats


def _topk_rank_from_scores(
    scores: torch.Tensor,
    index_names: Sequence[Any],
    topk: int,
    tag: str = "",
) -> List[List[str]]:
    """Return only top-k names per query.

    Previous versions used torch.argsort over the full gallery and then built
    full Python name lists. That is exactly where CIRCO looked "stuck": for
    every branch we sorted the entire gallery, then RRF iterated over full
    rankings in Python. For this experiment we only need topK artifacts and
    topK stage pools, so torch.topk is the correct primitive.
    """
    rank = int(os.environ.get("RANK", "0"))
    k = min(int(topk), int(scores.shape[-1]))
    tag_msg = f" for {tag}" if tag else ""
    print(
        f"[MultiText][TopK] rank={rank} start torch.topk{tag_msg}: "
        f"scores_shape={tuple(scores.shape)} k={k}",
        flush=True,
    )
    with torch.inference_mode():
        _, inds = torch.topk(scores.float(), k=k, dim=-1, largest=True, sorted=True)
    inds_list = inds.detach().cpu().tolist()
    names = [_to_str(x) for x in index_names]
    out = [[names[j] for j in row] for row in inds_list]
    print(f"[MultiText][TopK] rank={rank} finished torch.topk{tag_msg}", flush=True)
    return out


def _scores_from_features(features: torch.Tensor, index_features: torch.Tensor, tag: str = "") -> torch.Tensor:
    rank = int(os.environ.get("RANK", "0"))
    tag_msg = f" for {tag}" if tag else ""
    print(
        f"[MultiText][Score] rank={rank} start similarity{tag_msg}: "
        f"query_shape={tuple(features.shape)} gallery_shape={tuple(index_features.shape)}",
        flush=True,
    )
    device = features.device
    with torch.inference_mode():
        gallery = _normalize_feature_rows(index_features).to(device, non_blocking=True)
        scores = torch.matmul(_normalize_feature_rows(features), gallery.T)
    print(f"[MultiText][Score] rank={rank} finished similarity{tag_msg}: shape={tuple(scores.shape)}", flush=True)
    return scores


def _row_minmax(x: torch.Tensor) -> torch.Tensor:
    xmin = x.min(dim=-1, keepdim=True).values
    xmax = x.max(dim=-1, keepdim=True).values
    return (x - xmin) / (xmax - xmin).clamp_min(1e-8)


def _row_zscore(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std


def _normalize_scores(x: torch.Tensor, method: str) -> torch.Tensor:
    method = str(method or "minmax").lower()
    if method in {"none", "raw"}:
        return x.float()
    if method == "zscore":
        return _row_zscore(x.float())
    if method == "minmax":
        return _row_minmax(x.float())
    raise ValueError(f"Unsupported multi_text_score_norm={method!r}")


def _avg_features(*features: torch.Tensor) -> torch.Tensor:
    stack = torch.stack([_normalize_feature_rows(f) for f in features], dim=0)
    return _normalize_feature_rows(stack.mean(dim=0))


def _rrf_scores_from_topk(
    rankings: Sequence[List[List[str]]],
    index_names: Sequence[Any],
    k: int,
    device: torch.device,
    tag: str = "",
) -> torch.Tensor:
    """Sparse RRF from top-k branch rankings only.

    Old code iterated over full-gallery rankings. With CIRCO that can be tens
    of millions of Python string operations per RRF mode. This sparse variant
    only uses the saved topK lists, which is what the diagnostic comparison
    needs and what avoids the apparent hang after text encoding.
    """
    rank = int(os.environ.get("RANK", "0"))
    tag_msg = f" for {tag}" if tag else ""
    print(f"[MultiText][RRF] rank={rank} start sparse RRF{tag_msg}", flush=True)
    index_names_list = [_to_str(x) for x in index_names]
    name_to_idx = {name: i for i, name in enumerate(index_names_list)}
    num_queries = len(rankings[0])
    scores = torch.zeros((num_queries, len(index_names_list)), dtype=torch.float32, device=device)
    for branch_rankings in rankings:
        if len(branch_rankings) != num_queries:
            raise ValueError("RRF ranking branch length mismatch")
        for qi, names in enumerate(branch_rankings):
            for rank_pos, name in enumerate(names):
                idx = name_to_idx.get(_to_str(name))
                if idx is not None:
                    scores[qi, idx] += 1.0 / (float(k) + float(rank_pos) + 1.0)
    out = _row_minmax(scores)
    print(f"[MultiText][RRF] rank={rank} finished sparse RRF{tag_msg}", flush=True)
    return out


def _build_multi_text_features_and_scores(
    kwargs: Dict[str, Any],
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build only embedding-level multi-text features and top-k rankings.

    Embedding-only policy:
      - keep primitive text branches: cap, img_direct, img_prompt;
      - keep averaged embedding branches: avg_cap_direct, avg_cap_prompt,
        avg_direct_prompt, avg_all3;
      - do NOT compute similarity-fusion rankings;
      - do NOT compute RRF rankings;
      - do NOT expose S_t override matrices.

    This keeps the ablation focused on the representation fed into IP-CIR:
        f_t = selected CLIP text feature / averaged CLIP text feature.
    """
    args = kwargs["args"]
    index_features = kwargs["index_features"]
    index_names = kwargs["index_names"]
    save_topk = int(getattr(args, "stage1_save_topk", 100))
    save_topk = max(save_topk, int(getattr(args, "rank_topk", 100)))
    save_topk = min(save_topk, len(index_names))

    queries = {
        "cap": [str(x.get("modified_caption", "")) for x in items],
        "img_direct": [str(x.get("modified_caption_img_direct", "")) for x in items],
        "img_prompt": [str(x.get("modified_caption_img_prompt", "")) for x in items],
    }

    print("[MultiText][EmbeddingOnly] start CLIP text encoding for 3 query branches", flush=True)
    features: Dict[str, torch.Tensor] = {
        "cap": _encode_texts(kwargs, queries["cap"], tag="cap"),
        "img_direct": _encode_texts(kwargs, queries["img_direct"], tag="img_direct"),
        "img_prompt": _encode_texts(kwargs, queries["img_prompt"], tag="img_prompt"),
    }

    print("[MultiText][EmbeddingOnly] build averaged text embeddings", flush=True)
    features["avg_cap_direct"] = _avg_features(features["cap"], features["img_direct"])
    features["avg_cap_prompt"] = _avg_features(features["cap"], features["img_prompt"])
    features["avg_direct_prompt"] = _avg_features(features["img_direct"], features["img_prompt"])
    features["avg_all3"] = _avg_features(features["cap"], features["img_direct"], features["img_prompt"])

    rankings: Dict[str, List[List[str]]] = {}
    similarities: Dict[str, torch.Tensor] = {}
    feature_mode_to_tag = {
        "cap": "cap_only_t2i",
        "img_direct": "img_direct_t2i",
        "img_prompt": "img_prompt_t2i",
        "avg_cap_direct": "avg_emb_cap_direct",
        "avg_cap_prompt": "avg_emb_cap_prompt",
        "avg_direct_prompt": "avg_emb_direct_prompt",
        "avg_all3": "avg_emb_all3",
    }

    for feature_key, tag in feature_mode_to_tag.items():
        scores = _scores_from_features(features[feature_key], index_features, tag=tag)
        similarities[tag] = scores
        # aliases for IP-merge S_t replacement
        if feature_key == "cap":
            similarities["cap_sim"] = scores
        elif feature_key == "img_direct":
            similarities["img_direct_sim"] = scores
        elif feature_key == "img_prompt":
            similarities["img_prompt_sim"] = scores
        elif feature_key.startswith("avg_"):
            similarities[f"sim_{feature_key}"] = scores
        rankings[tag] = _topk_rank_from_scores(scores, index_names, save_topk, tag=tag)

    # Lightweight score-level alternatives. These are just matrix additions and
    # are still fast. They are not RRF and do not involve Python rank loops.
    norm = str(getattr(args, "multi_text_score_norm", "minmax"))
    s_cap = _normalize_scores(similarities["cap_sim"], norm)
    s_direct = _normalize_scores(similarities["img_direct_sim"], norm)
    s_prompt = _normalize_scores(similarities["img_prompt_sim"], norm)
    similarities["sim_avg_cap_direct"] = 0.5 * s_cap + 0.5 * s_direct
    similarities["sim_avg_cap_prompt"] = 0.5 * s_cap + 0.5 * s_prompt
    similarities["sim_avg_all3"] = (s_cap + s_direct + s_prompt) / 3.0

    for tag in ("sim_avg_cap_direct", "sim_avg_cap_prompt", "sim_avg_all3"):
        if tag in set(str(x) for x in getattr(args, "stage1_eval_modes", [])):
            rankings[tag] = _topk_rank_from_scores(similarities[tag], index_names, save_topk, tag=tag)

    print("[MultiText][Simple] finished embedding/similarity features and top-k rankings", flush=True)
    return {
        "queries": queries,
        "features": features,
        "similarities": similarities,
        "rankings": rankings,
    }

# -----------------------------------------------------------------------------
# Artifact saving
# -----------------------------------------------------------------------------


def _save_record_artifact(
    dataset_path: str,
    task: str,
    clip: str,
    dataset: str,
    tag: str,
    records: Sequence[Dict[str, Any]],
) -> str:
    save_path = os.path.join(_task_dir(dataset_path, task), f"top_rank_{clip}_{dataset}_{tag}_{_get_time()}.json")
    _write_json(save_path, list(records))
    print(f"[Artifact] saved top-rank records to: {save_path}")
    return save_path


def _save_metric_artifact(dataset_path: str, task: str, clip: str, dataset: str, tag: str, metrics: Dict[str, float]) -> str:
    save_path = os.path.join(_task_dir(dataset_path, task), f"result_{clip}_{dataset}_{tag}_{_get_time()}.json")
    _write_json(save_path, metrics)
    print(f"[Artifact] saved metrics to: {save_path}")
    return save_path


def _build_records(
    stage1_out: Dict[str, Any],
    rankings: Sequence[Sequence[str]],
    topk: int = 50,
    extra_per_query: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    records = compute_results_ipcir_qwen.build_top_rank_records(
        rankings=rankings,
        query_ids=stage1_out.get("query_ids"),
        reference_names=stage1_out.get("reference_names"),
        target_names=stage1_out.get("target_names"),
        topk=topk,
    )
    if extra_per_query:
        for rec, extra in zip(records, extra_per_query):
            rec.update(extra)
    return records


def _branch_query_for_mode(mode: str, item: Dict[str, Any]) -> str:
    if mode == "cap_only_t2i":
        return item.get("modified_caption", "")
    if mode == "img_direct_t2i":
        return item.get("modified_caption_img_direct", "")
    if mode == "img_prompt_t2i":
        return item.get("modified_caption_img_prompt", "")
    if mode == "avg_emb_cap_direct":
        return "[avg embedding] modified_caption + modified_caption_img_direct"
    if mode == "avg_emb_cap_prompt":
        return "[avg embedding] modified_caption + modified_caption_img_prompt"
    if mode == "avg_emb_direct_prompt":
        return "[avg embedding] modified_caption_img_direct + modified_caption_img_prompt"
    if mode == "avg_emb_all3":
        return "[avg embedding] modified_caption + modified_caption_img_direct + modified_caption_img_prompt"
    if mode == "sim_avg_cap_direct":
        return "[similarity avg] modified_caption + modified_caption_img_direct"
    if mode == "sim_avg_cap_prompt":
        return "[similarity avg] modified_caption + modified_caption_img_prompt"
    if mode == "sim_avg_all3":
        return "[similarity avg] all three text branches"
    return ""


def _build_simple_t2i_records(
    stage1_out: Dict[str, Any],
    rankings: Sequence[Sequence[str]],
    items: Sequence[Dict[str, Any]],
    mode: str,
    topk: int,
) -> List[Dict[str, Any]]:
    query_ids = list(stage1_out.get("query_ids", range(len(rankings))))
    reference_names = list(stage1_out.get("reference_names", [None] * len(rankings)))
    target_names = list(stage1_out.get("target_names", [None] * len(rankings)))
    out: List[Dict[str, Any]] = []
    for i, names in enumerate(rankings):
        item = items[i] if i < len(items) else {}
        ref_name = reference_names[i] if i < len(reference_names) else item.get("image_index", i)
        rec = {
            "image_index": _to_str(item.get("image_index", ref_name)),
            "query_id": _to_str(query_ids[i]) if i < len(query_ids) else _to_str(i),
            "reference_name": _to_str(ref_name),
            "target_name": None if i >= len(target_names) or target_names[i] is None else _to_str(target_names[i]),
            "branch": mode,
            "query_text": _branch_query_for_mode(mode, item),
            "modified_caption": item.get("modified_caption", ""),
            "modified_caption_img_direct": item.get("modified_caption_img_direct", ""),
            "modified_caption_img_prompt": item.get("modified_caption_img_prompt", ""),
            "top_names": [_to_str(x) for x in list(names)[:topk]],
        }
        out.append(rec)
    return out


def _save_multitext_t2i_file(
    kwargs: Dict[str, Any],
    mode: str,
    records: Sequence[Dict[str, Any]],
    topk: int,
) -> str:
    save_dir = _multitext_t2i_dir(kwargs["dataset_path"], kwargs["task"])
    save_path = os.path.join(save_dir, f"{mode}_top{topk}.json")
    _write_json_atomic_no_done(save_path, list(records))
    print(f"[MultiText][T2I] saved {mode}: {save_path}", flush=True)
    return save_path


def _save_multi_text_rank_artifacts(
    kwargs: Dict[str, Any],
    stage1_out: Dict[str, Any],
    multi: Dict[str, Any],
    items: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Always save every single/avg text branch to task/<task>/multitext_t2i/.

    This is intentionally independent of ip_merge_text_feature_source.  That
    config only decides which text embedding goes into IP-merge; it must not
    suppress branch-level t2i dumps.
    """
    if not _is_main_process():
        return {}
    args = kwargs["args"]
    topk = int(getattr(args, "stage1_save_topk", 100))

    core_modes = [
        "cap_only_t2i",
        "img_direct_t2i",
        "img_prompt_t2i",
        "avg_emb_cap_direct",
        "avg_emb_cap_prompt",
        "avg_emb_direct_prompt",
        "avg_emb_all3",
    ]
    similarity_modes = ["sim_avg_cap_direct", "sim_avg_cap_prompt", "sim_avg_all3"]
    save_similarity = bool(getattr(args, "multi_text_save_similarity_t2i", False)) or bool(
        getattr(args, "ip_merge_use_similarity_override", False)
    )
    modes = core_modes + (similarity_modes if save_similarity else [])

    artifact_paths: Dict[str, str] = {}
    print(f"[MultiText][T2I] saving branch t2i files: {modes}", flush=True)
    for mode in modes:
        if mode not in multi["rankings"]:
            print(f"[MultiText][T2I] skip missing ranking mode={mode}", flush=True)
            continue
        records = _build_simple_t2i_records(stage1_out, multi["rankings"][mode], items, mode=mode, topk=topk)
        artifact_paths[mode] = _save_multitext_t2i_file(kwargs, mode, records, topk=topk)
    return artifact_paths


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


def _save_raw_branch_top_rank_artifacts(kwargs: Dict[str, Any], stage1_out: Dict[str, Any]) -> Dict[str, Optional[str]]:
    if not _is_main_process():
        return {}
    saved: Dict[str, Optional[str]] = {"t2i_rank_artifact_path": None, "i2i_rank_artifact_path": None}
    for branch, key in (("t2i", "txt_sorted_index_names"), ("i2i", "img_sorted_index_names")):
        rankings = _safe_get_stage_sequence(stage1_out, key)
        if not rankings:
            continue
        records = _build_records(stage1_out, rankings, topk=50)
        saved[f"{branch}_rank_artifact_path"] = _save_record_artifact(
            dataset_path=kwargs["dataset_path"],
            task=kwargs["task"],
            clip=kwargs["clip"],
            dataset=kwargs["dataset_name"],
            tag=f"{branch}_loop_0",
            records=records,
        )
    return saved


# -----------------------------------------------------------------------------
# Stage-1 metrics and VQA support
# -----------------------------------------------------------------------------


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


def _build_candidate_side_info(
    candidate_names: Sequence[Sequence[str]],
    caption_lookup: Dict[str, str],
    path_lookup: Dict[str, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    all_caps: List[List[str]] = []
    all_paths: List[List[str]] = []
    for names in candidate_names:
        all_caps.append([_get_value_by_name(caption_lookup, n) for n in names])
        all_paths.append([_get_value_by_name(path_lookup, n) for n in names])
    return all_caps, all_paths


def _select_text_feature_source(multi: Dict[str, Any], source: str) -> torch.Tensor:
    source = str(source or "cap")
    features = multi["features"]
    if source not in features:
        raise KeyError(f"Unknown ip_merge_text_feature_source={source!r}; available={sorted(features.keys())}")
    return features[source]


def _select_similarity_source(multi: Dict[str, Any], source: str) -> torch.Tensor:
    source = str(source or "sim_avg_cap_prompt")
    sims = multi["similarities"]
    if source not in sims:
        raise KeyError(f"Unknown ip_merge_st_source={source!r}; available={sorted(sims.keys())}")
    return sims[source]



# -----------------------------------------------------------------------------
# Distributed Stage-1 shared cache
# -----------------------------------------------------------------------------


def _serializable(obj: Any) -> Any:
    """Convert tensors and nested containers to JSON-serializable objects."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return float(obj)
    except Exception:
        return _to_str(obj)


def _multi_text_stage1_cache_path(kwargs: Dict[str, Any], selected_text_source: str, st_source: Optional[str]) -> str:
    args = kwargs["args"]
    source = str(selected_text_source or "cap")
    filename = f"{kwargs['clip']}_{kwargs['dataset_name']}_{kwargs.get('split', getattr(args, 'split', 'val'))}_multitext_stage1_{source}.json"
    return os.path.join(_task_dir(kwargs["dataset_path"], kwargs["task"]), "stage1_cache", filename)


def _save_multi_text_stage1_cache(path: str, stage1_out: Dict[str, Any]) -> None:
    keys = [
        "stage1_pool_names",
        "stage1_pool_score_maps",
        "stage1_txt_score_maps",
        "stage1_img_score_maps",
        "stage1_lambda",
        "stage1_output_metrics",
        "stage1_output_labels",
        "stage1_metric_artifact_path",
        "stage1_rank_artifact_path",
        "multi_text_query_path",
        "multi_text_rank_artifact_paths",
        "ip_merge_text_feature_source_used",
        "ip_merge_st_source_used",
        "txt_sorted_index_names",
    ]
    payload = {k: _serializable(stage1_out.get(k)) for k in keys if k in stage1_out}
    payload["created_at"] = _get_time()
    _write_json(path, payload)
    print(f"[MultiText][Distributed] saved rank0 stage1 cache: {path}", flush=True)


def _load_multi_text_stage1_cache(path: str) -> Dict[str, Any]:
    payload = _read_json(path)
    print(f"[MultiText][Distributed] loaded rank0 stage1 cache: {path}", flush=True)
    return payload


def _distributed_nonmain_load_multitext_stage1_cache(
    kwargs: Dict[str, Any],
    stage1_out: Dict[str, Any],
    selected_text_source: str,
    st_source: Optional[str],
) -> Dict[str, Any]:
    path = _multi_text_stage1_cache_path(kwargs, selected_text_source, st_source)
    print(f"[MultiText][Distributed] rank={os.environ.get('RANK','?')} waits for stage1 cache: {path}", flush=True)
    _wait_for_file(path, timeout_s=int(getattr(kwargs["args"], "multi_text_stage1_cache_timeout_s", 7200)), poll_s=5)
    cached = _load_multi_text_stage1_cache(path)
    stage1_out.update(cached)
    return stage1_out

# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------


def generate_editimg_caption_iteration(**kwargs):
    """IP-CIR wrapper with explicit multi-text Stage-1 support.

    New behavior when args.enable_multi_text_queries=True:
      1. Run the original Stage-1 once to obtain caption_query and proxy image feature.
      2. Generate/load two additional target texts:
            ref image + mod text -> vision_direct_query
            ref image + mod text + retrieval prompt -> vision_prompted_query
      3. Save the three-query cache as *_multi_text_queries.json.
      4. Encode all text branches, save single/fused t2i rank artifacts.
      5. Feed IP-CIR with the selected text embedding only. No RRF / S_t override.
    """
    args = kwargs["args"]
    requested_stage_mode = getattr(args, "stage_mode", "initial_only")
    setattr(args, "stage_mode", "initial_only")
    stage1_out = _run_base_stage1(kwargs)
    setattr(args, "stage_mode", requested_stage_mode)

    raw_rank_paths = _save_raw_branch_top_rank_artifacts(kwargs, stage1_out)
    stage1_out.update(raw_rank_paths)

    multi_enabled_raw = getattr(args, "enable_multi_text_queries", False)
    if isinstance(multi_enabled_raw, str):
        multi_enabled = multi_enabled_raw.strip().lower() == "true"
    else:
        multi_enabled = bool(multi_enabled_raw)
    print(f"[MultiText][DEBUG] raw enable_multi_text_queries = {multi_enabled_raw!r}")
    print(f"[MultiText][DEBUG] bool enable_multi_text_queries = {multi_enabled!r}")
    print(f"[MultiText][DEBUG] task = {kwargs.get('task')!r}; dataset = {kwargs.get('dataset_name')!r}; split = {kwargs.get('split')!r}")
    text_similarity_override = None
    selected_text_source = str(getattr(args, "ip_merge_text_feature_source", "avg_cap_prompt") or "avg_cap_prompt")
    st_source = None
    pool_already_loaded_from_rank0 = False

    if multi_enabled:
        print("[MultiText] explicit multi-text Stage-1 enabled.", flush=True)
        query_path, items = _ensure_multi_text_queries(kwargs, stage1_out)
        _release_bagel_editor_from_kwargs(kwargs, reason="multi-text query cache is ready")

        # Simple distributed policy:
        #   rank0 generates the compact query JSON if missing;
        #   every rank loads the same compact JSON and computes its local CLIP/IP-CIR tensors;
        #   only rank0 writes artifacts.  No huge stage1 JSON cache is written/read.
        multi = _build_multi_text_features_and_scores(kwargs, items)
        rank_paths = _save_multi_text_rank_artifacts(kwargs, stage1_out, multi, items)

        selected_features = _select_text_feature_source(multi, selected_text_source)
        stage1_out["predicted_txt_features_single_branch"] = stage1_out.get("predicted_txt_features")
        stage1_out["predicted_txt_features"] = selected_features

        selected_rank_key = {
            "cap": "cap_only_t2i",
            "img_direct": "img_direct_t2i",
            "img_prompt": "img_prompt_t2i",
            "avg_cap_direct": "avg_emb_cap_direct",
            "avg_cap_prompt": "avg_emb_cap_prompt",
            "avg_direct_prompt": "avg_emb_direct_prompt",
            "avg_all3": "avg_emb_all3",
        }.get(selected_text_source, "avg_emb_cap_prompt")
        if selected_rank_key in multi["rankings"]:
            stage1_out["txt_sorted_index_names_single_branch"] = stage1_out.get("txt_sorted_index_names")
            stage1_out["txt_sorted_index_names"] = multi["rankings"][selected_rank_key]

        if bool(getattr(args, "ip_merge_use_similarity_override", False)):
            st_source = str(getattr(args, "ip_merge_st_source", "sim_avg_cap_prompt") or "sim_avg_cap_prompt")
            text_similarity_override = _select_similarity_source(multi, st_source)
            print(f"[MultiText][IP-CIR] using text embedding source={selected_text_source}; replacing S_t with {st_source}", flush=True)
        else:
            print(f"[MultiText][IP-CIR] using text embedding source: {selected_text_source}", flush=True)

        stage1_out.update(
            {
                "multi_text_query_path": query_path,
                "multi_text_items": items,
                "text_queries_by_source": multi["queries"],
                "text_features_by_source": multi["features"],
                "text_similarities_by_source": multi["similarities"],
                "text_rankings_by_mode": multi["rankings"],
                "multi_text_rank_artifact_paths": rank_paths,
                "ip_merge_text_feature_source_used": selected_text_source,
                "ip_merge_st_source_used": st_source,
            }
        )

    if not pool_already_loaded_from_rank0:
        print(f"[IP-CIR] rank={os.environ.get('RANK','0')} building merged pool", flush=True)
        with torch.inference_mode():
            query_caption_features = base_utils.text_encoding(
                device=kwargs["device"],
                clip_model=kwargs["clip_model"],
                input_captions=stage1_out["start_captions"],
                batch_size=32,
                mode=kwargs["retrieval"],
            )
        query_caption_features = _normalize_feature_rows(query_caption_features)

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
            text_similarity_override=text_similarity_override,
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
            records = _build_records(stage1_out, stage1_labels, topk=int(getattr(args, "stage1_save_topk", 100)))
            stage1_out["stage1_rank_artifact_path"] = _save_record_artifact(
                dataset_path=kwargs["dataset_path"],
                task=kwargs["task"],
                clip=kwargs["clip"],
                dataset=kwargs["dataset_name"],
                tag="merged_loop_0",
                records=records,
            )
            ipmerge_dir = _ipmerge_dir(kwargs["dataset_path"], kwargs["task"])
            merged_top_path = os.path.join(ipmerge_dir, f"merged_top{int(getattr(args, 'stage1_save_topk', 100))}.json")
            _write_json_atomic_no_done(merged_top_path, records)
            stage1_out["stage1_ipmerge_top_path"] = merged_top_path
            print(f"[IP-CIR] saved stable merged topK: {merged_top_path}", flush=True)
            if stage1_metrics is not None:
                merged_metric_path = os.path.join(ipmerge_dir, "merged_metrics.json")
                _write_json_atomic_no_done(merged_metric_path, stage1_metrics)
                stage1_out["stage1_ipmerge_metric_path"] = merged_metric_path
                print(f"[IP-CIR] saved stable merged metrics: {merged_metric_path}", flush=True)
    else:
        print(f"[IP-CIR] rank={os.environ.get('RANK','?')} uses merged pool loaded from rank0 cache", flush=True)

    if requested_stage_mode == "initial_only":
        return stage1_out
    if requested_stage_mode != "qwen_fusion":
        raise ValueError(f"Unsupported stage_mode in IP-CIR wrapper: {requested_stage_mode}")

    distributed_vqa = bool(getattr(args, "distributed_vqa", False))
    if _dist_is_ready() and (not distributed_vqa) and (not _is_main_process()):
        print("[VQA] Distributed run with distributed_vqa=False; non-main rank skips verifier work.")
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
