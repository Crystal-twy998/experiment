from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def get_time() -> str:
    return datetime.datetime.now().strftime("%Y.%m.%d-%H_%M_%S")


def _to_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _maybe_get(seq: Optional[Sequence[Any]], idx: int, default: Any = None) -> Any:
    if seq is None:
        return default
    try:
        if idx < len(seq):
            return seq[idx]
    except Exception:
        return default
    return default


def _listify(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [_to_str(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_str(v) for v in x]
    return [_to_str(x)]


def _unique_keep_order(names: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for n in names:
        s = _to_str(n)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = np.asarray(list(score_map.values()), dtype=np.float32)
    vmin = float(vals.min())
    vmax = float(vals.max())
    denom = max(vmax - vmin, 1e-8)
    return {k: float((float(v) - vmin) / denom) for k, v in score_map.items()}


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_top_rank_records(
    rankings: Sequence[Sequence[Any]],
    query_ids: Optional[Sequence[Any]] = None,
    reference_names: Optional[Sequence[Any]] = None,
    target_names: Optional[Sequence[Any]] = None,
    topk: int = 50,
) -> List[Dict[str, Any]]:
    """Build the human-checkable top-rank json format used by all datasets.

    Output example:
    [
      {
        "query_id": "...",
        "image_index": "dev-244-0-img0",
        "target_name": "dev-1028-1-img1",
        "top_names": ["...", ...]   # always at most topk, default 50
      }
    ]

    target_name can be None for test splits without labels.
    """
    records: List[Dict[str, Any]] = []
    for i, rank in enumerate(rankings):
        qid = _maybe_get(query_ids, i, i)
        ref = _maybe_get(reference_names, i, qid)
        tgt = _maybe_get(target_names, i, None)
        top_names = _unique_keep_order(_listify(rank))[: int(topk)]
        records.append(
            {
                "query_id": _to_str(qid),
                "image_index": None if ref is None else _to_str(ref),
                "target_name": None if tgt is None else _to_str(tgt),
                "top_names": top_names,
            }
        )
    return records


def save_top_rank_artifact(
    dataset_path: str,
    task: str,
    clip: str,
    dataset: str,
    tag: str,
    rankings: Sequence[Sequence[Any]],
    query_ids: Optional[Sequence[Any]] = None,
    reference_names: Optional[Sequence[Any]] = None,
    target_names: Optional[Sequence[Any]] = None,
    topk: int = 50,
) -> str:
    out_dir = os.path.join(dataset_path, "task", task)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"top_rank_{clip}_{dataset}_{tag}_{get_time()}.json")
    records = build_top_rank_records(
        rankings=rankings,
        query_ids=query_ids,
        reference_names=reference_names,
        target_names=target_names,
        topk=topk,
    )
    _save_json(save_path, records)
    print(f"[Artifact] saved top-rank records to: {save_path}")
    return save_path


def _build_submission_dict(rankings: Sequence[Sequence[Any]], query_ids: Sequence[Any], topk: int = 50) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for qid, ranking in zip(query_ids, rankings):
        out[_submission_query_key(qid)] = _unique_keep_order(_listify(ranking))[: int(topk)]
    return out


def _clip_prefix_from_kwargs(kwargs: Dict[str, Any]) -> str:
    clip_name = kwargs.get("clip", "")
    if clip_name is None:
        clip_name = ""
    clip_name = str(clip_name).strip()
    return f"{clip_name}_" if clip_name else ""


def _submission_query_key(qid: Any) -> str:
    try:
        if hasattr(qid, "detach"):
            qid = qid.detach().cpu().item()
        return str(int(qid))
    except Exception:
        return _to_str(qid)


def _task_output_dir(kwargs: Dict[str, Any]) -> str:
    # Same location as the original raw t2i/i2i submission files.
    out_dir = os.path.join(kwargs["dataset_path"], "task", kwargs["task"])
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _latest_json_matches(out_dir: str, prefixes: Sequence[str]) -> Optional[Dict[str, Any]]:
    try:
        candidates: List[str] = []
        for name in os.listdir(out_dir):
            if name.endswith(".json") and any(name.startswith(prefix) for prefix in prefixes):
                candidates.append(os.path.join(out_dir, name))
        if not candidates:
            return None
        latest = max(candidates, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _raw_cirr_uses_metadata(kwargs: Dict[str, Any], subset: bool) -> bool:
    """Match the t2i/i2i CIRR submission style already produced by base_utils."""
    out_dir = _task_output_dir(kwargs)
    clip_prefix = _clip_prefix_from_kwargs(kwargs)
    dataset = str(kwargs.get("dataset_name", "cirr")).lower()
    if subset:
        prefixes = [
            f"subset_test_submissions_{clip_prefix}{dataset}_t2i_cirr_loop_",
            f"subset_test_submissions_{clip_prefix}{dataset}_i2i_cirr_loop_",
            f"subset_test_submissions_{clip_prefix}{dataset}_final_rerank_cirr_loop_",
        ]
    else:
        prefixes = [
            f"test_submissions_{clip_prefix}{dataset}_t2i_cirr_loop_",
            f"test_submissions_{clip_prefix}{dataset}_i2i_cirr_loop_",
            f"test_submissions_{clip_prefix}{dataset}_final_rerank_cirr_loop_",
        ]
    payload = _latest_json_matches(out_dir, prefixes)
    if payload is None:
        return False
    return "version" in payload or "metric" in payload


def _wrap_cirr_submission(kwargs: Dict[str, Any], body: Dict[str, List[str]], subset: bool) -> Dict[str, Any]:
    if not _raw_cirr_uses_metadata(kwargs, subset=subset):
        return dict(body)
    meta = {"version": "rc2", "metric": "recall_subset" if subset else "recall"}
    meta.update(body)
    return meta


def _get_rank_at(seq: Any, idx: int) -> List[str]:
    if seq is None:
        return []
    try:
        return _listify(seq[idx])
    except Exception:
        return []


def _complete_cirr_subset(
    primary_rank: Sequence[Any],
    group: Sequence[Any],
    fallback_ranks: Sequence[Sequence[Any]] = (),
    ref_name: Any = None,
    topk: int = 3,
) -> List[str]:
    """Return a valid CIRR subset list.

    Merged/final rankings are usually truncated, so they may contain fewer than
    three images from the CIRR group. We first respect the merged/final order,
    then fill missing group images from full raw-branch rankings, then from the
    provided group order. This keeps the file valid without changing top_names.
    """
    ref = None if ref_name is None else _to_str(ref_name)
    group_order = [g for g in _unique_keep_order(_listify(group)) if ref is None or g != ref]
    group_set = set(group_order)
    out: List[str] = []

    def add_from(rank: Sequence[Any]) -> None:
        for name in _unique_keep_order(_listify(rank)):
            if name in group_set and name not in out:
                out.append(name)
                if len(out) >= topk:
                    return

    add_from(primary_rank)
    for fallback in fallback_ranks:
        if len(out) >= topk:
            break
        add_from(fallback)
    if len(out) < topk:
        add_from(group_order)
    return out[:topk]


def _save_cirr_test_submissions(
    kwargs: Dict[str, Any],
    rankings: Sequence[Sequence[str]],
    ways: str,
    fallback_rank_keys: Sequence[str] = (),
) -> Tuple[str, str]:
    """Save CIRR test/subset submissions using the same convention as raw t2i/i2i.

    This function only writes the requested branch (`merged` or `final_rerank`).
    It does not re-save raw t2i/i2i submissions. The subset ranking is derived
    strictly from this branch ranking: remove the reference image, then keep the
    first images that belong to the CIRR group, matching the original submission
    semantics.
    """
    out_dir = _task_output_dir(kwargs)
    ts = get_time()
    clip_prefix = _clip_prefix_from_kwargs(kwargs)
    dataset = str(kwargs.get("dataset_name", "cirr")).lower()
    loop = kwargs.get("loop", 0)
    query_ids = _listify(kwargs.get("query_ids", list(range(len(rankings)))))
    targets = kwargs.get("targets", [[] for _ in rankings])
    reference_names = _listify(kwargs.get("reference_names", [None for _ in rankings]))

    full_body: Dict[str, List[str]] = {}
    subset_body: Dict[str, List[str]] = {}
    for idx, (qid, rank, group) in enumerate(zip(query_ids, rankings, targets)):
        key = _submission_query_key(qid)
        ref = reference_names[idx] if idx < len(reference_names) else None
        filtered_rank = [n for n in _unique_keep_order(_listify(rank)) if ref is None or n != _to_str(ref)]
        full_body[key] = filtered_rank[:50]

        group_set = {_to_str(x) for x in _listify(group)}
        subset_body[key] = [name for name in filtered_rank if name in group_set][:3]

    submission = _wrap_cirr_submission(kwargs, full_body, subset=False)
    group_submission = _wrap_cirr_submission(kwargs, subset_body, subset=True)

    save_path = os.path.join(out_dir, f"test_submissions_{clip_prefix}{dataset}_{ways}_cirr_loop_{loop}_{ts}.json")
    subset_save_path = os.path.join(out_dir, f"subset_test_submissions_{clip_prefix}{dataset}_{ways}_cirr_loop_{loop}_{ts}.json")
    _save_json(save_path, submission)
    _save_json(subset_save_path, group_submission)
    print(f"Saved CIRR {ways} test submission to {save_path}")
    print(f"Saved CIRR {ways} subset test submission to {subset_save_path}")
    return save_path, subset_save_path

def _save_circo_test_submission(kwargs: Dict[str, Any], rankings: Sequence[Sequence[str]], ways: str) -> str:
    out_dir = _task_output_dir(kwargs)
    ts = get_time()
    clip_prefix = _clip_prefix_from_kwargs(kwargs)
    dataset = str(kwargs.get("dataset_name", "circo")).lower()
    loop = kwargs.get("loop", 0)
    query_ids = _listify(kwargs.get("query_ids", list(range(len(rankings)))))
    save_path = os.path.join(out_dir, f"test_submissions_{clip_prefix}{dataset}_{ways}_loop_{loop}_{ts}.json")
    _save_json(save_path, _build_submission_dict(rankings, query_ids, topk=50))
    print(f"Saved CIRCO {ways} test submission to {save_path}")
    return save_path


def _stage1_test_save_cirr(kwargs: Dict[str, Any], filtered_rankings: Sequence[Sequence[str]]) -> None:
    _save_cirr_test_submissions(kwargs, filtered_rankings, ways="merged")


def _stage1_test_save_circo(kwargs: Dict[str, Any], rankings: Sequence[Sequence[str]]) -> None:
    _save_circo_test_submission(kwargs, rankings, ways="merged")


def _is_scalar_conf(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def _extract_numeric_score(x: Any) -> float:
    if _is_scalar_conf(x):
        return float(x)
    if isinstance(x, str):
        return float(x)
    if isinstance(x, Mapping):
        for key in ("score", "confidence", "prob", "value"):
            if key in x:
                try:
                    return _extract_numeric_score(x[key])
                except Exception:
                    pass
        for v in x.values():
            try:
                return _extract_numeric_score(v)
            except Exception:
                continue
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        if len(x) >= 3:
            try:
                return _extract_numeric_score(x[-1])
            except Exception:
                pass
        if len(x) == 2:
            try:
                return _extract_numeric_score(x[1])
            except Exception:
                pass
        if len(x) > 0:
            try:
                return _extract_numeric_score(x[0])
            except Exception:
                pass
        for item in reversed(list(x)):
            try:
                return _extract_numeric_score(item)
            except Exception:
                continue
    raise TypeError(f"Cannot extract numeric score from object of type {type(x).__name__}: {x!r}")


def _build_conf_map(names_entry: Any, conf_entry: Any) -> Dict[str, float]:
    names = _listify(names_entry)
    if not names:
        return {}

    if isinstance(conf_entry, Mapping):
        out: Dict[str, float] = {}
        for n in names:
            if n in conf_entry:
                try:
                    out[n] = _extract_numeric_score(conf_entry[n])
                except Exception:
                    continue
        if out:
            return out

    try:
        if _is_scalar_conf(conf_entry) or isinstance(conf_entry, str):
            score = _extract_numeric_score(conf_entry)
            return {n: score for n in names}
    except Exception:
        pass

    if isinstance(conf_entry, np.ndarray):
        conf_values = conf_entry.tolist()
    elif isinstance(conf_entry, (list, tuple)):
        conf_values = list(conf_entry)
    else:
        try:
            conf_values = list(conf_entry)
        except Exception:
            try:
                score = _extract_numeric_score(conf_entry)
                return {n: score for n in names}
            except Exception:
                return {}

    if len(conf_values) == 0:
        return {}
    if len(conf_values) == 1:
        try:
            score = _extract_numeric_score(conf_values[0])
            return {n: score for n in names}
        except Exception:
            return {}

    out: Dict[str, float] = {}
    for n, s in zip(names, conf_values):
        try:
            out[n] = _extract_numeric_score(s)
        except Exception:
            continue
    return out


def _build_rank_score_map(names_entry: Any, rank_entry: Any) -> Dict[str, float]:
    names = _listify(names_entry)
    if not names:
        return {}

    if isinstance(rank_entry, Mapping):
        out: Dict[str, float] = {}
        for n in names:
            if n in rank_entry:
                try:
                    out[n] = _extract_numeric_score(rank_entry[n])
                except Exception:
                    continue
        return out

    if isinstance(rank_entry, np.ndarray):
        values = rank_entry.tolist()
    elif isinstance(rank_entry, (list, tuple)):
        values = list(rank_entry)
    else:
        try:
            values = list(rank_entry)
        except Exception:
            return {}

    if len(values) == 0:
        return {}

    if len(values) == len(names):
        out: Dict[str, float] = {}
        for n, item in zip(names, values):
            try:
                out[n] = _extract_numeric_score(item)
            except Exception:
                continue
        if out:
            return out

    out: Dict[str, float] = {}
    for item in values:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            k = _to_str(item[0])
            try:
                out[k] = _extract_numeric_score(item[1])
            except Exception:
                continue
    if out:
        filtered = {n: out[n] for n in names if n in out}
        if filtered:
            return filtered
    return {}


def _build_verifier_map(names_entry: Any, rank_entry: Any, conf_entry: Any) -> Dict[str, float]:
    rank_map = _build_rank_score_map(names_entry, rank_entry)
    if rank_map:
        return _normalize_score_map(rank_map)
    conf_map = _build_conf_map(names_entry, conf_entry)
    if conf_map:
        return _normalize_score_map(conf_map)
    return {}


def _build_final_rankings(
    args: Any,
    stage1_pool_names: Sequence[Sequence[str]],
    stage1_pool_score_maps: Sequence[Dict[str, float]],
    stage1_txt_score_maps: Sequence[Dict[str, float]],
    stage1_img_score_maps: Sequence[Dict[str, float]],
    candidates1: Sequence[Any],
    candidates2: Sequence[Any],
    ranks1: Sequence[Any],
    ranks2: Sequence[Any],
    confidences1: Sequence[Any],
    confidences2: Sequence[Any],
) -> List[List[str]]:
    """Fuse stage-1 prior and verifier scores.

    Reranking is done over the merged pool candidates, by default top-100.
    The caller can still save only top-50 for debug/submission.
    """
    final_rankings: List[List[str]] = []

    score_mode = str(getattr(args, "rerank_score_mode", "prior_plus_verifier"))
    prior_w = float(getattr(args, "rerank_prior_weight", 0.50))
    ver1_w = float(getattr(args, "rerank_verifier1_weight", 0.50))
    ver2_w = float(getattr(args, "rerank_verifier2_weight", 0.00))
    dual_verifier_bonus = float(getattr(args, "rerank_dual_verifier_bonus", 0.00))
    dual_retrieval_bonus = float(getattr(args, "rerank_dual_retrieval_bonus", 0.00))

    rerank_pool_size = int(getattr(args, "rerank_pool_size", getattr(args, "topk_for_vqa", 100)))
    rerank_pool_size = max(rerank_pool_size, 50)
    output_topk = int(getattr(args, "rank_topk", 50))
    output_topk = max(output_topk, 50)

    total = len(stage1_pool_names)
    for i in range(total):
        prior_map = _normalize_score_map(dict(stage1_pool_score_maps[i])) if i < len(stage1_pool_score_maps) else {}
        txt_map = _normalize_score_map(dict(stage1_txt_score_maps[i])) if i < len(stage1_txt_score_maps) else {}
        img_map = _normalize_score_map(dict(stage1_img_score_maps[i])) if i < len(stage1_img_score_maps) else {}

        merged_top = _listify(stage1_pool_names[i])[:rerank_pool_size]
        cand1 = candidates1[i] if i < len(candidates1) else []
        cand2 = candidates2[i] if i < len(candidates2) else []
        rank1 = ranks1[i] if i < len(ranks1) else []
        rank2 = ranks2[i] if i < len(ranks2) else []
        conf1 = confidences1[i] if i < len(confidences1) else []
        conf2 = confidences2[i] if i < len(confidences2) else []

        ver1_map = _build_verifier_map(cand1, rank1, conf1)
        ver2_map = _build_verifier_map(cand2, rank2, conf2)

        union = _unique_keep_order(merged_top + _listify(cand1) + _listify(cand2))
        score_map: Dict[str, float] = {}
        for name in union:
            prior = prior_map.get(name, 0.0)
            v1 = ver1_map.get(name, 0.0)
            v2 = ver2_map.get(name, 0.0)
            if score_mode == "verifier_only":
                score = ver1_w * v1 + ver2_w * v2
            elif score_mode == "sum":
                score = prior + v1 + v2
            else:
                score = prior_w * prior + ver1_w * v1 + ver2_w * v2
            if name in ver1_map and name in ver2_map:
                score += dual_verifier_bonus
            if name in txt_map and name in img_map:
                score += dual_retrieval_bonus
            score_map[name] = float(score)

        ordered = [
            n
            for n, _ in sorted(
                score_map.items(),
                key=lambda kv: (-kv[1], -prior_map.get(kv[0], 0.0), kv[0]),
            )
        ]
        final_rankings.append(ordered[:output_topk])
    return final_rankings


def _circo_metric_dict(rankings: Sequence[Sequence[str]], target_names: Sequence[Any], all_targets: Sequence[Sequence[Any]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    mAP_values: Dict[str, float] = {}
    recall_values: Dict[str, float] = {}
    target_names_list = _listify(target_names)
    for k in [5, 10, 25, 50]:
        recall_hits = []
        ap_scores = []
        for primary_tgt, pred, gt_list in zip(target_names_list, rankings, all_targets):
            pred_list = _listify(pred)
            recall_hits.append(1.0 if primary_tgt in pred_list[:k] else 0.0)
            gt_set = {_to_str(x) for x in _listify(gt_list)}
            pred_k = pred_list[:k]
            running_hits = 0
            precisions = []
            for rank_idx, name in enumerate(pred_k, start=1):
                if name in gt_set:
                    running_hits += 1
                    precisions.append(running_hits / rank_idx)
            denom = max(len(gt_set), 1)
            ap_scores.append(float(sum(precisions) / denom) if precisions else 0.0)
        mAP_values[f"mAP@{k}"] = float(np.mean(ap_scores) * 100.0) if ap_scores else 0.0
        recall_values[f"recall@{k}"] = float(np.mean(recall_hits) * 100.0) if recall_hits else 0.0
    for k in [5, 10, 25, 50]:
        metrics[f"mAP@{k}"] = mAP_values[f"mAP@{k}"]
    for k in [5, 10, 25, 50]:
        metrics[f"recall@{k}"] = recall_values[f"recall@{k}"]
    return metrics


def fiq_stage1_pool(**kwargs: Any) -> Tuple[Optional[Dict[str, float]], List[List[str]]]:
    split = kwargs["split"]
    rankings = [_listify(x) for x in kwargs["stage1_pool_names"]]
    if str(split).lower().startswith("test"):
        return None, rankings
    target_names = _listify(kwargs.get("target_names", []))
    recalls: Dict[str, float] = {}
    for k in [1, 5, 10, 50]:
        hits = [1.0 if tgt in pred[:k] else 0.0 for pred, tgt in zip(rankings, target_names)]
        recalls[f"recall@{k}"] = float(np.mean(hits) * 100.0) if hits else 0.0
    return recalls, rankings


def cirr_stage1_pool(**kwargs: Any) -> Tuple[Optional[Dict[str, float]], List[List[str]]]:
    split = kwargs["split"]
    rankings = [_listify(x) for x in kwargs["stage1_pool_names"]]
    reference_names = _listify(kwargs.get("reference_names", []))
    filtered_rankings = []
    for ref_name, pred in zip(reference_names, rankings):
        filtered_rankings.append([n for n in pred if n != ref_name])

    if str(split).lower().startswith("test"):
        if _is_main_process():
            _stage1_test_save_cirr(kwargs, filtered_rankings)
        return None, filtered_rankings

    target_names = _listify(kwargs.get("target_names", []))
    group_members = kwargs.get("targets", [[] for _ in filtered_rankings])
    metrics: Dict[str, float] = {}
    for k in [1, 5, 10, 50]:
        hits = [1.0 if tgt in rank[:k] else 0.0 for tgt, rank in zip(target_names, filtered_rankings)]
        metrics[f"recall@{k}"] = float(np.mean(hits) * 100.0) if hits else 0.0
    for k in [1, 2, 3]:
        group_hits = []
        for tgt, rank, group in zip(target_names, filtered_rankings, group_members):
            group_set = {_to_str(x) for x in _listify(group)}
            pred = [x for x in rank if x in group_set][:k]
            group_hits.append(1.0 if tgt in pred else 0.0)
        metrics[f"group_recall@{k}"] = float(np.mean(group_hits) * 100.0) if group_hits else 0.0
    return metrics, filtered_rankings


def circo_stage1_pool(**kwargs: Any) -> Tuple[Optional[Dict[str, float]], List[List[str]]]:
    split = kwargs["split"]
    rankings = [_listify(x) for x in kwargs["stage1_pool_names"]]
    if str(split).lower().startswith("test"):
        if _is_main_process():
            _stage1_test_save_circo(kwargs, rankings)
        return None, rankings
    target_names = _listify(kwargs.get("target_names", []))
    all_targets = kwargs.get("targets", [[] for _ in rankings])
    return _circo_metric_dict(rankings, target_names, all_targets), rankings


def fiq_fuse2paths(**kwargs: Any) -> Tuple[Optional[Dict[str, float]], List[List[str]]]:
    args = kwargs["args"]
    split = kwargs["split"]
    final_rankings = _build_final_rankings(
        args=args,
        stage1_pool_names=kwargs["stage1_pool_names"],
        stage1_pool_score_maps=kwargs["stage1_pool_score_maps"],
        stage1_txt_score_maps=kwargs.get("stage1_txt_score_maps", []),
        stage1_img_score_maps=kwargs.get("stage1_img_score_maps", []),
        candidates1=kwargs.get("candidates1", []),
        candidates2=kwargs.get("candidates2", []),
        ranks1=kwargs.get("ranks1", []),
        ranks2=kwargs.get("ranks2", []),
        confidences1=kwargs.get("confidences1", []),
        confidences2=kwargs.get("confidences2", []),
    )
    if str(split).lower().startswith("test"):
        return None, final_rankings
    target_names = _listify(kwargs.get("target_names", []))
    recalls: Dict[str, float] = {}
    for k in [1, 5, 10, 50]:
        hits = [1.0 if tgt in pred[:k] else 0.0 for pred, tgt in zip(final_rankings, target_names)]
        recalls[f"recall@{k}"] = float(np.mean(hits) * 100.0) if hits else 0.0
    return recalls, final_rankings


def cirr_fuse2paths(**kwargs: Any) -> Tuple[Optional[Dict[str, float]], List[List[str]]]:
    args = kwargs["args"]
    split = kwargs["split"]
    final_rankings = _build_final_rankings(
        args=args,
        stage1_pool_names=kwargs["stage1_pool_names"],
        stage1_pool_score_maps=kwargs["stage1_pool_score_maps"],
        stage1_txt_score_maps=kwargs.get("stage1_txt_score_maps", []),
        stage1_img_score_maps=kwargs.get("stage1_img_score_maps", []),
        candidates1=kwargs.get("candidates1", []),
        candidates2=kwargs.get("candidates2", []),
        ranks1=kwargs.get("ranks1", []),
        ranks2=kwargs.get("ranks2", []),
        confidences1=kwargs.get("confidences1", []),
        confidences2=kwargs.get("confidences2", []),
    )
    reference_names = _listify(kwargs.get("reference_names", []))
    filtered_rankings = []
    for ref_name, pred in zip(reference_names, final_rankings):
        filtered_rankings.append([n for n in _listify(pred) if n != ref_name])

    query_ids = _listify(kwargs.get("query_ids", list(range(len(filtered_rankings)))))
    group_members = kwargs.get("targets", [[] for _ in filtered_rankings])

    if str(split).lower().startswith("test"):
        if not _is_main_process():
            return None, filtered_rankings
        _save_cirr_test_submissions(
            kwargs,
            filtered_rankings,
            ways="final_rerank",
            fallback_rank_keys=("stage1_pool_names", "txt_sorted_index_names", "img_sorted_index_names"),
        )
        return None, filtered_rankings

    target_names = _listify(kwargs.get("target_names", []))
    metrics: Dict[str, float] = {}
    for k in [1, 5, 10, 50]:
        hits = [1.0 if tgt in rank[:k] else 0.0 for tgt, rank in zip(target_names, filtered_rankings)]
        metrics[f"recall@{k}"] = float(np.mean(hits) * 100.0) if hits else 0.0
    for k in [1, 2, 3]:
        group_hits = []
        for tgt, rank, group in zip(target_names, filtered_rankings, group_members):
            group_set = {_to_str(x) for x in _listify(group)}
            pred = [x for x in rank if x in group_set][:k]
            group_hits.append(1.0 if tgt in pred else 0.0)
        metrics[f"group_recall@{k}"] = float(np.mean(group_hits) * 100.0) if group_hits else 0.0
    return metrics, filtered_rankings


def circo_fuse2paths(**kwargs: Any) -> Tuple[Optional[Dict[str, float]], List[List[str]]]:
    args = kwargs["args"]
    split = kwargs["split"]
    final_rankings = _build_final_rankings(
        args=args,
        stage1_pool_names=kwargs["stage1_pool_names"],
        stage1_pool_score_maps=kwargs["stage1_pool_score_maps"],
        stage1_txt_score_maps=kwargs.get("stage1_txt_score_maps", []),
        stage1_img_score_maps=kwargs.get("stage1_img_score_maps", []),
        candidates1=kwargs.get("candidates1", []),
        candidates2=kwargs.get("candidates2", []),
        ranks1=kwargs.get("ranks1", []),
        ranks2=kwargs.get("ranks2", []),
        confidences1=kwargs.get("confidences1", []),
        confidences2=kwargs.get("confidences2", []),
    )
    query_ids = _listify(kwargs.get("query_ids", list(range(len(final_rankings)))))
    target_names = _listify(kwargs.get("target_names", []))
    all_targets = kwargs.get("targets", [[] for _ in final_rankings])

    if str(split).lower().startswith("test"):
        if not _is_main_process():
            return None, final_rankings
        _save_circo_test_submission(kwargs, final_rankings, ways="final_rerank")
        return None, final_rankings

    return _circo_metric_dict(final_rankings, target_names, all_targets), final_rankings
