
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def get_time():
    import datetime
    return datetime.datetime.now().strftime("%Y.%m.%d-%H_%M_%S")


def _to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _listify(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [_to_str(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_str(v) for v in x]
    return [_to_str(x)]


def _normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = np.asarray(list(score_map.values()), dtype=np.float32)
    vmin = float(vals.min())
    vmax = float(vals.max())
    denom = max(vmax - vmin, 1e-8)
    return {k: float((v - vmin) / denom) for k, v in score_map.items()}


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_submission_dict(rankings: Sequence[Sequence[str]], query_ids: Sequence[Any], topk: int) -> Dict[str, List[str]]:
    out = {}
    for qid, ranking in zip(query_ids, rankings):
        out[str(qid)] = [_to_str(x) for x in list(ranking)[:topk]]
    return out


def _stage1_test_save_cirr(kwargs, filtered_rankings):
    out_dir = os.path.join(kwargs["dataset_path"], "task", kwargs["task"], "submission")
    os.makedirs(out_dir, exist_ok=True)
    ts = get_time()
    save_path = os.path.join(out_dir, f"test_submission_{kwargs['clip']}_merged_loop_0_{ts}.json")
    subset_save_path = os.path.join(out_dir, f"subset_submission_{kwargs['clip']}_merged_loop_0_{ts}.json")
    standard = {str(qid): rank[:50] for qid, rank in zip(kwargs["query_ids"], filtered_rankings)}
    subset = {}
    for qid, rank, group in zip(kwargs["query_ids"], filtered_rankings, kwargs["targets"]):
        group_set = {_to_str(x) for x in group}
        subset[str(qid)] = [x for x in rank if x in group_set][:3]
    _save_json(save_path, standard)
    _save_json(subset_save_path, subset)
    print(f"Saved CIRR merged stage1 submission to {save_path}")
    print(f"Saved CIRR merged stage1 subset submission to {subset_save_path}")


def _stage1_test_save_circo(kwargs, rankings):
    out_dir = os.path.join(kwargs["dataset_path"], "task", kwargs["task"], "submission")
    os.makedirs(out_dir, exist_ok=True)
    ts = get_time()
    save_path = os.path.join(out_dir, f"test_submission_{kwargs['clip']}_merged_loop_0_{ts}.json")
    submission = {str(qid): rank[:50] for qid, rank in zip(kwargs["query_ids"], rankings)}
    _save_json(save_path, submission)
    print(f"Saved CIRCO merged stage1 submission to {save_path}")


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
        if len(x) == 2:
            try:
                return _extract_numeric_score(x[1])
            except Exception:
                pass
            try:
                return _extract_numeric_score(x[0])
            except Exception:
                pass
        for item in x:
            try:
                return _extract_numeric_score(item)
            except Exception:
                continue
    raise TypeError(f"Cannot extract numeric score from object of type {type(x).__name__}: {x!r}")


def _build_conf_map(names_entry: Any, conf_entry: Any) -> Dict[str, float]:
    """
    Fallback only. confidences1/2 in this repo are often branch-level reliabilities,
    not per-candidate ranking scores. Use rank-based maps first.
    """
    names = _listify(names_entry)
    if not names:
        return {}

    if isinstance(conf_entry, Mapping):
        out = {}
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
            score = _extract_numeric_score(conf_entry)
            return {n: score for n in names}

    if len(conf_values) == 0:
        return {}
    if len(conf_values) == 1:
        score = _extract_numeric_score(conf_values[0])
        return {n: score for n in names}
    if len(conf_values) != len(names):
        usable = min(len(conf_values), len(names))
        out = {}
        for i in range(usable):
            try:
                out[names[i]] = _extract_numeric_score(conf_values[i])
            except Exception:
                continue
        return out

    out = {}
    for n, s in zip(names, conf_values):
        try:
            out[n] = _extract_numeric_score(s)
        except Exception:
            continue
    return out


def _build_rank_score_map(names_entry: Any, rank_entry: Any) -> Dict[str, float]:
    """
    Preferred source for reranking. Handles common structures:
    - dict[name] = score
    - list/tuple aligned with candidate names: [score1, score2, ...]
    - list of (name, score)
    - list of (label, score) aligned with candidate names
    """
    names = _listify(names_entry)
    if not names:
        return {}

    # Mapping by candidate name.
    if isinstance(rank_entry, Mapping):
        out = {}
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

    # Case A: aligned list with same length as names.
    if len(values) == len(names):
        out = {}
        ok = 0
        for n, item in zip(names, values):
            try:
                out[n] = _extract_numeric_score(item)
                ok += 1
            except Exception:
                pass
        if ok > 0:
            return out

    # Case B: list of (name, score) pairs.
    out = {}
    for item in values:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            k = _to_str(item[0])
            try:
                v = _extract_numeric_score(item[1])
            except Exception:
                continue
            out[k] = v
    if out:
        filtered = {n: out[n] for n in names if n in out}
        if filtered:
            return filtered

    return {}


def _build_verifier_map(
    names_entry: Any,
    rank_entry: Any,
    conf_entry: Any,
) -> Dict[str, float]:
    # Prefer per-candidate rank scores.
    rank_map = _build_rank_score_map(names_entry, rank_entry)
    if rank_map:
        return _normalize_score_map(rank_map)

    # Fallback to confidence-derived map.
    conf_map = _build_conf_map(names_entry, conf_entry)
    if conf_map:
        return _normalize_score_map(conf_map)
    return {}


def _build_final_rankings(
    args,
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
    final_rankings: List[List[str]] = []

    score_mode = str(getattr(args, "rerank_score_mode", "prior_plus_verifier"))
    prior_w = float(getattr(args, "rerank_prior_weight", 0.50))
    ver1_w = float(getattr(args, "rerank_verifier1_weight", 0.50))
    ver2_w = float(getattr(args, "rerank_verifier2_weight", 0.00))
    dual_verifier_bonus = float(getattr(args, "rerank_dual_verifier_bonus", 0.00))
    dual_retrieval_bonus = float(getattr(args, "rerank_dual_retrieval_bonus", 0.00))
    rank_topk = int(getattr(args, "rank_topk", 50))

    total = len(stage1_pool_names)
    for i in range(total):
        prior_map = _normalize_score_map(dict(stage1_pool_score_maps[i]))
        txt_map = _normalize_score_map(dict(stage1_txt_score_maps[i])) if stage1_txt_score_maps else {}
        img_map = _normalize_score_map(dict(stage1_img_score_maps[i])) if stage1_img_score_maps else {}

        cand1 = candidates1[i] if i < len(candidates1) else []
        cand2 = candidates2[i] if i < len(candidates2) else []
        rank1 = ranks1[i] if i < len(ranks1) else []
        rank2 = ranks2[i] if i < len(ranks2) else []
        conf1 = confidences1[i] if i < len(confidences1) else []
        conf2 = confidences2[i] if i < len(confidences2) else []

        ver1_map = _build_verifier_map(cand1, rank1, conf1)
        ver2_map = _build_verifier_map(cand2, rank2, conf2)

        union = list(dict.fromkeys(_listify(stage1_pool_names[i]) + _listify(cand1) + _listify(cand2)))

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
            n for n, _ in sorted(
                score_map.items(),
                key=lambda kv: (-kv[1], -prior_map.get(kv[0], 0.0), kv[0]),
            )
        ]
        final_rankings.append(ordered[:rank_topk])

    return final_rankings


def _circo_metric_dict(rankings, target_names, all_targets):
    metrics = {}
    mAP_values = {}
    recall_values = {}
    for k in [5, 10, 25, 50]:
        recall_hits = []
        ap_scores = []
        for primary_tgt, pred, gt_list in zip(target_names, rankings, all_targets):
            recall_hits.append(1.0 if primary_tgt in pred[:k] else 0.0)
            gt_set = {_to_str(x) for x in gt_list}
            pred_k = pred[:k]
            running_hits = 0
            precisions = []
            for rank_idx, name in enumerate(pred_k, start=1):
                if name in gt_set:
                    running_hits += 1
                    precisions.append(running_hits / rank_idx)
            denom = max(len(gt_set), 1)
            ap_scores.append(float(sum(precisions) / denom) if precisions else 0.0)
        mAP_values[f"mAP@{k}"] = float(np.mean(ap_scores) * 100.0)
        recall_values[f"recall@{k}"] = float(np.mean(recall_hits) * 100.0)

    # Keep json order: mAP first, then recall.
    for k in [5, 10, 25, 50]:
        metrics[f"mAP@{k}"] = mAP_values[f"mAP@{k}"]
    for k in [5, 10, 25, 50]:
        metrics[f"recall@{k}"] = recall_values[f"recall@{k}"]
    return metrics


def fiq_stage1_pool(**kwargs):
    split = kwargs["split"]
    rankings = [_listify(x) for x in kwargs["stage1_pool_names"]]
    if split == "test":
        return None, rankings

    target_names = _listify(kwargs["target_names"])
    recalls = {}
    for k in [1, 5, 10, 50]:
        hits = []
        for pred, tgt in zip(rankings, target_names):
            hits.append(1.0 if tgt in pred[:k] else 0.0)
        recalls[f"recall@{k}"] = float(np.mean(hits) * 100.0)
    return recalls, rankings


def cirr_stage1_pool(**kwargs):
    split = kwargs["split"]
    rankings = [_listify(x) for x in kwargs["stage1_pool_names"]]
    reference_names = _listify(kwargs["reference_names"])
    filtered_rankings = []
    for ref_name, pred in zip(reference_names, rankings):
        filtered_rankings.append([n for n in pred if n != ref_name])

    if split == "test":
        _stage1_test_save_cirr(kwargs, filtered_rankings)
        return None, filtered_rankings

    target_names = _listify(kwargs["target_names"])
    group_members = kwargs["targets"]
    metrics = {}

    for k in [1, 5, 10, 50]:
        hits = [1.0 if tgt in rank[:k] else 0.0 for tgt, rank in zip(target_names, filtered_rankings)]
        metrics[f"recall@{k}"] = float(np.mean(hits) * 100.0)

    for k in [1, 2, 3]:
        group_hits = []
        for tgt, rank, group in zip(target_names, filtered_rankings, group_members):
            group_set = {_to_str(x) for x in group}
            pred = [x for x in rank if x in group_set][:k]
            group_hits.append(1.0 if tgt in pred else 0.0)
        metrics[f"group_recall@{k}"] = float(np.mean(group_hits) * 100.0)

    return metrics, filtered_rankings


def circo_stage1_pool(**kwargs):
    split = kwargs["split"]
    rankings = [_listify(x) for x in kwargs["stage1_pool_names"]]
    if split == "test":
        _stage1_test_save_circo(kwargs, rankings)
        return None, rankings

    target_names = _listify(kwargs["target_names"])
    all_targets = kwargs["targets"]
    metrics = _circo_metric_dict(rankings, target_names, all_targets)
    return metrics, rankings


def fiq_fuse2paths(**kwargs):
    args = kwargs["args"]
    split = kwargs["split"]
    final_rankings = _build_final_rankings(
        args=args,
        stage1_pool_names=kwargs["stage1_pool_names"],
        stage1_pool_score_maps=kwargs["stage1_pool_score_maps"],
        stage1_txt_score_maps=kwargs.get("stage1_txt_score_maps", []),
        stage1_img_score_maps=kwargs.get("stage1_img_score_maps", []),
        candidates1=kwargs["candidates1"],
        candidates2=kwargs["candidates2"],
        ranks1=kwargs.get("ranks1", []),
        ranks2=kwargs.get("ranks2", []),
        confidences1=kwargs["confidences1"],
        confidences2=kwargs["confidences2"],
    )
    if split == "test":
        return None, final_rankings

    target_names = _listify(kwargs["target_names"])
    recalls = {}
    for k in [1, 5, 10, 50]:
        hits = []
        for pred, tgt in zip(final_rankings, target_names):
            hits.append(1.0 if tgt in pred[:k] else 0.0)
        recalls[f"recall@{k}"] = float(np.mean(hits) * 100.0)
    return recalls, final_rankings


def cirr_fuse2paths(**kwargs):
    args = kwargs["args"]
    split = kwargs["split"]

    final_rankings = _build_final_rankings(
        args=args,
        stage1_pool_names=kwargs["stage1_pool_names"],
        stage1_pool_score_maps=kwargs["stage1_pool_score_maps"],
        stage1_txt_score_maps=kwargs.get("stage1_txt_score_maps", []),
        stage1_img_score_maps=kwargs.get("stage1_img_score_maps", []),
        candidates1=kwargs["candidates1"],
        candidates2=kwargs["candidates2"],
        ranks1=kwargs.get("ranks1", []),
        ranks2=kwargs.get("ranks2", []),
        confidences1=kwargs["confidences1"],
        confidences2=kwargs["confidences2"],
    )

    reference_names = _listify(kwargs["reference_names"])
    target_names = _listify(kwargs["target_names"])
    query_ids = _listify(kwargs["query_ids"])
    group_members = kwargs["targets"]

    filtered_rankings = []
    for ref_name, pred in zip(reference_names, final_rankings):
        filtered_rankings.append([n for n in pred if n != ref_name])

    if split == "test":
        out_dir = os.path.join(kwargs["dataset_path"], "task", kwargs["task"], "submission")
        os.makedirs(out_dir, exist_ok=True)
        ts = get_time()
        save_path = os.path.join(out_dir, f"test_submission_{kwargs['clip']}_{ts}.json")
        subset_save_path = os.path.join(out_dir, f"subset_submission_{kwargs['clip']}_{ts}.json")

        standard = {qid: rank[:50] for qid, rank in zip(query_ids, filtered_rankings)}
        subset = {}
        for qid, rank, group in zip(query_ids, filtered_rankings, group_members):
            group_set = {_to_str(x) for x in group}
            subset[qid] = [x for x in rank if x in group_set][:3]

        _save_json(save_path, standard)
        _save_json(subset_save_path, subset)
        print(f"Saved CIRR test submission to {save_path}")
        print(f"Saved CIRR subset submission to {subset_save_path}")
        return None, filtered_rankings

    metrics = {}
    for k in [1, 5, 10, 50]:
        hits = [1.0 if tgt in rank[:k] else 0.0 for tgt, rank in zip(target_names, filtered_rankings)]
        metrics[f"recall@{k}"] = float(np.mean(hits) * 100.0)

    for k in [1, 2, 3]:
        group_hits = []
        for tgt, rank, group in zip(target_names, filtered_rankings, group_members):
            group_set = {_to_str(x) for x in group}
            pred = [x for x in rank if x in group_set][:k]
            group_hits.append(1.0 if tgt in pred else 0.0)
        metrics[f"group_recall@{k}"] = float(np.mean(group_hits) * 100.0)

    return metrics, filtered_rankings


def circo_fuse2paths(**kwargs):
    args = kwargs["args"]
    split = kwargs["split"]

    final_rankings = _build_final_rankings(
        args=args,
        stage1_pool_names=kwargs["stage1_pool_names"],
        stage1_pool_score_maps=kwargs["stage1_pool_score_maps"],
        stage1_txt_score_maps=kwargs.get("stage1_txt_score_maps", []),
        stage1_img_score_maps=kwargs.get("stage1_img_score_maps", []),
        candidates1=kwargs["candidates1"],
        candidates2=kwargs["candidates2"],
        ranks1=kwargs.get("ranks1", []),
        ranks2=kwargs.get("ranks2", []),
        confidences1=kwargs["confidences1"],
        confidences2=kwargs["confidences2"],
    )

    query_ids = _listify(kwargs["query_ids"])
    target_names = _listify(kwargs["target_names"])
    all_targets = kwargs["targets"]

    if split == "test":
        out_dir = os.path.join(kwargs["dataset_path"], "task", kwargs["task"], "submission")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"test_submission_{kwargs['clip']}_{get_time()}.json")
        submission = {qid: rank[:50] for qid, rank in zip(query_ids, final_rankings)}
        _save_json(save_path, submission)
        print(f"Saved CIRCO test submission to {save_path}")
        return None, final_rankings

    metrics = _circo_metric_dict(final_rankings, target_names, all_targets)
    return metrics, final_rankings
