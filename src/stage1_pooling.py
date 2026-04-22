from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Stage1PoolResult:
    merged_names: List[List[str]]
    merged_score_maps: List[Dict[str, float]]
    text_score_maps: List[Dict[str, float]]
    proxy_score_maps: List[Dict[str, float]]
    lambda_weight: float


def _to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _canon_name(x) -> str:
    s = _to_str(x)
    stripped = s.lstrip("0")
    return stripped if stripped != "" else "0"


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def _build_name_to_index(index_names: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, name in enumerate(index_names):
        s = _to_str(name)
        c = _canon_name(name)
        if s not in mapping:
            mapping[s] = idx
        if c not in mapping:
            mapping[c] = idx
    return mapping


def _gather_query_image_features(
    reference_names: Sequence[str],
    index_names: Sequence[str],
    index_features: torch.Tensor,
) -> torch.Tensor:
    name_to_idx = _build_name_to_index(index_names)
    rows = []
    missing = []
    for name in reference_names:
        key_exact = _to_str(name)
        key_canon = _canon_name(name)
        idx = name_to_idx.get(key_exact, name_to_idx.get(key_canon, None))
        if idx is None:
            missing.append(key_exact)
            rows.append(index_features.new_zeros((index_features.shape[-1],)))
        else:
            rows.append(index_features[idx])
    if missing:
        raise KeyError(f"Failed to locate {len(missing)} reference images in index_names; examples: {missing[:5]}")
    return torch.stack(rows, dim=0)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    denom = max(vmax - vmin, 1e-8)
    return ((values - vmin) / denom).astype(np.float32)


def _topk_score_map(
    scores_row: torch.Tensor,
    index_names: Sequence[str],
    topk: int,
) -> Dict[str, float]:
    k = min(int(topk), int(scores_row.shape[0]))
    vals, inds = torch.topk(scores_row, k=k, dim=-1)
    vals = vals.detach().cpu().numpy().astype(np.float32)
    inds = inds.detach().cpu().numpy().astype(np.int64)
    vals = _minmax_normalize(vals)
    return {_to_str(index_names[i]): float(v) for i, v in zip(inds, vals)}


def _all_rankings(scores: torch.Tensor, index_names: Sequence[str]) -> List[List[str]]:
    inds = torch.argsort(scores, dim=-1, descending=True)
    inds_np = inds.detach().cpu().numpy()
    all_names = []
    index_names_list = [_to_str(x) for x in index_names]
    for row in inds_np:
        all_names.append([index_names_list[i] for i in row.tolist()])
    return all_names


def _pick_work_device(*tensors: torch.Tensor) -> torch.device:
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            return t.device
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def _move_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.device == device:
        return x
    return x.to(device, non_blocking=True)


def build_ipcir_stage1_pool(
    reference_names: Sequence[str],
    target_caption_features: torch.Tensor,
    proxy_image_features: torch.Tensor,
    query_caption_features: torch.Tensor,
    index_features: torch.Tensor,
    index_names: Sequence[str],
    lambda_weight: float = 0.3,
    prior_topk: int = 100,
) -> Stage1PoolResult:
    """
    Strict IP-CIR stage-1 retrieval:

    1) f_s = f_t - f_o
    2) f_RP = f_p + max(f_p)/max(f_q) * f_q + max(f_p)/max(f_s) * f_s
    3) S_p = sim(f_RP, gallery), S_t = sim(f_t, gallery)
    4) S_b = S_t * S_p
    5) S_f = lambda * S_t + (1 - lambda) * S_b
    """
    work_device = _pick_work_device(
        target_caption_features,
        proxy_image_features,
        query_caption_features,
        index_features,
    )

    target_caption_features = _move_to_device(target_caption_features, work_device)
    proxy_image_features = _move_to_device(proxy_image_features, work_device)
    query_caption_features = _move_to_device(query_caption_features, work_device)
    index_features = _move_to_device(index_features, work_device)

    gallery = _normalize_rows(index_features)
    f_t = _normalize_rows(target_caption_features)
    f_p = _normalize_rows(proxy_image_features)
    f_o = _normalize_rows(query_caption_features)

    f_q = _gather_query_image_features(reference_names, index_names, gallery)
    f_q = _move_to_device(f_q, work_device)
    f_q = _normalize_rows(f_q)

    f_s = f_t - f_o

    # Follow the paper's RP construction using max-based rescaling.
    max_fp = torch.amax(f_p, dim=-1, keepdim=True)
    max_fq = torch.amax(f_q, dim=-1, keepdim=True).clamp_min(1e-8)
    max_fs = torch.amax(torch.abs(f_s), dim=-1, keepdim=True).clamp_min(1e-8)

    f_rp = f_p + (max_fp / max_fq) * f_q + (max_fp / max_fs) * f_s
    f_rp = _normalize_rows(f_rp)

    s_t = torch.matmul(f_t, gallery.T)
    s_p = torch.matmul(f_rp, gallery.T)
    s_b = s_t * s_p
    s_f = float(lambda_weight) * s_t + (1.0 - float(lambda_weight)) * s_b

    merged_names = _all_rankings(s_f, index_names)
    merged_score_maps = []
    text_score_maps = []
    proxy_score_maps = []
    topk = int(prior_topk)

    for i in range(s_f.shape[0]):
        merged_score_maps.append(_topk_score_map(s_f[i], index_names, topk))
        text_score_maps.append(_topk_score_map(s_t[i], index_names, topk))
        proxy_score_maps.append(_topk_score_map(s_p[i], index_names, topk))

    return Stage1PoolResult(
        merged_names=merged_names,
        merged_score_maps=merged_score_maps,
        text_score_maps=text_score_maps,
        proxy_score_maps=proxy_score_maps,
        lambda_weight=float(lambda_weight),
    )
