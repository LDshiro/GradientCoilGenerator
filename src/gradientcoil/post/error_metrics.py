from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gradientcoil.physics.emdm import build_A_xyz
from gradientcoil.surfaces.base import SurfaceGrid


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    if q <= 0.0:
        return float(np.min(values))
    if q >= 1.0:
        return float(np.max(values))

    mask = weights > 0.0
    if not np.any(mask):
        return 0.0
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    target = float(q) * float(cum[-1])
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), v.shape[0] - 1)
    return float(v[idx])


def _safe_hist_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == 0:
        return np.linspace(0.0, 1.0, bins + 1, dtype=float)
    upper = float(np.max(values))
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 1.0
    return np.linspace(0.0, upper, bins + 1, dtype=float)


@dataclass(frozen=True)
class BzErrorDataset:
    bz_pred: np.ndarray
    bz_error_abs: np.ndarray
    bz_error_valid_mask: np.ndarray
    bz_error_tau: float
    bz_error_hist_edges: np.ndarray
    bz_error_hist_counts_weighted: np.ndarray
    bz_error_hist_counts_unweighted: np.ndarray
    bz_error_valid_count: int
    bz_error_total_count: int
    bz_error_mean: float
    bz_error_median: float
    bz_error_p95: float
    bz_error_max: float
    bz_error_weighted_mean: float
    bz_error_weighted_median: float
    bz_error_weighted_p95: float
    bz_error_weighted_max: float

    def to_npz_payload(self) -> dict[str, np.ndarray | float | int]:
        return {
            "bz_pred": self.bz_pred,
            "bz_error_abs": self.bz_error_abs,
            "bz_error_valid_mask": self.bz_error_valid_mask,
            "bz_error_tau": float(self.bz_error_tau),
            "bz_error_hist_edges": self.bz_error_hist_edges,
            "bz_error_hist_counts_weighted": self.bz_error_hist_counts_weighted,
            "bz_error_hist_counts_unweighted": self.bz_error_hist_counts_unweighted,
            "bz_error_valid_count": int(self.bz_error_valid_count),
            "bz_error_total_count": int(self.bz_error_total_count),
            "bz_error_mean": float(self.bz_error_mean),
            "bz_error_median": float(self.bz_error_median),
            "bz_error_p95": float(self.bz_error_p95),
            "bz_error_max": float(self.bz_error_max),
            "bz_error_weighted_mean": float(self.bz_error_weighted_mean),
            "bz_error_weighted_median": float(self.bz_error_weighted_median),
            "bz_error_weighted_p95": float(self.bz_error_weighted_p95),
            "bz_error_weighted_max": float(self.bz_error_weighted_max),
        }


def compute_bz_error_dataset(
    points: np.ndarray,
    bz_target: np.ndarray,
    s_opt: np.ndarray,
    surfaces: list[SurfaceGrid],
    emdm_mode: str,
    *,
    roi_weights: np.ndarray | None = None,
    cache_dir: str | Path | None = None,
    hist_bins: int = 50,
    zero_threshold_factor: float = 1e-9,
) -> BzErrorDataset:
    if hist_bins < 1:
        raise ValueError("hist_bins must be >= 1.")
    if zero_threshold_factor < 0.0:
        raise ValueError("zero_threshold_factor must be >= 0.")

    pts = np.asarray(points, dtype=float)
    bz = np.asarray(bz_target, dtype=float).reshape(-1)
    s_vec = np.asarray(s_opt, dtype=float).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (P,3).")
    if bz.shape[0] != pts.shape[0]:
        raise ValueError("bz_target length must match points.")

    if roi_weights is None:
        weights = np.ones((pts.shape[0],), dtype=float)
    else:
        weights = np.asarray(roi_weights, dtype=float).reshape(-1)
        if weights.shape[0] != pts.shape[0]:
            raise ValueError("roi_weights length must match points.")
        if np.any(weights < 0.0):
            raise ValueError("roi_weights must be non-negative.")

    _, _, Az = build_A_xyz(pts, surfaces, mode=emdm_mode, cache_dir=cache_dir)
    bz_pred = np.asarray(Az @ s_vec, dtype=float).reshape(-1)
    if bz_pred.shape[0] != pts.shape[0]:
        raise ValueError("Predicted Bz length mismatch.")
    if not np.all(np.isfinite(bz_pred)):
        raise ValueError("Predicted Bz contains non-finite values.")

    bz_error_abs = np.abs(bz_pred - bz)
    max_abs_target = float(np.max(np.abs(bz))) if bz.size > 0 else 0.0
    tau = float(zero_threshold_factor) * max_abs_target
    valid_mask = np.abs(bz) > tau

    valid_error = bz_error_abs[valid_mask]
    valid_weights = weights[valid_mask]
    hist_edges = _safe_hist_edges(valid_error, hist_bins)
    hist_counts_unweighted, _ = np.histogram(valid_error, bins=hist_edges)
    hist_counts_weighted, _ = np.histogram(valid_error, bins=hist_edges, weights=valid_weights)

    if valid_error.size > 0:
        error_mean = float(np.mean(valid_error))
        error_median = float(np.median(valid_error))
        error_p95 = float(np.quantile(valid_error, 0.95))
        error_max = float(np.max(valid_error))
    else:
        error_mean = 0.0
        error_median = 0.0
        error_p95 = 0.0
        error_max = 0.0

    sum_valid_weights = float(np.sum(valid_weights))
    if valid_error.size > 0 and sum_valid_weights > 0.0:
        weighted_mean = float(np.sum(valid_error * valid_weights) / sum_valid_weights)
        weighted_median = _weighted_quantile(valid_error, valid_weights, 0.5)
        weighted_p95 = _weighted_quantile(valid_error, valid_weights, 0.95)
        weighted_max = float(np.max(valid_error))
    else:
        weighted_mean = 0.0
        weighted_median = 0.0
        weighted_p95 = 0.0
        weighted_max = 0.0

    return BzErrorDataset(
        bz_pred=bz_pred,
        bz_error_abs=bz_error_abs,
        bz_error_valid_mask=valid_mask,
        bz_error_tau=tau,
        bz_error_hist_edges=np.asarray(hist_edges, dtype=float),
        bz_error_hist_counts_weighted=np.asarray(hist_counts_weighted, dtype=float),
        bz_error_hist_counts_unweighted=np.asarray(hist_counts_unweighted, dtype=np.int64),
        bz_error_valid_count=int(np.sum(valid_mask)),
        bz_error_total_count=int(bz_error_abs.shape[0]),
        bz_error_mean=error_mean,
        bz_error_median=error_median,
        bz_error_p95=error_p95,
        bz_error_max=error_max,
        bz_error_weighted_mean=weighted_mean,
        bz_error_weighted_median=weighted_median,
        bz_error_weighted_p95=weighted_p95,
        bz_error_weighted_max=weighted_max,
    )
