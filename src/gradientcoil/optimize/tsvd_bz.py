from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import svds

from gradientcoil.physics.emdm import build_A_xyz
from gradientcoil.surfaces.base import SurfaceGrid


@dataclass
class TsvdBzSpec:
    k: int
    svd_method: str = "full"  # {"full","svds"}
    emdm_mode: str = "shared"


@dataclass
class TsvdBzResult:
    status: str  # "ok"
    objective: float
    s_opt: np.ndarray
    S_grids: list[np.ndarray]
    solver_stats: dict


def _validate_unknowns(surfaces: list[SurfaceGrid], emdm_mode: str) -> int:
    if not surfaces:
        raise ValueError("surfaces must be a non-empty list.")
    if emdm_mode not in {"shared", "concat"}:
        raise ValueError("emdm_mode must be 'shared' or 'concat'.")
    if emdm_mode == "shared":
        nints = {s.Nint for s in surfaces}
        if len(nints) != 1:
            raise ValueError("All surfaces must have same Nint for shared mode.")
        return int(surfaces[0].Nint)
    return int(sum(s.Nint for s in surfaces))


def _unpack_s_grids(
    surfaces: list[SurfaceGrid], s_opt: np.ndarray, *, emdm_mode: str
) -> list[np.ndarray]:
    grids: list[np.ndarray] = []
    if emdm_mode == "shared":
        for surface in surfaces:
            grids.append(surface.unpack(s_opt, boundary_value=0.0, outside_value=np.nan))
        return grids

    offset = 0
    for surface in surfaces:
        s_i = s_opt[offset : offset + surface.Nint]
        grids.append(surface.unpack(s_i, boundary_value=0.0, outside_value=np.nan))
        offset += surface.Nint
    return grids


def solve_tsvd_bz(
    points: np.ndarray,
    bz_target: np.ndarray,
    surfaces: list[SurfaceGrid],
    spec: TsvdBzSpec,
    *,
    roi_weights: np.ndarray | None = None,
    cache_dir: str | None = None,
) -> TsvdBzResult:
    n_unknown = _validate_unknowns(surfaces, spec.emdm_mode)
    if int(spec.k) < 1:
        raise ValueError("k must be >= 1.")
    if spec.svd_method not in {"full", "svds"}:
        raise ValueError("svd_method must be 'full' or 'svds'.")

    pts = np.asarray(points, dtype=float)
    bz = np.asarray(bz_target, dtype=float).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (P, 3).")
    if bz.shape[0] != pts.shape[0]:
        raise ValueError("bz_target length must match points.")

    if roi_weights is None:
        w = np.ones((pts.shape[0],), dtype=float)
    else:
        w = np.asarray(roi_weights, dtype=float).reshape(-1)
        if w.shape[0] != pts.shape[0]:
            raise ValueError("roi_weights length must match points.")
    if np.any(w < 0.0):
        raise ValueError("roi_weights must be non-negative.")

    _, _, Az = build_A_xyz(pts, surfaces, mode=spec.emdm_mode, cache_dir=cache_dir)
    Az = np.asarray(Az, dtype=float)
    if Az.shape != (pts.shape[0], n_unknown):
        raise ValueError("Az shape mismatch.")

    sqrtw = np.sqrt(w)
    A_w = sqrtw[:, None] * Az
    b_w = sqrtw * bz

    if spec.svd_method == "full":
        U, svals, Vt = np.linalg.svd(A_w, full_matrices=False)
        k_eff = min(int(spec.k), int(len(svals)))
        U_k = U[:, :k_eff]
        s_k = svals[:k_eff]
        Vt_k = Vt[:k_eff, :]
    else:
        min_dim = int(min(A_w.shape))
        if min_dim <= 1:
            raise ValueError("svds requires min(A.shape) > 1.")
        k_eff = min(int(spec.k), min_dim - 1)
        U_k, s_k, Vt_k = svds(A_w, k=k_eff, which="LM")
        order = np.argsort(s_k)[::-1]
        U_k = U_k[:, order]
        s_k = s_k[order]
        Vt_k = Vt_k[order, :]

    if np.any(s_k <= 0.0):
        raise ValueError("Non-positive singular values encountered in TSVD.")

    coeff = (U_k.T @ b_w) / s_k
    s_opt = Vt_k.T @ coeff
    s_opt = np.asarray(s_opt, dtype=float).reshape(-1)
    objective = float(np.sum((A_w @ s_opt - b_w) ** 2))

    S_grids = _unpack_s_grids(surfaces, s_opt, emdm_mode=spec.emdm_mode)
    solver_stats = {
        "k": int(k_eff),
        "svd_method": str(spec.svd_method),
        "singular_values": np.asarray(s_k, dtype=float).tolist(),
        "emdm_mode": str(spec.emdm_mode),
    }
    return TsvdBzResult(
        status="ok",
        objective=objective,
        s_opt=s_opt,
        S_grids=S_grids,
        solver_stats=solver_stats,
    )
