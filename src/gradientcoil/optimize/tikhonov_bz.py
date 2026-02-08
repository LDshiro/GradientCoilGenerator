from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from gradientcoil.optimize.blocks import build_gradient_block
from gradientcoil.physics.emdm import build_A_xyz
from gradientcoil.surfaces.base import SurfaceGrid


@dataclass
class TikhonovBzSpec:
    lambda_reg: float = 0.0
    reg_operator: str = "grad"  # {"identity","grad","power"}
    r_sheet: float = 1.0  # used when reg_operator=="power"
    gradient_rows_reg: str = "interior"
    emdm_mode: str = "shared"  # {"shared","concat"}
    cg_tol: float = 1e-10
    cg_maxiter: int = 2000


@dataclass
class TikhonovBzResult:
    status: str  # "converged" / "not_converged" / "lstsq"
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


def solve_tikhonov_bz(
    points: np.ndarray,
    bz_target: np.ndarray,
    surfaces: list[SurfaceGrid],
    spec: TikhonovBzSpec,
    *,
    roi_weights: np.ndarray | None = None,
    cache_dir: str | None = None,
) -> TikhonovBzResult:
    n_unknown = _validate_unknowns(surfaces, spec.emdm_mode)
    if spec.lambda_reg < 0.0:
        raise ValueError("lambda_reg must be >= 0.")
    if spec.reg_operator not in {"identity", "grad", "power"}:
        raise ValueError("reg_operator must be 'identity', 'grad', or 'power'.")
    if spec.gradient_rows_reg not in {"interior", "active"}:
        raise ValueError("gradient_rows_reg must be 'interior' or 'active'.")

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

    lam = float(spec.lambda_reg)

    D = None
    wd = None
    Wsqrt = None
    if lam > 0.0 and spec.reg_operator in {"grad", "power"}:
        D, areas = build_gradient_block(
            surfaces,
            rows_mode=spec.gradient_rows_reg,
            emdm_mode=spec.emdm_mode,
        )
        nrows = int(D.shape[0])
        if spec.reg_operator == "grad":
            wd = np.ones((nrows,), dtype=float)
        else:
            Wsqrt = np.sqrt(2.0 * float(spec.r_sheet) * np.repeat(areas, 2))
            wd = Wsqrt**2
        if wd.shape[0] != nrows:
            raise ValueError("Regularization weights size mismatch.")

    if lam == 0.0:
        sqrtw = np.sqrt(w)
        A_w = sqrtw[:, None] * Az
        b_w = sqrtw * bz
        s_opt = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
        status = "lstsq"
        solver_stats = {"method": "lstsq", "emdm_mode": spec.emdm_mode}
    else:
        rhs = Az.T @ (w * bz)

        def matvec(x: np.ndarray) -> np.ndarray:
            y = Az.T @ (w * (Az @ x))
            if spec.reg_operator == "identity":
                y = y + (lam**2) * x
            else:
                if D is None or wd is None:
                    raise RuntimeError("Regularization operator not initialized.")
                Dx = D @ x
                y = y + (lam**2) * (D.T @ (wd * Dx))
            return y

        H = LinearOperator((n_unknown, n_unknown), matvec=matvec, dtype=float)
        try:
            s_opt, info = cg(
                H,
                rhs,
                maxiter=int(spec.cg_maxiter),
                rtol=float(spec.cg_tol),
                atol=0.0,
            )
        except TypeError:
            s_opt, info = cg(
                H,
                rhs,
                maxiter=int(spec.cg_maxiter),
                tol=float(spec.cg_tol),
            )
        status = "converged" if info == 0 else "not_converged"
        solver_stats = {
            "cg_info": int(info),
            "cg_tol": float(spec.cg_tol),
            "cg_maxiter": int(spec.cg_maxiter),
            "lambda_reg": float(spec.lambda_reg),
            "reg_operator": str(spec.reg_operator),
            "emdm_mode": str(spec.emdm_mode),
        }

    s_opt = np.asarray(s_opt, dtype=float).reshape(-1)
    data_term = float(np.sum(w * (Az @ s_opt - bz) ** 2))
    if spec.reg_operator == "identity":
        reg_term = float(np.sum(s_opt**2))
    elif D is None:
        reg_term = 0.0
    else:
        Ds = D @ s_opt
        if spec.reg_operator == "grad":
            reg_term = float(np.sum(Ds**2))
        else:
            if wd is None:
                raise RuntimeError("Power weights are not initialized.")
            reg_term = float(np.sum(wd * (Ds**2)))
    objective = data_term + (lam**2) * reg_term

    S_grids = _unpack_s_grids(surfaces, s_opt, emdm_mode=spec.emdm_mode)
    return TikhonovBzResult(
        status=status,
        objective=objective,
        s_opt=s_opt,
        S_grids=S_grids,
        solver_stats=solver_stats,
    )
