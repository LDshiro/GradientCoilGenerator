from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.sparse import block_diag

from gradientcoil.operators.gradient import build_gradient_operator
from gradientcoil.physics.emdm import build_A_xyz
from gradientcoil.surfaces.base import SurfaceGrid


@dataclass
class SocpBzSpec:
    use_tv: bool = False
    lambda_tv: float = 0.0
    use_pitch: bool = False
    J_max: float = 0.0
    use_power: bool = False
    lambda_pwr: float = 0.0
    r_sheet: float = 1.0
    gradient_scheme_pitch: str = "forward"
    gradient_scheme_tv: str = "forward"
    gradient_scheme_power: str = "forward"
    gradient_rows_pitch: str = "active"
    gradient_rows_tv: str = "active"
    gradient_rows_power: str = "active"
    emdm_mode: str = "shared"
    verbose: bool = False
    max_iter: int | None = None
    time_limit: float | None = None


@dataclass
class SocpBzResult:
    status: str
    objective: float | None
    s_opt: np.ndarray
    S_grids: list[np.ndarray]
    solver_stats: dict


def _solver_stats_dict(stats: object | None) -> dict:
    if stats is None:
        return {}
    out: dict = {}
    for name in dir(stats):
        if name.startswith("_"):
            continue
        val = getattr(stats, name)
        if callable(val):
            continue
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[name] = val
        elif isinstance(val, np.ndarray):
            out[name] = val.tolist()
    return out


def _build_gradient_block(
    surfaces: list[SurfaceGrid],
    *,
    rows_mode: str,
    emdm_mode: str,
    scheme: str,
) -> tuple[object, np.ndarray]:
    if emdm_mode == "shared":
        op = build_gradient_operator(surfaces[0], rows=rows_mode, scheme=scheme)
        areas = surfaces[0].areas_uv[op.row_coords[:, 0], op.row_coords[:, 1]]
        return op.D, areas

    ops = [build_gradient_operator(surface, rows=rows_mode, scheme=scheme) for surface in surfaces]
    D = block_diag([op.D for op in ops], format="csr")
    area_list = [
        surface.areas_uv[op.row_coords[:, 0], op.row_coords[:, 1]]
        for surface, op in zip(surfaces, ops, strict=True)
    ]
    areas = np.concatenate(area_list) if area_list else np.zeros((0,), dtype=float)
    return D, areas


def solve_socp_bz(
    points: np.ndarray,
    bz_target: np.ndarray,
    surfaces: list[SurfaceGrid],
    spec: SocpBzSpec,
    *,
    roi_weights: np.ndarray | None = None,
    cache_dir: str | None = None,
) -> SocpBzResult:
    if not surfaces:
        raise ValueError("surfaces must be a non-empty list.")
    if spec.emdm_mode not in {"shared", "concat"}:
        raise ValueError("emdm_mode must be 'shared' or 'concat'.")

    if spec.emdm_mode == "shared":
        nints = {s.Nint for s in surfaces}
        if len(nints) != 1:
            raise ValueError("All surfaces must have same Nint for shared mode.")
        n_unknown = surfaces[0].Nint
    else:
        n_unknown = int(sum(s.Nint for s in surfaces))

    points = np.asarray(points, dtype=float)
    bz = np.asarray(bz_target, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (P, 3).")
    if bz.shape[0] != points.shape[0]:
        raise ValueError("bz_target length must match points.")

    if roi_weights is None:
        weights = np.ones((points.shape[0],), dtype=float)
    else:
        weights = np.asarray(roi_weights, dtype=float).reshape(-1)
        if weights.shape[0] != points.shape[0]:
            raise ValueError("roi_weights length must match points.")

    _, _, Az = build_A_xyz(
        points,
        surfaces,
        mode=spec.emdm_mode,
        cache_dir=cache_dir,
    )

    s = cp.Variable(n_unknown)
    t = cp.Variable(points.shape[0], nonneg=True)

    Azs = Az @ s
    constraints = [Azs - bz <= t, -(Azs - bz) <= t]

    obj_terms = [cp.sum(cp.multiply(weights, t))]

    if spec.use_pitch:
        D_pitch, _ = _build_gradient_block(
            surfaces,
            rows_mode=spec.gradient_rows_pitch,
            emdm_mode=spec.emdm_mode,
            scheme=spec.gradient_scheme_pitch,
        )
        g_pitch = D_pitch @ s
        nrows_pitch = g_pitch.shape[0] // 2
        for k in range(nrows_pitch):
            constraints.append(
                cp.norm(cp.hstack([g_pitch[2 * k], g_pitch[2 * k + 1]])) <= spec.J_max
            )

    if spec.use_tv:
        rows_tv = spec.gradient_rows_tv
        if spec.gradient_scheme_tv == "forward":
            rows_tv = "interior"
        D_tv, _ = _build_gradient_block(
            surfaces,
            rows_mode=rows_tv,
            emdm_mode=spec.emdm_mode,
            scheme=spec.gradient_scheme_tv,
        )
        g_tv = D_tv @ s
        nrows_tv = g_tv.shape[0] // 2
        u = cp.Variable(nrows_tv, nonneg=True)
        for k in range(nrows_tv):
            constraints.append(cp.norm(cp.hstack([g_tv[2 * k], g_tv[2 * k + 1]])) <= u[k])
        obj_terms.append(spec.lambda_tv * cp.sum(u))

    if spec.use_power:
        D_pwr, areas = _build_gradient_block(
            surfaces,
            rows_mode=spec.gradient_rows_power,
            emdm_mode=spec.emdm_mode,
            scheme=spec.gradient_scheme_power,
        )
        g_pwr = D_pwr @ s
        nrows_pwr = g_pwr.shape[0] // 2
        Wsqrt = np.sqrt(2.0 * spec.r_sheet * np.repeat(areas, 2))
        if Wsqrt.shape[0] != 2 * nrows_pwr:
            raise ValueError("Power weights size mismatch.")
        p = cp.Variable(nonneg=True)
        constraints.append(cp.norm(cp.multiply(Wsqrt, g_pwr), 2) <= p)
        obj_terms.append(spec.lambda_pwr * p)

    objective = cp.Minimize(cp.sum(cp.hstack(obj_terms)))
    problem = cp.Problem(objective, constraints)

    if "CLARABEL" not in cp.installed_solvers():
        raise RuntimeError("CLARABEL solver is not available.")

    solver_opts: dict = {}
    if spec.max_iter is not None:
        solver_opts["max_iter"] = int(spec.max_iter)
    if spec.time_limit is not None:
        solver_opts["time_limit"] = float(spec.time_limit)

    problem.solve(solver="CLARABEL", verbose=spec.verbose, **solver_opts)

    if s.value is None:
        raise RuntimeError("Solver returned no solution.")

    s_opt = np.asarray(s.value).reshape(-1)
    S_grids: list[np.ndarray] = []
    if spec.emdm_mode == "shared":
        for surface in surfaces:
            S_grids.append(surface.unpack(s_opt, boundary_value=0.0, outside_value=np.nan))
    else:
        offset = 0
        for surface in surfaces:
            n = surface.Nint
            s_i = s_opt[offset : offset + n]
            S_grids.append(surface.unpack(s_i, boundary_value=0.0, outside_value=np.nan))
            offset += n

    return SocpBzResult(
        status=str(problem.status),
        objective=float(problem.value) if problem.value is not None else None,
        s_opt=s_opt,
        S_grids=S_grids,
        solver_stats=_solver_stats_dict(problem.solver_stats),
    )
