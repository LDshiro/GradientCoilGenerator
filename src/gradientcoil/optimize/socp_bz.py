from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.sparse import block_diag, coo_matrix, csr_matrix

from gradientcoil.operators.gradient import (
    build_edge_difference_operator,
    build_gradient_operator,
)
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
    use_tgv: bool = False
    alpha1_tgv: float = 1e-6
    alpha0_tgv: float = 1e-6
    tgv_area_weights: bool = True
    gradient_rows_tgv: str = "interior"
    gradient_scheme_tgv: str = "forward"
    use_curv_r1: bool = False
    lambda_curv_r1: float = 0.0
    use_curv_en: bool = False
    lambda_curv_en: float = 0.0
    gradient_scheme_curv: str = "forward"
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


@dataclass
class SocpBzProblemSize:
    n_variables: int
    n_constraints: int
    n_nonneg: int
    variables: dict[str, int]
    constraints: dict[str, int]


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
    if scheme not in {"forward", "central"}:
        raise ValueError("scheme must be 'forward' or 'central' for cell gradients.")
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


def _build_edge_block(
    surfaces: list[SurfaceGrid],
    *,
    rows_mode: str,
    emdm_mode: str,
    bidirectional: bool,
) -> tuple[object, np.ndarray]:
    if emdm_mode == "shared":
        op = build_edge_difference_operator(
            surfaces[0], rows=rows_mode, bidirectional=bidirectional
        )
        return op.D, op.edge_areas

    ops = [
        build_edge_difference_operator(surface, rows=rows_mode, bidirectional=bidirectional)
        for surface in surfaces
    ]
    D = block_diag([op.D for op in ops], format="csr")
    areas = np.concatenate([op.edge_areas for op in ops]) if ops else np.zeros((0,), dtype=float)
    return D, areas


def _build_edge_ops(
    surfaces: list[SurfaceGrid],
    *,
    rows_mode: str,
    emdm_mode: str,
    bidirectional: bool,
) -> list:
    if emdm_mode == "shared":
        return [
            build_edge_difference_operator(surfaces[0], rows=rows_mode, bidirectional=bidirectional)
        ]
    return [
        build_edge_difference_operator(surface, rows=rows_mode, bidirectional=bidirectional)
        for surface in surfaces
    ]


def _build_line_graph_operator(op) -> tuple[csr_matrix, np.ndarray]:
    cell_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for idx, (iu, iv) in enumerate(op.uv0):
        cell_edges.setdefault((int(iu), int(iv)), []).append((idx, 1))
    for idx, k1 in enumerate(op.k1):
        if k1 >= 0:
            iu, iv = op.uv1[idx]
            cell_edges.setdefault((int(iu), int(iv)), []).append((idx, -1))

    rows_idx: list[int] = []
    cols_idx: list[int] = []
    data: list[float] = []
    edge_area_list: list[float] = []

    for edges in cell_edges.values():
        n = len(edges)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                e0, s0 = edges[i]
                e1, s1 = edges[j]
                row = len(edge_area_list)
                rows_idx.extend([row, row])
                cols_idx.extend([e0, e1])
                data.extend([-float(s0), float(s1)])
                edge_area_list.append(0.5 * (op.edge_areas[e0] + op.edge_areas[e1]))

    D_line = coo_matrix(
        (data, (rows_idx, cols_idx)), shape=(len(edge_area_list), op.D.shape[0])
    ).tocsr()
    return D_line, np.asarray(edge_area_list, dtype=float)


def estimate_socp_bz_problem_size(
    points: np.ndarray,
    surfaces: list[SurfaceGrid],
    spec: SocpBzSpec,
) -> SocpBzProblemSize:
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
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (P, 3).")
    n_points = int(points.shape[0])

    variables: dict[str, int] = {"s": n_unknown, "t": n_points}
    constraints: dict[str, int] = {"data_abs": 2 * n_points}
    n_nonneg = n_points

    if spec.use_pitch:
        rows_pitch = spec.gradient_rows_pitch
        if spec.gradient_scheme_pitch == "central":
            rows_pitch = "interior"
        if spec.gradient_scheme_pitch == "edge":
            D_pitch, _ = _build_edge_block(
                surfaces,
                rows_mode=rows_pitch,
                emdm_mode=spec.emdm_mode,
                bidirectional=True,
            )
            constraints["pitch"] = 2 * int(D_pitch.shape[0])
        else:
            D_pitch, _ = _build_gradient_block(
                surfaces,
                rows_mode=rows_pitch,
                emdm_mode=spec.emdm_mode,
                scheme=spec.gradient_scheme_pitch,
            )
            nrows_pitch = int(D_pitch.shape[0] // 2)
            constraints["pitch"] = nrows_pitch

    if spec.use_tv:
        rows_tv = spec.gradient_rows_tv
        if spec.gradient_scheme_tv in {"forward", "central"}:
            rows_tv = "interior"
        if spec.gradient_scheme_tv == "edge":
            D_tv, _ = _build_edge_block(
                surfaces,
                rows_mode=rows_tv,
                emdm_mode=spec.emdm_mode,
                bidirectional=True,
            )
            n_edges = int(D_tv.shape[0])
            variables["tv_u"] = n_edges
            n_nonneg += n_edges
            constraints["tv"] = 2 * n_edges
        else:
            D_tv, _ = _build_gradient_block(
                surfaces,
                rows_mode=rows_tv,
                emdm_mode=spec.emdm_mode,
                scheme=spec.gradient_scheme_tv,
            )
            nrows_tv = int(D_tv.shape[0] // 2)
            variables["tv_u"] = nrows_tv
            n_nonneg += nrows_tv
            constraints["tv"] = nrows_tv

    if spec.use_power:
        variables["pwr_p"] = 1
        n_nonneg += 1
        constraints["power"] = 1

    if spec.use_tgv:
        if spec.alpha1_tgv <= 0.0 or spec.alpha0_tgv <= 0.0:
            raise ValueError("TGV requires alpha1_tgv > 0 and alpha0_tgv > 0.")
        if spec.gradient_rows_tgv != "interior":
            raise ValueError("TGV requires gradient_rows_tgv='interior'.")
        if spec.gradient_scheme_tgv == "edge":
            ops = _build_edge_ops(
                surfaces,
                rows_mode=spec.gradient_rows_tgv,
                emdm_mode=spec.emdm_mode,
                bidirectional=True,
            )
            if spec.emdm_mode == "shared":
                op0 = ops[0]
                n_edges = int(op0.D.shape[0])
                D_line, _ = _build_line_graph_operator(op0)
                n_line = int(D_line.shape[0])
            else:
                n_edges = int(sum(op.D.shape[0] for op in ops))
                n_line = 0
                for op in ops:
                    D_line_i, _ = _build_line_graph_operator(op)
                    n_line += int(D_line_i.shape[0])
            variables["tgv_w"] = n_edges
            variables["tgv_u1"] = n_edges
            variables["tgv_u0"] = n_line
            n_nonneg += n_edges + n_line
            constraints["tgv_u1"] = 2 * n_edges
            constraints["tgv_u0"] = 2 * n_line
        else:
            if spec.gradient_scheme_tgv not in {"forward", "central"}:
                raise ValueError("gradient_scheme_tgv must be 'forward', 'central', or 'edge'.")
            D_tgv, _ = _build_gradient_block(
                surfaces,
                rows_mode=spec.gradient_rows_tgv,
                emdm_mode=spec.emdm_mode,
                scheme=spec.gradient_scheme_tgv,
            )
            nrows = int(D_tgv.shape[0] // 2)
            if nrows != n_unknown:
                raise ValueError("TGV requires rows_mode='interior' so that nrows == n_unknown.")
            variables["tgv_w"] = 2 * nrows
            variables["tgv_u1"] = nrows
            variables["tgv_u0"] = nrows
            n_nonneg += 2 * nrows
            constraints["tgv_u1"] = nrows
            constraints["tgv_u0"] = nrows

    if spec.lambda_curv_r1 < 0.0 or spec.lambda_curv_en < 0.0:
        raise ValueError("Curvature regularizer lambdas must be non-negative.")
    if spec.lambda_curv_r1 < 0.0 or spec.lambda_curv_en < 0.0:
        raise ValueError("Curvature regularizer lambdas must be non-negative.")
    use_curv = spec.use_curv_r1 or spec.use_curv_en
    if use_curv:
        if spec.gradient_scheme_curv == "edge":
            D_curv, _ = _build_edge_block(
                surfaces,
                rows_mode="interior",
                emdm_mode=spec.emdm_mode,
                bidirectional=False,
            )
        else:
            if spec.gradient_scheme_curv not in {"forward", "central"}:
                raise ValueError("gradient_scheme_curv must be 'forward', 'central', or 'edge'.")
            D_curv, _ = _build_gradient_block(
                surfaces,
                rows_mode="interior",
                emdm_mode=spec.emdm_mode,
                scheme=spec.gradient_scheme_curv,
            )
        nrows = int(D_curv.shape[0] // 2)
        if nrows != n_unknown:
            raise ValueError(
                "Curvature regularizer requires rows_mode='interior' so that nrows == n_unknown."
            )
        if spec.use_curv_r1:
            variables["curv_u"] = nrows
            n_nonneg += nrows
            constraints["curv_r1"] = nrows
        if spec.use_curv_en and spec.lambda_curv_en > 0.0:
            constraints["curv_en"] = 0

    n_variables = int(sum(variables.values()))
    n_constraints = int(sum(constraints.values()))
    return SocpBzProblemSize(
        n_variables=n_variables,
        n_constraints=n_constraints,
        n_nonneg=n_nonneg,
        variables=variables,
        constraints=constraints,
    )


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
        rows_pitch = spec.gradient_rows_pitch
        if spec.gradient_scheme_pitch == "central":
            rows_pitch = "interior"
        if spec.gradient_scheme_pitch == "edge":
            D_pitch, _ = _build_edge_block(
                surfaces,
                rows_mode=rows_pitch,
                emdm_mode=spec.emdm_mode,
                bidirectional=True,
            )
            g_pitch = D_pitch @ s
            constraints.append(g_pitch <= spec.J_max)
            constraints.append(-g_pitch <= spec.J_max)
        else:
            D_pitch, _ = _build_gradient_block(
                surfaces,
                rows_mode=rows_pitch,
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
        if spec.gradient_scheme_tv in {"forward", "central"}:
            rows_tv = "interior"
        if spec.gradient_scheme_tv == "edge":
            D_tv, _ = _build_edge_block(
                surfaces,
                rows_mode=rows_tv,
                emdm_mode=spec.emdm_mode,
                bidirectional=True,
            )
            g_tv = D_tv @ s
            u = cp.Variable(g_tv.shape[0], nonneg=True)
            constraints.append(g_tv <= u)
            constraints.append(-g_tv <= u)
            obj_terms.append(spec.lambda_tv * cp.sum(u))
        else:
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
        rows_pwr = spec.gradient_rows_power
        if spec.gradient_scheme_power == "central":
            rows_pwr = "interior"
        if spec.gradient_scheme_power == "edge":
            D_pwr, areas = _build_edge_block(
                surfaces,
                rows_mode=rows_pwr,
                emdm_mode=spec.emdm_mode,
                bidirectional=True,
            )
            g_pwr = D_pwr @ s
            Wsqrt = np.sqrt(2.0 * spec.r_sheet * areas)
            if Wsqrt.shape[0] != g_pwr.shape[0]:
                raise ValueError("Power weights size mismatch.")
            p = cp.Variable(nonneg=True)
            constraints.append(cp.norm(cp.multiply(Wsqrt, g_pwr), 2) <= p)
            obj_terms.append(spec.lambda_pwr * p)
        else:
            D_pwr, areas = _build_gradient_block(
                surfaces,
                rows_mode=rows_pwr,
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

    if spec.use_tgv:
        if spec.alpha1_tgv <= 0.0 or spec.alpha0_tgv <= 0.0:
            raise ValueError("TGV requires alpha1_tgv > 0 and alpha0_tgv > 0.")
        if spec.gradient_rows_tgv != "interior":
            raise ValueError("TGV requires gradient_rows_tgv='interior'.")
        if spec.gradient_scheme_tgv == "edge":
            ops = _build_edge_ops(
                surfaces,
                rows_mode=spec.gradient_rows_tgv,
                emdm_mode=spec.emdm_mode,
                bidirectional=True,
            )
            if spec.emdm_mode == "shared":
                op0 = ops[0]
                D_edge = op0.D
                areas_edge = op0.edge_areas
                D_line, areas_line = _build_line_graph_operator(op0)
            else:
                D_edge = block_diag([op.D for op in ops], format="csr")
                areas_edge = (
                    np.concatenate([op.edge_areas for op in ops])
                    if ops
                    else np.zeros((0,), dtype=float)
                )
                line_blocks: list[csr_matrix] = []
                area_blocks: list[np.ndarray] = []
                for op in ops:
                    D_line_i, areas_line_i = _build_line_graph_operator(op)
                    line_blocks.append(D_line_i)
                    area_blocks.append(areas_line_i)
                if line_blocks:
                    D_line = block_diag(line_blocks, format="csr")
                else:
                    D_line = csr_matrix((0, D_edge.shape[0]))
                areas_line = (
                    np.concatenate(area_blocks) if area_blocks else np.zeros((0,), dtype=float)
                )

            g = D_edge @ s
            w = cp.Variable(D_edge.shape[0])
            u1 = cp.Variable(D_edge.shape[0], nonneg=True)
            constraints.append(g - w <= u1)
            constraints.append(-(g - w) <= u1)

            n_line = int(D_line.shape[0])
            u0 = None
            if n_line > 0:
                u0 = cp.Variable(n_line, nonneg=True)
                g_line = D_line @ w
                constraints.append(g_line <= u0)
                constraints.append(-g_line <= u0)

            if spec.tgv_area_weights:
                term1 = spec.alpha1_tgv * cp.sum(cp.multiply(areas_edge, u1))
                if n_line > 0 and u0 is not None:
                    term0 = spec.alpha0_tgv * cp.sum(cp.multiply(areas_line, u0))
                else:
                    term0 = 0.0
            else:
                term1 = spec.alpha1_tgv * cp.sum(u1)
                term0 = spec.alpha0_tgv * cp.sum(u0) if n_line > 0 and u0 is not None else 0.0
            obj_terms.append(term1 + term0)
        else:
            if spec.gradient_scheme_tgv not in {"forward", "central"}:
                raise ValueError("gradient_scheme_tgv must be 'forward', 'central', or 'edge'.")
            D_tgv, areas_tgv = _build_gradient_block(
                surfaces,
                rows_mode=spec.gradient_rows_tgv,
                emdm_mode=spec.emdm_mode,
                scheme=spec.gradient_scheme_tgv,
            )
            nrows = D_tgv.shape[0] // 2
            if D_tgv.shape[0] != 2 * nrows or D_tgv.shape[1] != n_unknown:
                raise ValueError("TGV gradient operator shape mismatch.")
            if nrows != n_unknown:
                raise ValueError("TGV requires rows_mode='interior' so that nrows == n_unknown.")

            g = D_tgv @ s
            w = cp.Variable(2 * nrows)
            u1 = cp.Variable(nrows, nonneg=True)
            u0 = cp.Variable(nrows, nonneg=True)

            for k in range(nrows):
                constraints.append(
                    cp.norm(cp.hstack([g[2 * k] - w[2 * k], g[2 * k + 1] - w[2 * k + 1]]), 2)
                    <= u1[k]
                )

            w_u = w[0::2]
            w_v = w[1::2]
            Dw_u = D_tgv @ w_u
            Dw_v = D_tgv @ w_v
            sqrt2 = float(np.sqrt(2.0))
            for k in range(nrows):
                du_wu = Dw_u[2 * k]
                dv_wu = Dw_u[2 * k + 1]
                du_wv = Dw_v[2 * k]
                dv_wv = Dw_v[2 * k + 1]
                e11 = du_wu
                e22 = dv_wv
                e12 = 0.5 * (dv_wu + du_wv)
                constraints.append(cp.norm(cp.hstack([e11, e22, sqrt2 * e12]), 2) <= u0[k])

            if spec.tgv_area_weights:
                tgv_term = spec.alpha1_tgv * cp.sum(
                    cp.multiply(areas_tgv, u1)
                ) + spec.alpha0_tgv * cp.sum(cp.multiply(areas_tgv, u0))
            else:
                tgv_term = spec.alpha1_tgv * cp.sum(u1) + spec.alpha0_tgv * cp.sum(u0)
            obj_terms.append(tgv_term)

    use_curv = spec.use_curv_r1 or spec.use_curv_en
    if use_curv:
        if spec.gradient_scheme_curv == "edge":
            D_curv, _ = _build_edge_block(
                surfaces,
                rows_mode="interior",
                emdm_mode=spec.emdm_mode,
                bidirectional=False,
            )
        else:
            if spec.gradient_scheme_curv not in {"forward", "central"}:
                raise ValueError("gradient_scheme_curv must be 'forward', 'central', or 'edge'.")
            D_curv, _ = _build_gradient_block(
                surfaces,
                rows_mode="interior",
                emdm_mode=spec.emdm_mode,
                scheme=spec.gradient_scheme_curv,
            )
        nrows = D_curv.shape[0] // 2
        if D_curv.shape[0] != 2 * nrows or D_curv.shape[1] != n_unknown:
            raise RuntimeError("Curvature gradient operator shape mismatch.")
        if nrows != n_unknown:
            raise RuntimeError(
                "Curvature regularizer requires rows_mode='interior' so that nrows == n_unknown."
            )

        g = D_curv @ s
        g_u = g[0::2]
        g_v = g[1::2]
        Dg_u = D_curv @ g_u
        Dg_v = D_curv @ g_v

        if spec.use_curv_r1:
            H4 = cp.vstack([Dg_u[0::2], Dg_u[1::2], Dg_v[0::2], Dg_v[1::2]]).T
            u_curv = cp.Variable(nrows, nonneg=True)
            constraints.append(cp.norm(H4, 2, axis=1) <= u_curv)
            obj_terms.append(spec.lambda_curv_r1 * cp.sum(u_curv))

        if spec.use_curv_en and spec.lambda_curv_en > 0.0:
            obj_terms.append(spec.lambda_curv_en * (cp.sum_squares(Dg_u) + cp.sum_squares(Dg_v)))

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
