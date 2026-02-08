from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


@dataclass(frozen=True)
class ContourLevelConfig:
    mode: str = "n_levels"  # "n_levels" | "delta_s"
    n_levels_total: int = 6
    delta_s: float = 1.0


@dataclass(frozen=True)
class ContourFilterConfig:
    min_vertices: int = 12
    area_eps: float = 1e-6
    close_tol: float = 1e-10


@dataclass(frozen=True)
class TriangulationConfig:
    use_refine: bool = True
    flat_ratio: float = 0.02
    subdiv: int = 2


@dataclass(frozen=True)
class PlotConfig:
    show_filled: bool = True
    show_level_lines: bool = True
    show_extracted_loops: bool = True


@dataclass(frozen=True)
class ResampleConfig:
    ds_segment: float = 0.005


@dataclass(frozen=True)
class PostprocessConfig:
    levels: ContourLevelConfig = field(default_factory=ContourLevelConfig)
    filters: ContourFilterConfig = field(default_factory=ContourFilterConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    resample: ResampleConfig = field(default_factory=ResampleConfig)
    extraction_method: str = "auto"  # "auto" | "grid" | "tri"


@dataclass(frozen=True)
class RunSurface:
    index: int
    X: np.ndarray
    Y: np.ndarray
    S: np.ndarray


@dataclass(frozen=True)
class PostprocessResult:
    levels: np.ndarray
    loops_resampled: list[dict[str, Any]]
    segments: dict[str, np.ndarray]
    extraction_method: str
    surface_type: str


def load_run(
    npz_path: str | os.PathLike[str],
    json_path: str | os.PathLike[str] | None = None,
) -> tuple[np.lib.npyio.NpzFile, dict[str, Any]]:
    data = np.load(os.fspath(npz_path), allow_pickle=True)

    cfg: dict[str, Any] = {}
    if json_path is not None and os.path.exists(os.fspath(json_path)):
        with open(os.fspath(json_path), encoding="utf-8") as f:
            cfg = json.load(f)
        return data, cfg

    if "config_json" in data.files:
        try:
            cfg = json.loads(str(data["config_json"]))
        except Exception:
            cfg = {}
    return data, cfg


def find_surfaces(data: np.lib.npyio.NpzFile) -> list[RunSurface]:
    idxs: list[int] = []
    for key in data.files:
        m = re.match(r"^S_grid_(\d+)$", key)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(set(idxs))

    if not idxs and "S_grid" in data.files:
        idxs = [0]

    surfaces: list[RunSurface] = []
    for i in idxs:
        s_key = f"S_grid_{i}" if f"S_grid_{i}" in data.files else "S_grid"
        x_key = (
            f"X_plot_{i}"
            if f"X_plot_{i}" in data.files
            else ("X_plot" if "X_plot" in data.files else None)
        )
        y_key = (
            f"Y_plot_{i}"
            if f"Y_plot_{i}" in data.files
            else ("Y_plot" if "Y_plot" in data.files else None)
        )

        if x_key is None or y_key is None:
            raise KeyError(
                f"X/Y coordinates not found for surface {i}. "
                f"Need X_plot_{i}, Y_plot_{i} (or X_plot/Y_plot)."
            )

        surfaces.append(
            RunSurface(
                index=i,
                X=np.asarray(data[x_key], float),
                Y=np.asarray(data[y_key], float),
                S=np.asarray(data[s_key], float),
            )
        )
    return surfaces


def infer_surface_type(cfg: dict[str, Any]) -> str:
    if "surface_type" in cfg:
        return str(cfg.get("surface_type"))
    surface_cfg = cfg.get("surface", {})
    if isinstance(surface_cfg, dict) and "kind" in surface_cfg:
        return str(surface_cfg.get("kind"))
    return "unknown"


def polygon_signed_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    return points if polygon_signed_area(points) >= 0 else points[::-1].copy()


def split_path_to_closed_loops(
    path_obj: mpath.Path,
    *,
    min_vertices: int,
    area_eps: float,
    close_tol: float,
) -> list[np.ndarray]:
    vertices = path_obj.vertices
    codes = path_obj.codes

    loops: list[np.ndarray] = []

    if codes is None:
        if vertices.shape[0] < 3:
            return []
        is_closed = np.linalg.norm(vertices[0] - vertices[-1]) <= close_tol
        if not is_closed:
            return []
        loop = vertices.copy()
        if not np.allclose(loop[0], loop[-1]):
            loop = np.vstack([loop, loop[0]])
        loops.append(loop)
    else:
        current: list[np.ndarray] = []
        current_closed = False

        def finalize_current() -> None:
            nonlocal current, current_closed
            if len(current) == 0:
                return
            loop = np.array(current, float)
            is_closed = current_closed or (np.linalg.norm(loop[0] - loop[-1]) <= close_tol)
            if not is_closed:
                current = []
                current_closed = False
                return
            if np.linalg.norm(loop[0] - loop[-1]) > close_tol:
                loop = np.vstack([loop, loop[0]])
            loops.append(loop)
            current = []
            current_closed = False

        for v, code in zip(vertices, codes, strict=False):
            if code == mpath.Path.MOVETO:
                finalize_current()
                current = [v]
                current_closed = False
            elif code == mpath.Path.LINETO:
                current.append(v)
            elif code == mpath.Path.CLOSEPOLY:
                current_closed = True
                finalize_current()
            else:
                current.append(v)

        finalize_current()

    cleaned: list[np.ndarray] = []
    for loop in loops:
        if loop.shape[0] < max(4, min_vertices):
            continue
        if np.linalg.norm(loop[0] - loop[-1]) > close_tol:
            continue
        if abs(polygon_signed_area(loop)) < area_eps:
            continue
        cleaned.append(loop)

    return cleaned


def make_levels_from_n(smax: float, n_levels_total: int) -> np.ndarray:
    if smax <= 0.0 or not np.isfinite(smax) or n_levels_total <= 0:
        return np.array([], float)

    if n_levels_total % 2 == 1:
        n_levels_total -= 1

    n_half = n_levels_total // 2
    if n_half <= 0:
        return np.array([], float)

    step = smax / (n_half + 1)
    pos = step * np.arange(1, n_half + 1)
    levels = np.concatenate([-pos[::-1], pos])
    return levels


def make_levels_from_delta_s(
    values: np.ndarray, delta_s: float, max_turns: int | None = None
) -> np.ndarray:
    zvals = np.asarray(values, float).ravel()
    zvals = zvals[np.isfinite(zvals)]
    if zvals.size == 0:
        return np.array([], float)

    smin, smax = float(np.min(zvals)), float(np.max(zvals))
    if not np.isfinite(smin) or not np.isfinite(smax) or (smax - smin) < 1e-15:
        return np.array([], float)

    k0 = int(np.ceil((smin - 0.5 * delta_s) / delta_s))
    k1 = int(np.floor((smax - 0.5 * delta_s) / delta_s))
    ks = np.arange(k0, k1 + 1, dtype=int)

    if max_turns is not None and ks.size > max_turns:
        mid = ks.size // 2
        half = max_turns // 2
        ks = ks[mid - half : mid - half + max_turns]

    if ks.size == 0:
        return np.array([], float)

    levels = (0.5 * delta_s + ks * delta_s).astype(float)
    levels = levels[np.abs(levels) > 1e-14 * max(1.0, np.max(np.abs(levels)))]
    return levels


def _iter_contour_paths(
    contour_set: Any, levels: np.ndarray
) -> list[tuple[float, list[mpath.Path]]]:
    """Return contour paths across matplotlib API variants.

    Older matplotlib exposes ``collections``; newer versions may only expose
    ``allsegs``. This helper normalizes both into path lists.
    """
    contour_levels = np.asarray(getattr(contour_set, "levels", levels), float)

    if hasattr(contour_set, "collections"):
        out: list[tuple[float, list[mpath.Path]]] = []
        for level, coll in zip(contour_levels, contour_set.collections, strict=False):
            out.append((float(level), list(coll.get_paths())))
        return out

    allsegs = getattr(contour_set, "allsegs", None)
    if allsegs is None:
        raise AttributeError("ContourSet has neither 'collections' nor 'allsegs'.")

    out = []
    for level, segs in zip(contour_levels, allsegs, strict=False):
        paths: list[mpath.Path] = []
        for seg in segs:
            verts = np.asarray(seg, float)
            if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 2:
                continue
            paths.append(mpath.Path(verts))
        out.append((float(level), paths))
    return out


def grid_contour_loops(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    levels: np.ndarray,
    *,
    filters: ContourFilterConfig,
) -> list[tuple[float, list[np.ndarray]]]:
    fig = plt.figure()
    cs = plt.contour(X, Y, Z, levels=levels)
    path_groups = _iter_contour_paths(cs, levels)
    plt.close(fig)
    out: list[tuple[float, list[np.ndarray]]] = []
    for level, paths in path_groups:
        level_loops: list[np.ndarray] = []
        for path in paths:
            level_loops.extend(
                split_path_to_closed_loops(
                    path,
                    min_vertices=filters.min_vertices,
                    area_eps=filters.area_eps,
                    close_tol=filters.close_tol,
                )
            )
        out.append((float(level), level_loops))
    return out


def build_delaunay_triangulation(X: np.ndarray, Y: np.ndarray) -> mtri.Triangulation:
    return mtri.Triangulation(X, Y)


def mask_triangles_with_nan(tri: mtri.Triangulation, Z: np.ndarray) -> None:
    if np.isnan(Z).any():
        tris = tri.triangles
        mask = np.any(np.isnan(Z[tris]), axis=1)
        tri.set_mask(mask)


def refine_and_mask_tri(
    tri: mtri.Triangulation,
    Z: np.ndarray,
    *,
    flat_ratio: float,
    subdiv: int,
) -> tuple[mtri.Triangulation, np.ndarray]:
    analyzer = mtri.TriAnalyzer(tri)
    mask = analyzer.get_flat_tri_mask(min_circle_ratio=flat_ratio)
    if tri.mask is None:
        tri.set_mask(mask)
    else:
        tri.set_mask(tri.mask | mask)
    refiner = mtri.UniformTriRefiner(tri)
    tri_fine, z_fine = refiner.refine_field(Z, subdiv=subdiv)
    return tri_fine, z_fine


def tricontour_loops(
    tri: mtri.Triangulation,
    Zvals: np.ndarray,
    levels: np.ndarray,
    *,
    filters: ContourFilterConfig,
) -> list[tuple[float, list[np.ndarray]]]:
    fig = plt.figure()
    cs = plt.tricontour(tri, Zvals, levels=levels)
    path_groups = _iter_contour_paths(cs, levels)
    plt.close(fig)
    out: list[tuple[float, list[np.ndarray]]] = []
    for level, paths in path_groups:
        level_loops: list[np.ndarray] = []
        for path in paths:
            level_loops.extend(
                split_path_to_closed_loops(
                    path,
                    min_vertices=filters.min_vertices,
                    area_eps=filters.area_eps,
                    close_tol=filters.close_tol,
                )
            )
        out.append((float(level), level_loops))
    return out


def groups_from_raw(
    raw: list[tuple[float, list[np.ndarray]]], zero_eps: float = 1e-14
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for level, loops in raw:
        if abs(level) < zero_eps:
            continue
        sgn = 1 if level > 0 else -1
        loops_ccw = [ensure_ccw(loop) for loop in loops]
        if loops_ccw:
            groups.append({"level": float(level), "sign": int(sgn), "loops": loops_ccw})
    return groups


def resample_polyline_uniform(
    points: np.ndarray, ds: float, *, closed: bool = True, eps: float = 1e-12
) -> np.ndarray:
    pts = np.asarray(points, float)
    if closed and np.linalg.norm(pts[0] - pts[-1]) > eps:
        pts = np.vstack([pts, pts[0]])

    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    keep = d > eps
    if not np.all(keep):
        pts = np.vstack([pts[:-1][keep], pts[-1]])
        d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)

    length = float(np.sum(d))
    if length <= ds:
        return pts[[0, -1], :]

    sacc = np.concatenate([[0.0], np.cumsum(d)])
    s_vals = np.arange(0.0, length, ds)
    idx = np.searchsorted(sacc[1:], s_vals, side="right")

    seg_start = pts[idx, :]
    seg_end = pts[idx + 1, :]
    seg_len = sacc[idx + 1] - sacc[idx]
    alpha = (s_vals - sacc[idx]) / np.maximum(seg_len, eps)
    resampled = (1.0 - alpha)[:, None] * seg_start + alpha[:, None] * seg_end
    if closed and resampled.shape[0] > 0:
        if np.linalg.norm(resampled[0] - resampled[-1]) > eps:
            resampled = np.vstack([resampled, resampled[0]])
    return resampled


def resample_groups_to_segments(
    groups: list[dict[str, Any]], ds: float, *, surface_index: int
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    loops_out: list[dict[str, Any]] = []
    P_list: list[np.ndarray] = []
    Q_list: list[np.ndarray] = []
    loop_id_list: list[np.ndarray] = []
    level_list: list[np.ndarray] = []
    sign_list: list[np.ndarray] = []
    surf_list: list[np.ndarray] = []

    loop_id = 0
    for group in groups:
        level = float(group["level"])
        sgn = int(group["sign"])
        for loop in group["loops"]:
            pts = resample_polyline_uniform(loop, ds, closed=True)
            loops_out.append({"surface": surface_index, "level": level, "sign": sgn, "points": pts})

            pts_seg = pts
            if pts_seg.shape[0] >= 2 and np.linalg.norm(pts_seg[0] - pts_seg[-1]) <= 1e-12:
                pts_seg = pts_seg[:-1]

            if pts_seg.shape[0] >= 2:
                P = pts_seg
                Q = np.roll(pts_seg, -1, axis=0)
                nseg = P.shape[0]

                P_list.append(P)
                Q_list.append(Q)
                loop_id_list.append(np.full(nseg, loop_id, dtype=int))
                level_list.append(np.full(nseg, level, dtype=float))
                sign_list.append(np.full(nseg, sgn, dtype=int))
                surf_list.append(np.full(nseg, surface_index, dtype=int))

            loop_id += 1

    if P_list:
        P_all = np.vstack(P_list)
        Q_all = np.vstack(Q_list)
        loop_all = np.concatenate(loop_id_list)
        level_all = np.concatenate(level_list)
        sign_all = np.concatenate(sign_list)
        surf_all = np.concatenate(surf_list)

        L_all = np.linalg.norm(Q_all - P_all, axis=1)
        M_all = 0.5 * (P_all + Q_all)
    else:
        P_all = np.zeros((0, 2), float)
        Q_all = np.zeros((0, 2), float)
        M_all = np.zeros((0, 2), float)
        L_all = np.zeros((0,), float)
        loop_all = np.zeros((0,), int)
        level_all = np.zeros((0,), float)
        sign_all = np.zeros((0,), int)
        surf_all = np.zeros((0,), int)

    segments = {
        "P": P_all,
        "Q": Q_all,
        "M": M_all,
        "L": L_all,
        "loop": loop_all,
        "level": level_all,
        "sign": sign_all,
        "surface": surf_all,
    }
    return loops_out, segments


def plot_filled_and_levels_grid(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, levels: np.ndarray, title: str, plots: PlotConfig
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    if plots.show_filled:
        cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
        plt.colorbar(cf, ax=ax, shrink=0.9, label="S")
    if plots.show_level_lines:
        ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.8, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tripcolor(
    tri: mtri.Triangulation, Z: np.ndarray, levels: np.ndarray, title: str, plots: PlotConfig
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    if plots.show_filled:
        tpc = ax.tripcolor(tri, Z, shading="gouraud", cmap="viridis")
        plt.colorbar(tpc, ax=ax, shrink=0.9, label="S")
    if plots.show_level_lines:
        ax.tricontour(tri, Z, levels=levels, colors="k", linewidths=0.8, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_extracted_loops(groups: list[dict[str, Any]], title: str, plots: PlotConfig) -> None:
    if not plots.show_extracted_loops:
        return
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    for group in groups:
        col = "r" if group["sign"] > 0 else "b"
        for loop in group["loops"]:
            ax.plot(loop[:, 0], loop[:, 1], color=col, lw=1.4, alpha=0.95)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def decide_extraction_method(method: str, surface_type: str) -> str:
    if method == "auto":
        if "polar" in surface_type or "disk" in surface_type:
            return "tri"
        return "grid"
    return method


def compute_levels(surfaces: Iterable[RunSurface], level_cfg: ContourLevelConfig) -> np.ndarray:
    if level_cfg.mode == "n_levels":
        smax_global = max(float(np.nanmax(np.abs(s.S))) for s in surfaces)
        return make_levels_from_n(smax_global, level_cfg.n_levels_total)
    if level_cfg.mode == "delta_s":
        first_surface = next(iter(surfaces))
        return make_levels_from_delta_s(first_surface.S, level_cfg.delta_s, max_turns=None)
    raise ValueError("level mode must be 'n_levels' or 'delta_s'")


def extract_groups_for_surface(
    surface: RunSurface,
    levels: np.ndarray,
    *,
    method: str,
    filters: ContourFilterConfig,
    tri_cfg: TriangulationConfig,
    plots: PlotConfig,
) -> list[dict[str, Any]]:
    if method == "grid":
        if plots.show_filled or plots.show_level_lines:
            plot_filled_and_levels_grid(
                surface.X, surface.Y, surface.S, levels, f"S surface {surface.index}", plots
            )
        raw = grid_contour_loops(surface.X, surface.Y, surface.S, levels, filters=filters)
        return groups_from_raw(raw)

    if method == "tri":
        Xf = surface.X.reshape(-1)
        Yf = surface.Y.reshape(-1)
        Zf = surface.S.reshape(-1)

        tri = build_delaunay_triangulation(Xf, Yf)
        mask_triangles_with_nan(tri, Zf)

        if tri_cfg.use_refine:
            tri_use, z_use = refine_and_mask_tri(
                tri, Zf, flat_ratio=tri_cfg.flat_ratio, subdiv=tri_cfg.subdiv
            )
        else:
            tri_use, z_use = tri, Zf

        if plots.show_filled or plots.show_level_lines:
            plot_tripcolor(tri_use, z_use, levels, f"S surface {surface.index}", plots)

        raw = tricontour_loops(tri_use, z_use, levels, filters=filters)
        return groups_from_raw(raw)

    raise ValueError("Unknown extraction method")


def merge_segments(segment_parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if segment_parts:
        P = np.vstack([seg["P"] for seg in segment_parts])
        Q = np.vstack([seg["Q"] for seg in segment_parts])
        M = np.vstack([seg["M"] for seg in segment_parts])
        L = np.concatenate([seg["L"] for seg in segment_parts])
        loop = np.concatenate([seg["loop"] for seg in segment_parts])
        level = np.concatenate([seg["level"] for seg in segment_parts])
        sign = np.concatenate([seg["sign"] for seg in segment_parts])
        surface = np.concatenate([seg["surface"] for seg in segment_parts])
    else:
        P = Q = M = np.zeros((0, 2), float)
        L = loop = sign = surface = np.zeros((0,), int)
        level = np.zeros((0,), float)
    return dict(P=P, Q=Q, M=M, L=L, loop=loop, level=level, sign=sign, surface=surface)


def process_run(
    run_npz_path: str | os.PathLike[str],
    config_json_path: str | os.PathLike[str] | None,
    cfg: PostprocessConfig,
) -> tuple[PostprocessResult, dict[str, Any]]:
    data, run_cfg = load_run(run_npz_path, config_json_path)
    surfaces = find_surfaces(data)
    surface_type = infer_surface_type(run_cfg)
    method = decide_extraction_method(cfg.extraction_method, surface_type)

    levels = compute_levels(surfaces, cfg.levels)
    if levels.size == 0:
        raise RuntimeError("No contour levels were generated. Check S range and level settings.")

    all_loops: list[dict[str, Any]] = []
    segment_parts: list[dict[str, np.ndarray]] = []

    for surface in surfaces:
        groups = extract_groups_for_surface(
            surface,
            levels,
            method=method,
            filters=cfg.filters,
            tri_cfg=cfg.triangulation,
            plots=cfg.plots,
        )

        if cfg.plots.show_extracted_loops:
            plot_extracted_loops(
                groups, title=f"Extracted closed loops (surface {surface.index})", plots=cfg.plots
            )

        loops_out, segments = resample_groups_to_segments(
            groups, cfg.resample.ds_segment, surface_index=surface.index
        )
        all_loops.extend(loops_out)
        segment_parts.append(segments)

    segments_all = merge_segments(segment_parts)
    result = PostprocessResult(
        levels=levels,
        loops_resampled=all_loops,
        segments=segments_all,
        extraction_method=method,
        surface_type=surface_type,
    )
    return result, run_cfg


def save_resampled_npz(
    out_path: str | os.PathLike[str],
    *,
    result: PostprocessResult,
    run_cfg: dict[str, Any],
    run_npz_path: str | os.PathLike[str],
    cfg: PostprocessConfig,
) -> None:
    np.savez(
        os.fspath(out_path),
        source_npz=os.fspath(run_npz_path),
        source_config_json=json.dumps(run_cfg),
        level_mode=cfg.levels.mode,
        n_levels_total=cfg.levels.n_levels_total,
        delta_s=cfg.levels.delta_s,
        levels_used=result.levels.astype(float),
        ds=cfg.resample.ds_segment,
        loops_resampled=np.array(result.loops_resampled, dtype=object),
        segments=result.segments,
        filters=dict(
            MIN_VERTICES=cfg.filters.min_vertices,
            AREA_EPS=cfg.filters.area_eps,
            CLOSE_TOL=cfg.filters.close_tol,
        ),
        extraction=dict(
            method=result.extraction_method,
            USE_TRI_REFINE=cfg.triangulation.use_refine,
            FLAT_RATIO=cfg.triangulation.flat_ratio,
            SUBDIV=cfg.triangulation.subdiv,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process contours and resample to uniform segment length."
    )
    parser.add_argument("--run-npz", required=True, help="Path to run.npz")
    parser.add_argument("--config-json", default=None, help="Optional config.json path")
    parser.add_argument(
        "--out-npz", default="coil_contours_resampled_post1.npz", help="Output npz path"
    )
    parser.add_argument(
        "--ds-segment", type=float, default=0.005, help="Segment length for resampling [m]"
    )
    parser.add_argument("--level-mode", choices=["n_levels", "delta_s"], default="n_levels")
    parser.add_argument("--n-levels-total", type=int, default=6)
    parser.add_argument("--delta-s", type=float, default=1.0)
    parser.add_argument("--min-vertices", type=int, default=12)
    parser.add_argument("--area-eps", type=float, default=1e-6)
    parser.add_argument("--close-tol", type=float, default=1e-10)
    parser.add_argument("--extraction-method", choices=["auto", "grid", "tri"], default="auto")
    parser.add_argument("--tri-refine", dest="tri_refine", action="store_true", default=True)
    parser.add_argument("--no-tri-refine", dest="tri_refine", action="store_false")
    parser.add_argument("--flat-ratio", type=float, default=0.02)
    parser.add_argument("--subdiv", type=int, default=2)
    parser.add_argument("--show-filled", dest="show_filled", action="store_true", default=True)
    parser.add_argument("--no-show-filled", dest="show_filled", action="store_false")
    parser.add_argument(
        "--show-level-lines", dest="show_level_lines", action="store_true", default=True
    )
    parser.add_argument("--no-show-level-lines", dest="show_level_lines", action="store_false")
    parser.add_argument(
        "--show-extracted-loops", dest="show_extracted_loops", action="store_true", default=True
    )
    parser.add_argument(
        "--no-show-extracted-loops", dest="show_extracted_loops", action="store_false"
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PostprocessConfig(
        levels=ContourLevelConfig(
            mode=args.level_mode,
            n_levels_total=args.n_levels_total,
            delta_s=args.delta_s,
        ),
        filters=ContourFilterConfig(
            min_vertices=args.min_vertices,
            area_eps=args.area_eps,
            close_tol=args.close_tol,
        ),
        triangulation=TriangulationConfig(
            use_refine=args.tri_refine,
            flat_ratio=args.flat_ratio,
            subdiv=args.subdiv,
        ),
        plots=PlotConfig(
            show_filled=args.show_filled,
            show_level_lines=args.show_level_lines,
            show_extracted_loops=args.show_extracted_loops,
        ),
        resample=ResampleConfig(ds_segment=args.ds_segment),
        extraction_method=args.extraction_method,
    )

    result, run_cfg = process_run(args.run_npz, args.config_json, cfg)
    save_resampled_npz(
        args.out_npz, result=result, run_cfg=run_cfg, run_npz_path=args.run_npz, cfg=cfg
    )
    print(f"[Saved] {args.out_npz}")
    print(f"[Summary] loops={len(result.loops_resampled)} segments={result.segments['P'].shape[0]}")
    return 0


def main() -> None:
    raise SystemExit(cli_main())


__all__ = [
    "ContourFilterConfig",
    "ContourLevelConfig",
    "PlotConfig",
    "PostprocessConfig",
    "PostprocessResult",
    "ResampleConfig",
    "RunSurface",
    "TriangulationConfig",
    "cli_main",
    "load_run",
    "process_run",
    "save_resampled_npz",
]
