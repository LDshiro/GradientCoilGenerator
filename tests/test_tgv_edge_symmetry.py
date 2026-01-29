from __future__ import annotations

import numpy as np

from gradientcoil.operators.gradient import build_edge_difference_operator
from gradientcoil.optimize.socp_bz import _build_line_graph_operator
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface


def _mirror_perm_x(surface) -> np.ndarray:
    coords = surface.coords_int
    xs = surface.X_plot[coords[:, 0], coords[:, 1]]
    ys = surface.Y_plot[coords[:, 0], coords[:, 1]]
    tol = 1e-12

    def key(x: float, y: float) -> tuple[int, int]:
        return (int(round(x / tol)), int(round(y / tol)))

    idx_map = {key(x, y): k for k, (x, y) in enumerate(zip(xs, ys, strict=True))}
    perm = np.empty((coords.shape[0],), dtype=int)
    for k, (x, y) in enumerate(zip(xs, ys, strict=True)):
        k_m = idx_map[key(-x, y)]
        perm[k] = k_m
    return perm


def test_edge_tgv_line_graph_symmetry_plane_cart() -> None:
    surface = build_plane_cart_surface(PlaneCartSurfaceConfig(PLANE_HALF=0.2, NX=8, NY=8))
    op = build_edge_difference_operator(surface, rows="interior")
    D_line, _ = _build_line_graph_operator(op)
    assert D_line.shape[0] > 0

    rng = np.random.default_rng(0)
    s = rng.standard_normal(surface.Nint)
    perm = _mirror_perm_x(surface)
    s_m = s[perm]

    g_line = D_line @ (op.D @ s)
    g_line_m = D_line @ (op.D @ s_m)
    e = float(np.sum(np.abs(g_line)))
    e_m = float(np.sum(np.abs(g_line_m)))

    diff = abs(e - e_m)
    assert diff <= 1e-10 * max(1.0, e)
