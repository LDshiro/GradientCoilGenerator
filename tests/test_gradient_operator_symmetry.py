from __future__ import annotations

import numpy as np

from gradientcoil.operators.gradient import build_gradient_operator
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface


def _mirror_perm_x(surface, tol: float = 1e-12) -> np.ndarray:
    coords = surface.coords_int
    X = surface.X_plot
    Y = surface.Y_plot

    def key(x: float, y: float) -> tuple[int, int]:
        return (int(np.round(x / tol)), int(np.round(y / tol)))

    lookup = {key(X[iu, iv], Y[iu, iv]): k for k, (iu, iv) in enumerate(coords)}
    perm = np.empty((coords.shape[0],), dtype=int)
    for k, (iu, iv) in enumerate(coords):
        x = float(X[iu, iv])
        y = float(Y[iu, iv])
        perm[k] = lookup[key(-x, y)]
    return perm


def _tv_like(D, s: np.ndarray) -> float:
    g = D @ s
    g2 = g.reshape(-1, 2)
    return float(np.sum(np.linalg.norm(g2, axis=1)))


def test_gradient_operator_central_x_mirror_symmetry() -> None:
    surface = build_plane_cart_surface(
        PlaneCartSurfaceConfig(PLANE_HALF=0.2, NX=16, NY=16, R_AP=0.19)
    )
    op_forward = build_gradient_operator(surface, rows="interior", scheme="forward")
    op_central = build_gradient_operator(surface, rows="interior", scheme="central")

    rng = np.random.default_rng(0)
    s = rng.standard_normal(surface.Nint)
    perm = _mirror_perm_x(surface)
    s_m = s[perm]

    diff_c = abs(_tv_like(op_central.D, s) - _tv_like(op_central.D, s_m))
    base = max(1.0, _tv_like(op_central.D, s))
    assert diff_c <= 1e-10 * base

    diff_f = abs(_tv_like(op_forward.D, s) - _tv_like(op_forward.D, s_m))
    assert diff_f >= diff_c
