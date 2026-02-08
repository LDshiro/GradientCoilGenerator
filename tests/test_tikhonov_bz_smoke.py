from __future__ import annotations

import numpy as np

from gradientcoil.optimize.tikhonov_bz import TikhonovBzSpec, solve_tikhonov_bz
from gradientcoil.physics.roi_sampling import hammersley_sphere
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface
from gradientcoil.targets.bz_shim import BzShimTargetSpec, standard_shim_terms


def test_tikhonov_bz_smoke() -> None:
    surface = build_plane_cart_surface(
        PlaneCartSurfaceConfig(PLANE_HALF=0.1, NX=6, NY=6, R_AP=0.09)
    )
    points = hammersley_sphere(6, 0.05, rotate=False)
    terms = list(standard_shim_terms(max_order=1).keys())
    target_spec = BzShimTargetSpec(coeffs={"Y": 0.02}, terms=terms, L_ref=0.05)
    bz_target = target_spec.evaluate(points)

    spec = TikhonovBzSpec(lambda_reg=1e-2, reg_operator="grad", cg_maxiter=200)
    result = solve_tikhonov_bz(points, bz_target, [surface], spec)

    assert result.status in {"converged", "not_converged", "lstsq"}
    assert result.s_opt.shape == (surface.Nint,)
    assert np.all(np.isfinite(result.s_opt))
    assert len(result.S_grids) == 1
    assert result.S_grids[0].shape == surface.grid_shape
