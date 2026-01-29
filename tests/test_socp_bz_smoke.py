from __future__ import annotations

import numpy as np
import pytest

from gradientcoil.optimize.socp_bz import SocpBzSpec, solve_socp_bz
from gradientcoil.physics.roi_sampling import hammersley_sphere
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface
from gradientcoil.targets.bz_shim import BzShimTargetSpec, standard_shim_terms


def test_socp_bz_smoke() -> None:
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("CLARABEL not installed")

    surface = build_plane_cart_surface(
        PlaneCartSurfaceConfig(PLANE_HALF=0.1, NX=6, NY=6, R_AP=0.09)
    )
    points = hammersley_sphere(6, 0.05, rotate=False)
    terms = list(standard_shim_terms(max_order=1).keys())
    target_spec = BzShimTargetSpec(coeffs={"Y": 0.02}, terms=terms, L_ref=0.05)
    bz_target = target_spec.evaluate(points)

    spec = SocpBzSpec(use_tv=False, use_pitch=False, use_power=False, max_iter=200)
    result = solve_socp_bz(points, bz_target, [surface], spec)

    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.s_opt.shape[0] == surface.Nint
    assert np.isfinite(result.s_opt).all()
    assert len(result.S_grids) == 1
    assert result.S_grids[0].shape == surface.grid_shape


def test_socp_bz_tgv_smoke_disk_polar() -> None:
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("CLARABEL not installed")

    surface = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.1, NR=4, NT=8))
    points = hammersley_sphere(6, 0.05, rotate=False)
    terms = list(standard_shim_terms(max_order=1).keys())
    target_spec = BzShimTargetSpec(coeffs={"Y": 0.02}, terms=terms, L_ref=0.05)
    bz_target = target_spec.evaluate(points)

    spec = SocpBzSpec(
        use_tgv=True,
        alpha1_tgv=1e-6,
        alpha0_tgv=1e-6,
        use_tv=False,
        use_pitch=False,
        use_power=False,
        max_iter=200,
    )
    result = solve_socp_bz(points, bz_target, [surface], spec)

    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.s_opt.shape[0] == surface.Nint
    assert np.isfinite(result.s_opt).all()


def test_socp_bz_tgv_edge_smoke_disk_polar() -> None:
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("CLARABEL not installed")

    surface = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.1, NR=4, NT=8))
    points = hammersley_sphere(6, 0.05, rotate=False)
    terms = list(standard_shim_terms(max_order=1).keys())
    target_spec = BzShimTargetSpec(coeffs={"Y": 0.02}, terms=terms, L_ref=0.05)
    bz_target = target_spec.evaluate(points)

    spec = SocpBzSpec(
        use_tgv=True,
        alpha1_tgv=1e-6,
        alpha0_tgv=1e-6,
        gradient_scheme_tgv="edge",
        use_tv=False,
        use_pitch=False,
        use_power=False,
        max_iter=200,
    )
    result = solve_socp_bz(points, bz_target, [surface], spec)

    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.s_opt.shape[0] == surface.Nint
    assert np.isfinite(result.s_opt).all()


def test_socp_bz_curv_r1_smoke_disk_polar() -> None:
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("CLARABEL not installed")

    surface = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.1, NR=4, NT=8))
    points = hammersley_sphere(6, 0.05, rotate=False)
    terms = list(standard_shim_terms(max_order=1).keys())
    target_spec = BzShimTargetSpec(coeffs={"Y": 0.02}, terms=terms, L_ref=0.05)
    bz_target = target_spec.evaluate(points)

    spec = SocpBzSpec(
        use_curv_r1=True,
        lambda_curv_r1=1e-4,
        use_tv=False,
        use_pitch=False,
        use_power=False,
        max_iter=200,
    )
    result = solve_socp_bz(points, bz_target, [surface], spec)

    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.s_opt.shape[0] == surface.Nint
    assert np.isfinite(result.s_opt).all()
