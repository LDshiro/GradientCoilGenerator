from __future__ import annotations

import numpy as np

from gradientcoil.physics.roi_sampling import hammersley_sphere
from gradientcoil.post.error_metrics import compute_bz_error_dataset
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface
from gradientcoil.targets.target_bz_source import ShimBasisTargetBz


def test_compute_bz_error_dataset_shapes_and_histogram() -> None:
    surface = build_plane_cart_surface(
        PlaneCartSurfaceConfig(PLANE_HALF=0.1, NX=6, NY=6, R_AP=0.09, z0=0.0)
    )
    points = hammersley_sphere(10, 0.05, rotate=False)
    weights = np.linspace(1.0, 2.0, points.shape[0], dtype=float)
    target_source = ShimBasisTargetBz(
        max_order=1,
        coeffs={"Y": 0.02},
        L_ref=0.05,
        scale_policy="T_per_m",
    )
    bz_target = target_source.evaluate(points)
    s_opt = np.zeros((surface.Nint,), dtype=float)

    dataset = compute_bz_error_dataset(
        points,
        bz_target,
        s_opt,
        [surface],
        "shared",
        roi_weights=weights,
        hist_bins=20,
        zero_threshold_factor=1e-9,
    )

    assert dataset.bz_pred.shape == (points.shape[0],)
    assert dataset.bz_error_abs.shape == (points.shape[0],)
    assert dataset.bz_error_valid_mask.shape == (points.shape[0],)
    assert dataset.bz_error_hist_edges.shape == (21,)
    assert dataset.bz_error_hist_counts_unweighted.shape == (20,)
    assert dataset.bz_error_hist_counts_weighted.shape == (20,)
    assert np.all(dataset.bz_error_abs >= 0.0)

    valid_mask = np.abs(bz_target) > dataset.bz_error_tau
    assert np.array_equal(dataset.bz_error_valid_mask, valid_mask)
    assert int(np.sum(dataset.bz_error_hist_counts_unweighted)) == int(np.sum(valid_mask))
    assert np.isclose(
        np.sum(dataset.bz_error_hist_counts_weighted),
        np.sum(weights[valid_mask]),
        rtol=1e-12,
        atol=1e-12,
    )
