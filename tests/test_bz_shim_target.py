from __future__ import annotations

import numpy as np

from gradientcoil.targets.bz_shim import BzShimTargetSpec


def test_bz_shim_y_term() -> None:
    y = np.array([-0.1, 0.0, 0.1], dtype=float)
    points = np.column_stack([np.zeros_like(y), y, np.zeros_like(y)])
    G = 1.23
    spec = BzShimTargetSpec(coeffs={"Y": G}, terms=["Y"], L_ref=0.1)
    bz = spec.evaluate(points)
    assert np.allclose(bz, G * y, rtol=1e-12, atol=1e-12)


def test_bz_shim_scaling_z2() -> None:
    points = np.array([[0.0, 0.0, 0.1]], dtype=float)
    spec = BzShimTargetSpec(coeffs={"Z2": 1.0}, terms=["Z2"], L_ref=0.1)
    bz = spec.evaluate(points)
    assert np.allclose(bz, 0.1, rtol=1e-12, atol=1e-12)


def test_bz_shim_missing_terms() -> None:
    points = np.array([[0.1, -0.2, 0.3]], dtype=float)
    spec = BzShimTargetSpec(coeffs={"X": 1.0}, terms=["X", "Y"], L_ref=0.2)
    bz = spec.evaluate(points)
    assert np.isfinite(bz).all()
