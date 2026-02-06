from __future__ import annotations

import numpy as np

from gradientcoil.physics.roi_sampling import (
    hammersley_sphere,
    sample_sphere_fibonacci,
    symmetrize_points,
)


def test_hammersley_sphere_radius() -> None:
    radius = 0.25
    pts = hammersley_sphere(32, radius, rotate=False)
    assert pts.shape == (32, 3)
    r = np.linalg.norm(pts, axis=1)
    assert np.allclose(r, radius, rtol=1e-6, atol=1e-6)


def test_sample_sphere_fibonacci_radius() -> None:
    radius = 0.12
    pts = sample_sphere_fibonacci(25, radius, rotate=False)
    assert pts.shape == (25, 3)
    r = np.linalg.norm(pts, axis=1)
    assert np.allclose(r, radius, rtol=1e-6, atol=1e-6)


def test_symmetrize_points_axes() -> None:
    base = np.array([[1.0, 2.0, 3.0]])
    sym = symmetrize_points(base, axes=("x", "z"))
    assert sym.shape == (4, 3)
    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 2.0, 3.0],
            [1.0, 2.0, -3.0],
            [-1.0, 2.0, -3.0],
        ]
    )
    for row in expected:
        assert np.any(np.all(np.isclose(sym, row), axis=1))
