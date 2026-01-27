from __future__ import annotations

import numpy as np

from gradientcoil.physics.roi_sampling import hammersley_sphere, symmetrize_points


def test_hammersley_sphere_radius() -> None:
    radius = 0.25
    pts = hammersley_sphere(32, radius, rotate=False)
    assert pts.shape == (32, 3)
    r = np.linalg.norm(pts, axis=1)
    assert np.allclose(r, radius, rtol=1e-6, atol=1e-6)


def test_symmetrize_points_count() -> None:
    p0 = np.array([[0.1, 0.2, 0.3]], dtype=float)
    pts = symmetrize_points(p0, axes=("x", "y", "z"))
    assert pts.shape == (8, 3)
