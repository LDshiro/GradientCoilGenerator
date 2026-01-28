from __future__ import annotations

import numpy as np

from gradientcoil.physics.roi_sampling import sample_sphere_sym_hammersley, symmetrize_points


def _key(p: np.ndarray, tol: float) -> tuple[int, int, int]:
    return (
        int(np.round(p[0] / tol)),
        int(np.round(p[1] / tol)),
        int(np.round(p[2] / tol)),
    )


def _assert_mirror(points: np.ndarray, axis: str, tol: float) -> None:
    keys = {_key(p, tol) for p in points}
    for p in points:
        if axis == "x":
            mirror = np.array([-p[0], p[1], p[2]])
        elif axis == "y":
            mirror = np.array([p[0], -p[1], p[2]])
        else:
            mirror = np.array([p[0], p[1], -p[2]])
        assert _key(mirror, tol) in keys


def test_sym_hammersley_xy_symmetry() -> None:
    base = sample_sphere_sym_hammersley(20, 0.05, sym_axes=("x", "y"))
    points = symmetrize_points(base, axes=("x", "y"))
    _assert_mirror(points, "x", tol=1e-12)
    _assert_mirror(points, "y", tol=1e-12)


def test_sym_hammersley_xyz_symmetry() -> None:
    base = sample_sphere_sym_hammersley(20, 0.05, sym_axes=("x", "y", "z"))
    points = symmetrize_points(base, axes=("x", "y", "z"))
    _assert_mirror(points, "x", tol=1e-12)
    _assert_mirror(points, "y", tol=1e-12)
    _assert_mirror(points, "z", tol=1e-12)
