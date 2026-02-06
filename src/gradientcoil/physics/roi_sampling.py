from __future__ import annotations

from itertools import product

import numpy as np


def _vdc_base2(n: int) -> np.ndarray:
    out = np.empty(n, dtype=float)
    for k in range(n):
        x = 0.0
        denom = 1.0
        m = k
        while m:
            denom *= 2.0
            x += (m & 1) / denom
            m >>= 1
        out[k] = x
    return out


def _apply_random_rotation(pts: np.ndarray, *, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    xq, yq, zq, wq = q1, q2, q3, q4
    R = np.array(
        [
            [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * wq), 2 * (xq * zq + yq * wq)],
            [2 * (xq * yq + zq * wq), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * wq)],
            [2 * (xq * zq - yq * wq), 2 * (yq * zq + xq * wq), 1 - 2 * (xq * xq + yq * yq)],
        ],
        dtype=float,
    )
    return pts @ R.T


def symmetrize_points(
    points: np.ndarray, *, axes: tuple[str, ...] | list[str] = ("x", "y", "z")
) -> np.ndarray:
    """Mirror points across the selected axes (x/y/z)."""
    pts = np.asarray(points, float)
    if pts.size == 0:
        return pts.copy()

    axis_map = {"x": 0, "y": 1, "z": 2}
    axes_norm = []
    for ax in axes:
        key = str(ax).lower()
        if key in axis_map:
            axes_norm.append(key)
    if not axes_norm:
        return pts.copy()

    idx = [axis_map[ax] for ax in axes_norm]
    out = []
    for signs in product([1.0, -1.0], repeat=len(idx)):
        mirrored = pts.copy()
        for axis_i, s in zip(idx, signs, strict=False):
            mirrored[:, axis_i] *= s
        out.append(mirrored)
    return np.vstack(out)


def hammersley_sphere(
    n: int, radius: float, *, rotate: bool = False, seed: int | None = None
) -> np.ndarray:
    """Generate Hammersley points on a sphere."""
    N = int(n)
    if N <= 0:
        return np.zeros((0, 3), dtype=float)

    k = np.arange(N, dtype=float)
    u = (k + 0.5) / N
    z = 1.0 - 2.0 * u
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = 2.0 * np.pi * _vdc_base2(N)
    x = r * np.cos(th)
    y = r * np.sin(th)
    pts = np.column_stack([x, y, z])

    if rotate:
        pts = _apply_random_rotation(pts, seed=seed)

    return radius * pts


def sample_sphere_fibonacci(
    n: int, radius: float, *, rotate: bool = False, seed: int | None = None
) -> np.ndarray:
    """Generate Fibonacci points on a sphere."""
    N = int(n)
    if N <= 0:
        return np.zeros((0, 3), dtype=float)

    k = np.arange(N, dtype=float)
    z = 1.0 - 2.0 * (k + 0.5) / N
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    th = k * golden_angle
    x = r * np.cos(th)
    y = r * np.sin(th)
    pts = np.column_stack([x, y, z])

    if rotate:
        pts = _apply_random_rotation(pts, seed=seed)

    return radius * pts
