from __future__ import annotations

from collections.abc import Iterable

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
        pts = pts @ R.T

    return radius * pts


def symmetrize_points(points: np.ndarray, axes: Iterable[str] = ("x", "y", "z")) -> np.ndarray:
    """Reflect points across specified axes."""
    axes_set = set(axes)
    P = np.asarray(points, dtype=float)
    if P.size == 0:
        return P.reshape(0, 3)

    flips = [(1.0, 1.0, 1.0)]
    for ax in axes_set:
        if ax == "x":
            flips = [f for f in flips] + [(-f[0], f[1], f[2]) for f in flips]
        elif ax == "y":
            flips = [f for f in flips] + [(f[0], -f[1], f[2]) for f in flips]
        elif ax == "z":
            flips = [f for f in flips] + [(f[0], f[1], -f[2]) for f in flips]
        else:
            raise ValueError("axes must be a subset of {'x','y','z'}.")

    flips_unique = np.unique(np.array(flips, dtype=float), axis=0)
    out = np.vstack([P * f for f in flips_unique])
    return out


def dedup_points_with_weights(
    points: np.ndarray, eps: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate points within eps tolerance and return weights."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if P.size == 0:
        return P.reshape(0, 3), np.zeros((0,), dtype=float)
    if eps < 0.0:
        raise ValueError("eps must be non-negative.")

    if eps == 0.0:
        uniq, idx, counts = np.unique(P, axis=0, return_index=True, return_counts=True)
        order = np.argsort(idx)
        return uniq[order], counts[order].astype(float)

    q = np.round(P / eps).astype(np.int64)
    q = np.ascontiguousarray(q)
    key_dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    keys = q.view(key_dtype).reshape(-1)
    _, idx, counts = np.unique(keys, return_index=True, return_counts=True)
    order = np.argsort(idx)
    uniq = P[idx[order]]
    weights = counts[order].astype(float)
    return uniq, weights
