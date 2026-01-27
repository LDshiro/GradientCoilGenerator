from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

import numpy as np

from gradientcoil.surfaces.base import SurfaceGrid

MU0_DEFAULT = 4 * np.pi * 1e-7 * 1e3


def _hash_array(hasher: sha256, arr: np.ndarray) -> None:
    arr_c = np.ascontiguousarray(arr)
    hasher.update(str(arr_c.shape).encode("utf-8"))
    hasher.update(arr_c.dtype.str.encode("utf-8"))
    hasher.update(arr_c.tobytes())


def _cache_key(
    points: np.ndarray,
    centers_list: Iterable[np.ndarray],
    normals_list: Iterable[np.ndarray],
    areas_list: Iterable[np.ndarray],
    *,
    mode: str,
    weights: np.ndarray,
    mu0: float,
) -> str:
    h = sha256()
    _hash_array(h, points)
    for centers, normals, areas in zip(centers_list, normals_list, areas_list, strict=True):
        _hash_array(h, centers)
        _hash_array(h, normals)
        _hash_array(h, areas)
    h.update(mode.encode("utf-8"))
    _hash_array(h, weights)
    h.update(np.array([mu0], dtype=float).tobytes())
    return h.hexdigest()


def emdm_components(
    points: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    *,
    chunk: int = 4096,
    mu0: float | None = None,
    mu0_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute EMDM field components for dipole surfaces."""
    if mu0 is None:
        mu0 = MU0_DEFAULT * float(mu0_scale)
    else:
        mu0 = float(mu0)

    P = points.shape[0]
    M = centers.shape[0]
    Ax = np.zeros((P, M), dtype=float)
    Ay = np.zeros((P, M), dtype=float)
    Az = np.zeros((P, M), dtype=float)

    c = mu0 / (4.0 * np.pi)
    for j0 in range(0, M, chunk):
        j1 = min(M, j0 + chunk)
        C = centers[j0:j1]
        N = normals[j0:j1]
        A = areas[j0:j1][:, None]

        R = points[:, None, :] - C[None, :, :]
        R2 = np.sum(R * R, axis=2)
        Rn = np.sqrt(R2)
        Rh = R / np.maximum(Rn[..., None], 1e-30)
        m = A * N
        md = np.sum(m[None, :, :] * Rh, axis=2)
        B = c * ((3.0 * md)[..., None] * Rh - m[None, :, :]) / np.maximum(Rn[..., None] ** 3, 1e-30)
        Ax[:, j0:j1] = B[..., 0]
        Ay[:, j0:j1] = B[..., 1]
        Az[:, j0:j1] = B[..., 2]

    return Ax, Ay, Az


def build_A_xyz(
    points_world: np.ndarray,
    surfaces: Iterable[SurfaceGrid],
    *,
    mode: str = "concat",
    weights: Iterable[float] | None = None,
    cache_dir: str | Path | None = None,
    mu0: float | None = None,
    mu0_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build forward matrices A_x, A_y, A_z for one or more surfaces."""
    surfaces_list = list(surfaces)
    if len(surfaces_list) == 0:
        raise ValueError("surfaces must be a non-empty list.")
    if mode not in {"concat", "shared"}:
        raise ValueError("mode must be 'concat' or 'shared'.")

    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (P, 3).")

    if weights is None:
        weights_arr = np.ones((len(surfaces_list),), dtype=float)
    else:
        weights_arr = np.asarray(list(weights), dtype=float)
        if weights_arr.shape != (len(surfaces_list),):
            raise ValueError("weights length must match surfaces length.")

    centers_list = []
    normals_list = []
    areas_list = []
    nints = []
    for surface in surfaces_list:
        centers = surface.centers_world_uv[surface.interior_mask]
        normals = surface.normals_world_uv[surface.interior_mask]
        areas = surface.areas_uv[surface.interior_mask]
        centers_list.append(centers)
        normals_list.append(normals)
        areas_list.append(areas)
        nints.append(centers.shape[0])

    if mode == "shared" and len(set(nints)) != 1:
        raise ValueError("All surfaces must have the same Nint for shared mode.")

    mu0_val = MU0_DEFAULT * float(mu0_scale) if mu0 is None else float(mu0)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(
            points,
            centers_list,
            normals_list,
            areas_list,
            mode=mode,
            weights=weights_arr,
            mu0=mu0_val,
        )
        cache_path = cache_dir / f"emdm_{key}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            return data["Ax"], data["Ay"], data["Az"]

    if mode == "concat":
        Ax_list = []
        Ay_list = []
        Az_list = []
        for w, centers, normals, areas in zip(
            weights_arr, centers_list, normals_list, areas_list, strict=True
        ):
            Ax, Ay, Az = emdm_components(
                points,
                centers,
                normals,
                areas,
                mu0=mu0_val,
            )
            if w != 1.0:
                Ax = Ax * w
                Ay = Ay * w
                Az = Az * w
            Ax_list.append(Ax)
            Ay_list.append(Ay)
            Az_list.append(Az)
        Ax_out = np.hstack(Ax_list)
        Ay_out = np.hstack(Ay_list)
        Az_out = np.hstack(Az_list)
    else:
        Ax_out = None
        Ay_out = None
        Az_out = None
        for w, centers, normals, areas in zip(
            weights_arr, centers_list, normals_list, areas_list, strict=True
        ):
            Ax, Ay, Az = emdm_components(
                points,
                centers,
                normals,
                areas,
                mu0=mu0_val,
            )
            if Ax_out is None:
                Ax_out = w * Ax
                Ay_out = w * Ay
                Az_out = w * Az
            else:
                Ax_out += w * Ax
                Ay_out += w * Ay
                Az_out += w * Az
        if Ax_out is None or Ay_out is None or Az_out is None:
            raise RuntimeError("Failed to build shared matrices.")

    if cache_path is not None:
        np.savez(cache_path, Ax=Ax_out, Ay=Ay_out, Az=Az_out)

    return Ax_out, Ay_out, Az_out
