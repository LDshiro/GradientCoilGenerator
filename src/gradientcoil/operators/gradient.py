from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from gradientcoil.surfaces.base import SurfaceGrid


@dataclass
class GradientOperator:
    """Sparse gradient operator for SurfaceGrid."""

    D: csr_matrix
    row_coords: np.ndarray
    rows_mode: str
    scheme: str


def build_gradient_operator(
    surface: SurfaceGrid,
    *,
    rows: str = "interior",
    boundary_value: float = 0.0,
    scheme: str = "forward",
) -> GradientOperator:
    """Build a sparse gradient operator for a surface.

    Central differences fall back to one-sided differences when a neighbor is missing.
    """
    if boundary_value != 0.0:
        raise ValueError("Only boundary_value=0.0 is supported.")

    if rows not in {"interior", "active"}:
        raise ValueError("rows must be 'interior' or 'active'.")
    if scheme not in {"forward", "central"}:
        raise ValueError("scheme must be 'forward' or 'central'.")

    if rows == "interior":
        row_mask = surface.interior_mask
    else:
        row_mask = surface.interior_mask | surface.boundary_mask

    row_coords = np.argwhere(row_mask)
    nrows = row_coords.shape[0]

    Nu, Nv = surface.grid_shape
    rows_idx: list[int] = []
    cols_idx: list[int] = []
    data: list[float] = []

    def add_entry(r: int, c: int, val: float) -> None:
        rows_idx.append(r)
        cols_idx.append(c)
        data.append(val)

    for ridx, (iu, iv) in enumerate(row_coords):
        scale_u = surface.scale_u[iu, iv]
        scale_v = surface.scale_v[iu, iv]
        if scale_u <= 0.0 or scale_v <= 0.0:
            raise ValueError("scale_u and scale_v must be positive.")

        k_cur = surface.idx_map[iu, iv]
        inv_u = 1.0 / scale_u
        inv_v = 1.0 / scale_v

        row_u = 2 * ridx
        row_v = row_u + 1

        if scheme == "forward":
            if surface.periodic_u:
                iu_n = (iu + 1) % Nu
                if surface.idx_map[iu_n, iv] >= 0:
                    add_entry(row_u, int(surface.idx_map[iu_n, iv]), inv_u)
            else:
                iu_n = iu + 1
                if iu_n < Nu and surface.idx_map[iu_n, iv] >= 0:
                    add_entry(row_u, int(surface.idx_map[iu_n, iv]), inv_u)

            if k_cur >= 0:
                add_entry(row_u, int(k_cur), -inv_u)

            if surface.periodic_v:
                iv_n = (iv + 1) % Nv
                if surface.idx_map[iu, iv_n] >= 0:
                    add_entry(row_v, int(surface.idx_map[iu, iv_n]), inv_v)
            else:
                iv_n = iv + 1
                if iv_n < Nv and surface.idx_map[iu, iv_n] >= 0:
                    add_entry(row_v, int(surface.idx_map[iu, iv_n]), inv_v)

            if k_cur >= 0:
                add_entry(row_v, int(k_cur), -inv_v)
            continue

        if surface.periodic_u:
            iu_p = (iu + 1) % Nu
            iu_m = (iu - 1) % Nu
            k_p = int(surface.idx_map[iu_p, iv])
            k_m = int(surface.idx_map[iu_m, iv])
            if k_p >= 0 and k_m >= 0:
                add_entry(row_u, k_p, 0.5 * inv_u)
                add_entry(row_u, k_m, -0.5 * inv_u)
            elif k_p >= 0:
                add_entry(row_u, k_p, inv_u)
                if k_cur >= 0:
                    add_entry(row_u, int(k_cur), -inv_u)
            elif k_m >= 0:
                add_entry(row_u, k_m, -inv_u)
                if k_cur >= 0:
                    add_entry(row_u, int(k_cur), inv_u)
        else:
            iu_p = iu + 1
            iu_m = iu - 1
            k_p = int(surface.idx_map[iu_p, iv]) if iu_p < Nu else -1
            k_m = int(surface.idx_map[iu_m, iv]) if iu_m >= 0 else -1
            if k_p >= 0 and k_m >= 0:
                add_entry(row_u, k_p, 0.5 * inv_u)
                add_entry(row_u, k_m, -0.5 * inv_u)
            elif k_p >= 0:
                add_entry(row_u, k_p, inv_u)
                if k_cur >= 0:
                    add_entry(row_u, int(k_cur), -inv_u)
            elif k_m >= 0:
                add_entry(row_u, k_m, -inv_u)
                if k_cur >= 0:
                    add_entry(row_u, int(k_cur), inv_u)

        if surface.periodic_v:
            iv_p = (iv + 1) % Nv
            iv_m = (iv - 1) % Nv
            k_p = int(surface.idx_map[iu, iv_p])
            k_m = int(surface.idx_map[iu, iv_m])
            if k_p >= 0 and k_m >= 0:
                add_entry(row_v, k_p, 0.5 * inv_v)
                add_entry(row_v, k_m, -0.5 * inv_v)
            elif k_p >= 0:
                add_entry(row_v, k_p, inv_v)
                if k_cur >= 0:
                    add_entry(row_v, int(k_cur), -inv_v)
            elif k_m >= 0:
                add_entry(row_v, k_m, -inv_v)
                if k_cur >= 0:
                    add_entry(row_v, int(k_cur), inv_v)
        else:
            iv_p = iv + 1
            iv_m = iv - 1
            k_p = int(surface.idx_map[iu, iv_p]) if iv_p < Nv else -1
            k_m = int(surface.idx_map[iu, iv_m]) if iv_m >= 0 else -1
            if k_p >= 0 and k_m >= 0:
                add_entry(row_v, k_p, 0.5 * inv_v)
                add_entry(row_v, k_m, -0.5 * inv_v)
            elif k_p >= 0:
                add_entry(row_v, k_p, inv_v)
                if k_cur >= 0:
                    add_entry(row_v, int(k_cur), -inv_v)
            elif k_m >= 0:
                add_entry(row_v, k_m, -inv_v)
                if k_cur >= 0:
                    add_entry(row_v, int(k_cur), inv_v)

    D = coo_matrix((data, (rows_idx, cols_idx)), shape=(2 * nrows, surface.Nint)).tocsr()
    return GradientOperator(D=D, row_coords=row_coords, rows_mode=rows, scheme=scheme)
