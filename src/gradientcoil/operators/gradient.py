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


@dataclass
class EdgeDifferenceOperator:
    """Sparse edge-based difference operator for SurfaceGrid."""

    D: csr_matrix
    k0: np.ndarray
    k1: np.ndarray
    inv_h: np.ndarray
    edge_areas: np.ndarray
    edge_dir: np.ndarray
    uv0: np.ndarray
    uv1: np.ndarray
    rows_mode: str


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
            if surface.idx_map[iu_p, iv] >= 0:
                add_entry(row_u, int(surface.idx_map[iu_p, iv]), 0.5 * inv_u)
            if surface.idx_map[iu_m, iv] >= 0:
                add_entry(row_u, int(surface.idx_map[iu_m, iv]), -0.5 * inv_u)
        else:
            iu_p = iu + 1
            iu_m = iu - 1
            if iu_p < Nu and surface.idx_map[iu_p, iv] >= 0:
                add_entry(row_u, int(surface.idx_map[iu_p, iv]), 0.5 * inv_u)
            if iu_m >= 0 and surface.idx_map[iu_m, iv] >= 0:
                add_entry(row_u, int(surface.idx_map[iu_m, iv]), -0.5 * inv_u)

        if surface.periodic_v:
            iv_p = (iv + 1) % Nv
            iv_m = (iv - 1) % Nv
            if surface.idx_map[iu, iv_p] >= 0:
                add_entry(row_v, int(surface.idx_map[iu, iv_p]), 0.5 * inv_v)
            if surface.idx_map[iu, iv_m] >= 0:
                add_entry(row_v, int(surface.idx_map[iu, iv_m]), -0.5 * inv_v)
        else:
            iv_p = iv + 1
            iv_m = iv - 1
            if iv_p < Nv and surface.idx_map[iu, iv_p] >= 0:
                add_entry(row_v, int(surface.idx_map[iu, iv_p]), 0.5 * inv_v)
            if iv_m >= 0 and surface.idx_map[iu, iv_m] >= 0:
                add_entry(row_v, int(surface.idx_map[iu, iv_m]), -0.5 * inv_v)

    D = coo_matrix((data, (rows_idx, cols_idx)), shape=(2 * nrows, surface.Nint)).tocsr()
    return GradientOperator(D=D, row_coords=row_coords, rows_mode=rows, scheme=scheme)


def build_edge_difference_operator(
    surface: SurfaceGrid,
    *,
    rows: str = "interior",
    boundary_value: float = 0.0,
) -> EdgeDifferenceOperator:
    """Build an edge-based difference operator using interior-to-neighbor edges."""
    if boundary_value != 0.0:
        raise ValueError("Only boundary_value=0.0 is supported.")
    if rows not in {"interior", "active"}:
        raise ValueError("rows must be 'interior' or 'active'.")

    Nu, Nv = surface.grid_shape
    coords_int = surface.coords_int

    rows_idx: list[int] = []
    cols_idx: list[int] = []
    data: list[float] = []

    k0_list: list[int] = []
    k1_list: list[int] = []
    inv_h_list: list[float] = []
    edge_area_list: list[float] = []
    edge_dir_list: list[int] = []
    uv0_list: list[list[int]] = []
    uv1_list: list[list[int]] = []

    def add_edge(
        *,
        k0: int,
        k1: int,
        inv_h: float,
        edge_area: float,
        edge_dir: int,
        uv0: tuple[int, int],
        uv1: tuple[int, int],
    ) -> None:
        row = len(k0_list)
        rows_idx.append(row)
        cols_idx.append(k0)
        data.append(-inv_h)
        if k1 >= 0:
            rows_idx.append(row)
            cols_idx.append(k1)
            data.append(inv_h)
        k0_list.append(int(k0))
        k1_list.append(int(k1))
        inv_h_list.append(float(inv_h))
        edge_area_list.append(float(edge_area))
        edge_dir_list.append(int(edge_dir))
        uv0_list.append([int(uv0[0]), int(uv0[1])])
        uv1_list.append([int(uv1[0]), int(uv1[1])])

    def neighbor_index(iu: int, iv: int, du: int, dv: int) -> tuple[int, int] | None:
        if du != 0:
            if surface.periodic_u:
                iu = (iu + du) % Nu
            else:
                iu = iu + du
                if iu < 0 or iu >= Nu:
                    return None
        if dv != 0:
            if surface.periodic_v:
                iv = (iv + dv) % Nv
            else:
                iv = iv + dv
                if iv < 0 or iv >= Nv:
                    return None
        return iu, iv

    for iu, iv in coords_int:
        k0 = int(surface.idx_map[iu, iv])
        if k0 < 0:
            continue

        # Positive directions: include interior-interior and interior-boundary edges
        for du, dv, edge_dir in ((1, 0, 0), (0, 1, 1)):
            nb = neighbor_index(iu, iv, du, dv)
            if nb is None:
                continue
            iu1, iv1 = nb
            if surface.idx_map[iu1, iv1] >= 0:
                k1 = int(surface.idx_map[iu1, iv1])
                h = 0.5 * (surface.scale_u[iu, iv] + surface.scale_u[iu1, iv1])
                if edge_dir == 1:
                    h = 0.5 * (surface.scale_v[iu, iv] + surface.scale_v[iu1, iv1])
                if h <= 0.0:
                    raise ValueError("scale_u/scale_v must be positive.")
                edge_area = 0.5 * (surface.areas_uv[iu, iv] + surface.areas_uv[iu1, iv1])
                add_edge(
                    k0=k0,
                    k1=k1,
                    inv_h=1.0 / h,
                    edge_area=edge_area,
                    edge_dir=edge_dir,
                    uv0=(iu, iv),
                    uv1=(iu1, iv1),
                )
            elif surface.boundary_mask[iu1, iv1]:
                h = 0.5 * (surface.scale_u[iu, iv] + surface.scale_u[iu1, iv1])
                if edge_dir == 1:
                    h = 0.5 * (surface.scale_v[iu, iv] + surface.scale_v[iu1, iv1])
                if h <= 0.0:
                    raise ValueError("scale_u/scale_v must be positive.")
                edge_area = 0.5 * (surface.areas_uv[iu, iv] + surface.areas_uv[iu1, iv1])
                add_edge(
                    k0=k0,
                    k1=-1,
                    inv_h=1.0 / h,
                    edge_area=edge_area,
                    edge_dir=edge_dir,
                    uv0=(iu, iv),
                    uv1=(iu1, iv1),
                )

        # Negative directions: include only interior-boundary edges to avoid duplicates
        for du, dv, edge_dir in ((-1, 0, 0), (0, -1, 1)):
            nb = neighbor_index(iu, iv, du, dv)
            if nb is None:
                continue
            iu1, iv1 = nb
            if surface.boundary_mask[iu1, iv1]:
                h = 0.5 * (surface.scale_u[iu, iv] + surface.scale_u[iu1, iv1])
                if edge_dir == 1:
                    h = 0.5 * (surface.scale_v[iu, iv] + surface.scale_v[iu1, iv1])
                if h <= 0.0:
                    raise ValueError("scale_u/scale_v must be positive.")
                edge_area = 0.5 * (surface.areas_uv[iu, iv] + surface.areas_uv[iu1, iv1])
                add_edge(
                    k0=k0,
                    k1=-1,
                    inv_h=1.0 / h,
                    edge_area=edge_area,
                    edge_dir=edge_dir,
                    uv0=(iu, iv),
                    uv1=(iu1, iv1),
                )

    D = coo_matrix((data, (rows_idx, cols_idx)), shape=(len(k0_list), surface.Nint)).tocsr()
    return EdgeDifferenceOperator(
        D=D,
        k0=np.asarray(k0_list, dtype=int),
        k1=np.asarray(k1_list, dtype=int),
        inv_h=np.asarray(inv_h_list, dtype=float),
        edge_areas=np.asarray(edge_area_list, dtype=float),
        edge_dir=np.asarray(edge_dir_list, dtype=int),
        uv0=np.asarray(uv0_list, dtype=int),
        uv1=np.asarray(uv1_list, dtype=int),
        rows_mode=rows,
    )
