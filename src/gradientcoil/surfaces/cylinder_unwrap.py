from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import SurfaceGrid


@dataclass
class CylinderUnwrapSurfaceConfig:
    """Configuration for an unwrapped cylindrical surface grid."""

    R_CYL: float
    H: float
    NZ: int
    NTH: int
    dirichlet_z_edges: bool = True
    z_center: float = 0.0


def build_cylinder_unwrap_surface(cfg: CylinderUnwrapSurfaceConfig) -> SurfaceGrid:
    """Build a SurfaceGrid for a cylinder unwrapped to (z, theta)."""
    if cfg.R_CYL <= 0.0:
        raise ValueError("R_CYL must be positive.")
    if cfg.H <= 0.0:
        raise ValueError("H must be positive.")
    if cfg.NZ < 1:
        raise ValueError("NZ must be >= 1.")
    if cfg.NTH < 3:
        raise ValueError("NTH must be >= 3.")

    nz = int(cfg.NZ)
    nth = int(cfg.NTH)
    r_cyl = float(cfg.R_CYL)
    height = float(cfg.H)
    z_center = float(cfg.z_center)
    dirichlet = bool(cfg.dirichlet_z_edges)

    dz = height / nz
    dth = 2.0 * np.pi / nth
    z_c = z_center - height / 2.0 + (np.arange(nz) + 0.5) * dz
    th_c = (np.arange(nth) + 0.5) * dth

    Zc, Th = np.meshgrid(z_c, th_c, indexing="ij")
    cos_th = np.cos(Th)
    sin_th = np.sin(Th)

    X_world = r_cyl * cos_th
    Y_world = r_cyl * sin_th
    Z_world = Zc

    centers_world_uv = np.stack([X_world, Y_world, Z_world], axis=-1)
    normals_world_uv = np.stack([cos_th, sin_th, np.zeros_like(cos_th)], axis=-1)

    areas_uv = np.full((nz, nth), r_cyl * dz * dth, dtype=float)
    scale_u = np.full((nz, nth), dz, dtype=float)
    scale_v = np.full((nz, nth), r_cyl * dth, dtype=float)

    X_plot = Zc
    Y_plot = r_cyl * Th

    if dirichlet:
        boundary_mask = np.zeros((nz, nth), dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        interior_mask = np.zeros((nz, nth), dtype=bool)
        if nz > 2:
            interior_mask[1:-1, :] = True
    else:
        boundary_mask = np.zeros((nz, nth), dtype=bool)
        interior_mask = np.ones((nz, nth), dtype=bool)

    coords_int = np.argwhere(interior_mask)
    idx_map = -np.ones((nz, nth), dtype=int)
    for k, (iu, iv) in enumerate(coords_int):
        idx_map[iu, iv] = k

    surface = SurfaceGrid(
        centers_world_uv=centers_world_uv,
        normals_world_uv=normals_world_uv,
        areas_uv=areas_uv,
        scale_u=scale_u,
        scale_v=scale_v,
        X_plot=X_plot,
        Y_plot=Y_plot,
        interior_mask=interior_mask,
        boundary_mask=boundary_mask,
        idx_map=idx_map,
        coords_int=coords_int,
        periodic_u=False,
        periodic_v=True,
    )

    surface.R_CYL = r_cyl
    surface.H = height
    surface.NZ = nz
    surface.NTH = nth
    surface.dz = dz
    surface.dth = dth
    surface.z_c = z_c
    surface.th_c = th_c
    surface.z_center = z_center
    surface.dirichlet_z_edges = dirichlet

    return surface
