from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import SurfaceGrid


@dataclass
class PlaneCartSurfaceConfig:
    """Configuration for a Cartesian plane surface clipped by a circular aperture."""

    PLANE_HALF: float
    NX: int
    NY: int
    R_AP: float
    z0: float = 0.0


def build_plane_cart_surface(cfg: PlaneCartSurfaceConfig) -> SurfaceGrid:
    """Build a SurfaceGrid for a Cartesian plane with circular aperture."""
    if cfg.NX < 1 or cfg.NY < 1:
        raise ValueError("NX and NY must be >= 1.")
    if cfg.PLANE_HALF <= 0.0:
        raise ValueError("PLANE_HALF must be positive.")
    if cfg.R_AP <= 0.0:
        raise ValueError("R_AP must be positive.")

    nx = int(cfg.NX)
    ny = int(cfg.NY)
    plane_half = float(cfg.PLANE_HALF)
    r_ap = float(cfg.R_AP)
    z0 = float(cfg.z0)

    dx = 2.0 * plane_half / nx
    dy = 2.0 * plane_half / ny
    x_c = -plane_half + (np.arange(nx) + 0.5) * dx
    y_c = -plane_half + (np.arange(ny) + 0.5) * dy

    X_plot, Y_plot = np.meshgrid(x_c, y_c, indexing="xy")
    Z_plot = np.full_like(X_plot, z0)

    centers_world_uv = np.stack([X_plot, Y_plot, Z_plot], axis=-1)
    normals_world_uv = np.zeros_like(centers_world_uv)
    normals_world_uv[..., 2] = 1.0

    areas_uv = np.full((ny, nx), dx * dy, dtype=float)
    scale_u = np.full((ny, nx), dy, dtype=float)
    scale_v = np.full((ny, nx), dx, dtype=float)

    mask_circle = (X_plot**2 + Y_plot**2) <= (r_ap**2)

    inside_up = np.zeros_like(mask_circle, dtype=bool)
    inside_up[1:, :] = mask_circle[:-1, :]
    inside_down = np.zeros_like(mask_circle, dtype=bool)
    inside_down[:-1, :] = mask_circle[1:, :]
    inside_left = np.zeros_like(mask_circle, dtype=bool)
    inside_left[:, 1:] = mask_circle[:, :-1]
    inside_right = np.zeros_like(mask_circle, dtype=bool)
    inside_right[:, :-1] = mask_circle[:, 1:]

    boundary_mask = mask_circle & ~(inside_up & inside_down & inside_left & inside_right)
    interior_mask = mask_circle & ~boundary_mask

    coords_int = np.argwhere(interior_mask)
    idx_map = -np.ones((ny, nx), dtype=int)
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
        periodic_v=False,
    )

    surface.PLANE_HALF = plane_half
    surface.NX = nx
    surface.NY = ny
    surface.R_AP = r_ap
    surface.z0 = z0
    surface.dx = dx
    surface.dy = dy
    surface.x_c = x_c
    surface.y_c = y_c
    surface.mask_circle = mask_circle

    return surface
