from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import SurfaceGrid


@dataclass
class PlaneCartSurfaceConfig:
    """Configuration for a rectangular Cartesian plane surface (no mask).

    R_AP is retained for backward compatibility and ignored.
    """

    PLANE_HALF: float
    NX: int
    NY: int
    R_AP: float | None = None
    z0: float = 0.0


def build_plane_cart_surface(cfg: PlaneCartSurfaceConfig) -> SurfaceGrid:
    """Build a SurfaceGrid for a rectangular Cartesian plane (no mask)."""
    if cfg.NX < 1 or cfg.NY < 1:
        raise ValueError("NX and NY must be >= 1.")
    if cfg.PLANE_HALF <= 0.0:
        raise ValueError("PLANE_HALF must be positive.")

    nx = int(cfg.NX)
    ny = int(cfg.NY)
    plane_half = float(cfg.PLANE_HALF)
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

    boundary_mask = np.zeros((ny, nx), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    interior_mask = ~boundary_mask

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
    surface.R_AP = None if cfg.R_AP is None else float(cfg.R_AP)
    surface.z0 = z0
    surface.dx = dx
    surface.dy = dy
    surface.x_c = x_c
    surface.y_c = y_c

    return surface
