from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import SurfaceGrid


@dataclass
class DiskPolarSurfaceConfig:
    """Configuration for a polar disk surface."""

    R_AP: float
    NR: int
    NT: int
    z0: float = 0.0
    eps_scale: float = 1e-12


def build_disk_polar_surface(cfg: DiskPolarSurfaceConfig) -> SurfaceGrid:
    """Build a SurfaceGrid for a polar disk with outer Dirichlet boundary."""
    if cfg.NR < 2:
        raise ValueError("NR must be >= 2 to define an interior ring.")
    if cfg.NT < 3:
        raise ValueError("NT must be >= 3.")
    if cfg.R_AP <= 0.0:
        raise ValueError("R_AP must be positive.")
    if cfg.eps_scale <= 0.0:
        raise ValueError("eps_scale must be positive.")

    NR = int(cfg.NR)
    NT = int(cfg.NT)
    R_AP = float(cfg.R_AP)
    z0 = float(cfg.z0)
    eps_scale = float(cfg.eps_scale)

    dr = R_AP / NR
    dth = 2.0 * np.pi / NT
    r_c = (np.arange(NR) + 0.5) * dr
    th_c = (np.arange(NT) + 0.5) * dth

    Rc, Th = np.meshgrid(r_c, th_c, indexing="ij")
    X_plot = Rc * np.cos(Th)
    Y_plot = Rc * np.sin(Th)
    Z_plot = np.full_like(X_plot, z0)

    centers_world_uv = np.stack([X_plot, Y_plot, Z_plot], axis=-1)
    normals_world_uv = np.zeros_like(centers_world_uv)
    normals_world_uv[..., 2] = 1.0

    areas_uv = Rc * dr * dth
    scale_u = np.full((NR, NT), dr, dtype=float)
    scale_v = np.maximum(eps_scale, Rc * dth)

    interior_mask = np.zeros((NR, NT), dtype=bool)
    interior_mask[: NR - 1, :] = True
    boundary_mask = np.zeros((NR, NT), dtype=bool)
    boundary_mask[NR - 1, :] = True

    coords_int = np.argwhere(interior_mask)
    idx_map = -np.ones((NR, NT), dtype=int)
    for k, (ir, jt) in enumerate(coords_int):
        idx_map[ir, jt] = k

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

    surface.R_AP = R_AP
    surface.NR = NR
    surface.NT = NT
    surface.z0 = z0
    surface.dr = dr
    surface.dth = dth
    surface.r_c = r_c
    surface.th_c = th_c
    surface.Rc = Rc
    surface.Th = Th

    return surface
