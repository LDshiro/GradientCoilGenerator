from __future__ import annotations

import numpy as np

from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface


def test_plane_cart_surface_masks_and_indices() -> None:
    cfg = PlaneCartSurfaceConfig(PLANE_HALF=0.2, NX=20, NY=16, R_AP=0.15, z0=0.0)
    surface = build_plane_cart_surface(cfg)
    surface.validate()

    assert surface.boundary_mask.sum() > 0
    assert surface.interior_mask.sum() > 0
    assert not np.any(surface.boundary_mask & surface.interior_mask)
    assert surface.Nint == surface.interior_mask.sum()

    coords = surface.coords_int
    for k in range(coords.shape[0]):
        iu, iv = coords[k]
        assert surface.idx_map[iu, iv] == k

    mask_circle = surface.mask_circle
    outside = ~mask_circle
    assert np.all(surface.idx_map[outside] == -1)
    assert not np.any(surface.interior_mask[outside])
    assert not np.any(surface.boundary_mask[outside])


def test_plane_cart_geometry_scales() -> None:
    cfg = PlaneCartSurfaceConfig(PLANE_HALF=0.3, NX=12, NY=10, R_AP=0.18, z0=0.02)
    surface = build_plane_cart_surface(cfg)

    dx = 2.0 * cfg.PLANE_HALF / cfg.NX
    dy = 2.0 * cfg.PLANE_HALF / cfg.NY
    assert np.allclose(surface.areas_uv, dx * dy)
    assert np.allclose(surface.scale_u, dy)
    assert np.allclose(surface.scale_v, dx)

    assert surface.periodic_u is False
    assert surface.periodic_v is False

    assert np.allclose(surface.centers_world_uv[..., 2], cfg.z0)
