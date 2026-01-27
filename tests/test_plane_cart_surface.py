from __future__ import annotations

import numpy as np

from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface


def test_plane_cart_surface_masks_and_indices() -> None:
    cfg = PlaneCartSurfaceConfig(PLANE_HALF=0.2, NX=20, NY=16, z0=0.0)
    surface = build_plane_cart_surface(cfg)
    surface.validate()

    expected_boundary = 2 * cfg.NX + 2 * (cfg.NY - 2)
    expected_interior = (cfg.NX - 2) * (cfg.NY - 2)
    assert surface.boundary_mask.sum() == expected_boundary
    assert surface.interior_mask.sum() == expected_interior
    assert not np.any(surface.boundary_mask & surface.interior_mask)
    assert surface.Nint == surface.interior_mask.sum()

    coords = surface.coords_int
    for k in range(coords.shape[0]):
        iu, iv = coords[k]
        assert surface.idx_map[iu, iv] == k

    outside = ~(surface.interior_mask | surface.boundary_mask)
    assert np.count_nonzero(outside) == 0
    assert np.all(surface.idx_map[outside] == -1)

    assert np.isfinite(surface.X_plot).all()
    assert np.isfinite(surface.Y_plot).all()
    assert np.isfinite(surface.centers_world_uv).all()


def test_plane_cart_geometry_scales() -> None:
    cfg = PlaneCartSurfaceConfig(PLANE_HALF=0.3, NX=12, NY=10, z0=0.02)
    surface = build_plane_cart_surface(cfg)

    dx = 2.0 * cfg.PLANE_HALF / cfg.NX
    dy = 2.0 * cfg.PLANE_HALF / cfg.NY
    assert surface.X_plot.shape == (cfg.NY, cfg.NX)
    assert surface.Y_plot.shape == (cfg.NY, cfg.NX)
    assert np.allclose(surface.areas_uv, dx * dy)
    assert np.allclose(surface.scale_u, dy)
    assert np.allclose(surface.scale_v, dx)

    assert surface.periodic_u is False
    assert surface.periodic_v is False

    assert np.allclose(surface.centers_world_uv[..., 2], cfg.z0)
