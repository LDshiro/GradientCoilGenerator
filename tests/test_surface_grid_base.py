from __future__ import annotations

import numpy as np
import pytest

from gradientcoil.surfaces.base import SurfaceGrid


def make_dummy_surface() -> SurfaceGrid:
    Nu, Nv = 3, 4
    u = np.arange(Nu)[:, None]
    v = np.arange(Nv)[None, :]
    X_plot = u + 0.1 * v
    Y_plot = v + 0.2 * u
    centers_world_uv = np.stack([X_plot, Y_plot, np.zeros_like(X_plot)], axis=-1)
    normals_world_uv = np.zeros_like(centers_world_uv)
    normals_world_uv[..., 2] = 1.0
    areas_uv = np.ones((Nu, Nv), dtype=float)
    scale_u = np.full((Nu, Nv), 0.5, dtype=float)
    scale_v = np.full((Nu, Nv), 0.25, dtype=float)

    interior_mask = np.zeros((Nu, Nv), dtype=bool)
    interior_mask[0, 0] = True
    interior_mask[0, 1] = True
    interior_mask[1, 2] = True
    interior_mask[2, 3] = True

    boundary_mask = np.zeros((Nu, Nv), dtype=bool)
    boundary_mask[2, 0] = True
    boundary_mask[2, 1] = True

    coords_int = np.argwhere(interior_mask)
    idx_map = -np.ones((Nu, Nv), dtype=int)
    for k, (iu, iv) in enumerate(coords_int):
        idx_map[iu, iv] = k

    return SurfaceGrid(
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


def test_pack_unpack_roundtrip() -> None:
    surface = make_dummy_surface()
    Nu, Nv = surface.grid_shape
    S_grid = np.arange(Nu * Nv, dtype=float).reshape(Nu, Nv)

    s_vec = surface.pack(S_grid)
    assert s_vec.shape == (surface.Nint,)

    boundary_value = 0.0
    outside_value = np.nan
    S_back = surface.unpack(s_vec, boundary_value=boundary_value, outside_value=outside_value)

    assert np.allclose(S_back[surface.interior_mask], S_grid[surface.interior_mask])
    assert np.allclose(S_back[surface.boundary_mask], boundary_value)
    outside_mask = ~(surface.interior_mask | surface.boundary_mask)
    assert np.all(np.isnan(S_back[outside_mask]))


def test_validate_rejects_overlapping_masks() -> None:
    surface = make_dummy_surface()
    bad_boundary = surface.boundary_mask.copy()
    bad_boundary[0, 0] = True

    with pytest.raises(ValueError, match="disjoint"):
        SurfaceGrid(
            centers_world_uv=surface.centers_world_uv,
            normals_world_uv=surface.normals_world_uv,
            areas_uv=surface.areas_uv,
            scale_u=surface.scale_u,
            scale_v=surface.scale_v,
            X_plot=surface.X_plot,
            Y_plot=surface.Y_plot,
            interior_mask=surface.interior_mask,
            boundary_mask=bad_boundary,
            idx_map=surface.idx_map,
            coords_int=surface.coords_int,
            periodic_u=surface.periodic_u,
            periodic_v=surface.periodic_v,
        )
