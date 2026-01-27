from __future__ import annotations

import numpy as np

from gradientcoil.surfaces.cylinder_unwrap import (
    CylinderUnwrapSurfaceConfig,
    build_cylinder_unwrap_surface,
)


def test_cylinder_unwrap_surface_dirichlet_edges() -> None:
    cfg = CylinderUnwrapSurfaceConfig(R_CYL=0.25, H=0.4, NZ=6, NTH=10, dirichlet_z_edges=True)
    surface = build_cylinder_unwrap_surface(cfg)
    surface.validate()

    assert surface.periodic_u is False
    assert surface.periodic_v is True

    assert surface.boundary_mask.sum() == 2 * cfg.NTH
    assert surface.interior_mask.sum() == (cfg.NZ - 2) * cfg.NTH

    radius = np.sqrt(surface.centers_world_uv[..., 0] ** 2 + surface.centers_world_uv[..., 1] ** 2)
    assert np.allclose(radius, cfg.R_CYL)

    norms = np.linalg.norm(surface.normals_world_uv, axis=2)
    assert np.allclose(norms, 1.0)

    dz = cfg.H / cfg.NZ
    dth = 2.0 * np.pi / cfg.NTH
    expected_area = cfg.R_CYL * dz * dth
    assert np.allclose(surface.areas_uv, expected_area)
    assert np.allclose(surface.scale_v, cfg.R_CYL * dth)

    coords = surface.coords_int
    for k in range(coords.shape[0]):
        iu, iv = coords[k]
        assert surface.idx_map[iu, iv] == k
