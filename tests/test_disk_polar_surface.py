from __future__ import annotations

import numpy as np

from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface


def test_disk_polar_surface_basic() -> None:
    cfg = DiskPolarSurfaceConfig(R_AP=0.25, NR=8, NT=12, z0=0.1, eps_scale=1e-12)
    surface = build_disk_polar_surface(cfg)
    surface.validate()

    assert surface.Nint == (cfg.NR - 1) * cfg.NT
    assert surface.boundary_mask.sum() == cfg.NT
    assert surface.interior_mask.sum() == (cfg.NR - 1) * cfg.NT
    assert surface.periodic_v is True
    assert surface.periodic_u is False

    assert np.all(surface.areas_uv[surface.interior_mask] > 0.0)

    dr = cfg.R_AP / cfg.NR
    dth = 2.0 * np.pi / cfg.NT
    r_c0 = (0.5) * dr
    expected_scale_v0 = max(cfg.eps_scale, r_c0 * dth)
    assert np.allclose(surface.scale_v[0, 0], expected_scale_v0)
    assert surface.scale_v[0, 0] > 0.0

    r_sq = surface.X_plot**2 + surface.Y_plot**2
    assert np.max(np.sqrt(r_sq)) <= cfg.R_AP + 1e-12

    r_c = (np.arange(cfg.NR) + 0.5) * dr
    th_c = (np.arange(cfg.NT) + 0.5) * dth
    Rc, Th = np.meshgrid(r_c, th_c, indexing="ij")
    expected_r_sq = Rc**2
    assert np.allclose(r_sq, expected_r_sq)

    coords = surface.coords_int
    for k in range(coords.shape[0]):
        iu, iv = coords[k]
        assert surface.idx_map[iu, iv] == k
