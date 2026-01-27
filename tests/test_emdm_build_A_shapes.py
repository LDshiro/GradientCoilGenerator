from __future__ import annotations

import numpy as np

from gradientcoil.physics.emdm import build_A_xyz
from gradientcoil.physics.roi_sampling import hammersley_sphere
from gradientcoil.surfaces.cylinder_unwrap import (
    CylinderUnwrapSurfaceConfig,
    build_cylinder_unwrap_surface,
)
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface


def test_emdm_build_A_shapes_concat_and_shared(tmp_path) -> None:
    disk = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.2, NR=4, NT=6))
    plane = build_plane_cart_surface(PlaneCartSurfaceConfig(PLANE_HALF=0.2, NX=6, NY=6, R_AP=0.15))
    cyl = build_cylinder_unwrap_surface(
        CylinderUnwrapSurfaceConfig(R_CYL=0.2, H=0.3, NZ=4, NTH=6, dirichlet_z_edges=True)
    )

    points = hammersley_sphere(6, 0.1, rotate=False)
    Ax, Ay, Az = build_A_xyz(points, [disk, plane, cyl], mode="concat", cache_dir=tmp_path)
    total_nint = disk.Nint + plane.Nint + cyl.Nint
    assert Ax.shape == (points.shape[0], total_nint)
    assert Ay.shape == (points.shape[0], total_nint)
    assert Az.shape == (points.shape[0], total_nint)
    assert np.isfinite(Ax).all()
    assert np.isfinite(Ay).all()
    assert np.isfinite(Az).all()

    cache_files = list(tmp_path.glob("emdm_*.npz"))
    assert len(cache_files) >= 1

    Ax2, Ay2, Az2 = build_A_xyz(points, [disk, plane, cyl], mode="concat", cache_dir=tmp_path)
    assert np.allclose(Ax, Ax2)
    assert np.allclose(Ay, Ay2)
    assert np.allclose(Az, Az2)

    disk2 = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.2, NR=4, NT=6, z0=0.05))
    Ax_s, Ay_s, Az_s = build_A_xyz(
        points,
        [disk, disk2],
        mode="shared",
        weights=[1.0, 0.5],
    )
    assert Ax_s.shape == (points.shape[0], disk.Nint)
    assert Ay_s.shape == (points.shape[0], disk.Nint)
    assert Az_s.shape == (points.shape[0], disk.Nint)
    assert np.isfinite(Ax_s).all()
    assert np.isfinite(Ay_s).all()
    assert np.isfinite(Az_s).all()
