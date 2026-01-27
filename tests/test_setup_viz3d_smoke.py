from __future__ import annotations

import matplotlib

from gradientcoil.physics.roi_sampling import hammersley_sphere
from gradientcoil.post.setup_viz3d import plot_problem_setup_3d
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface


def test_setup_viz3d_smoke() -> None:
    matplotlib.use("Agg")
    surface = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.2, NR=4, NT=6))
    roi_points = hammersley_sphere(8, 0.1, rotate=False)

    fig, ax = plot_problem_setup_3d(
        [surface],
        roi_points=roi_points,
        roi_radius=0.1,
        show_normals=False,
        show_boundary=True,
    )
    assert fig is not None
    assert ax is not None
    fig.clf()
