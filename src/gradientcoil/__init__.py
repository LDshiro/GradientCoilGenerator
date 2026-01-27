"""Core package for gradient coil design utilities."""

from .physics import MU0_DEFAULT, build_A_xyz, emdm_components
from .post import plot_problem_setup_3d
from .surfaces import (
    CylinderUnwrapSurfaceConfig,
    DiskPolarSurfaceConfig,
    PlaneCartSurfaceConfig,
    SurfaceGrid,
    build_cylinder_unwrap_surface,
    build_disk_polar_surface,
    build_plane_cart_surface,
)

__all__ = [
    "CylinderUnwrapSurfaceConfig",
    "DiskPolarSurfaceConfig",
    "MU0_DEFAULT",
    "PlaneCartSurfaceConfig",
    "SurfaceGrid",
    "build_cylinder_unwrap_surface",
    "build_disk_polar_surface",
    "build_plane_cart_surface",
    "build_A_xyz",
    "emdm_components",
    "plot_problem_setup_3d",
]
