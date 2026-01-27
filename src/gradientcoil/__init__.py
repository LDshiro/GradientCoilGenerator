"""Core package for gradient coil design utilities."""

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
    "PlaneCartSurfaceConfig",
    "SurfaceGrid",
    "build_cylinder_unwrap_surface",
    "build_disk_polar_surface",
    "build_plane_cart_surface",
]
