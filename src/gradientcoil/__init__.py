"""Core package for gradient coil design utilities."""

from .surfaces import (
    DiskPolarSurfaceConfig,
    PlaneCartSurfaceConfig,
    SurfaceGrid,
    build_disk_polar_surface,
    build_plane_cart_surface,
)

__all__ = [
    "DiskPolarSurfaceConfig",
    "PlaneCartSurfaceConfig",
    "SurfaceGrid",
    "build_disk_polar_surface",
    "build_plane_cart_surface",
]
