"""Surface definitions for gradient coil optimization."""

from .base import SurfaceGrid
from .disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from .plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface

__all__ = [
    "DiskPolarSurfaceConfig",
    "PlaneCartSurfaceConfig",
    "SurfaceGrid",
    "build_disk_polar_surface",
    "build_plane_cart_surface",
]
