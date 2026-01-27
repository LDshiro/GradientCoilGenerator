"""Surface definitions for gradient coil optimization."""

from .base import SurfaceGrid
from .cylinder_unwrap import CylinderUnwrapSurfaceConfig, build_cylinder_unwrap_surface
from .disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from .plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface

__all__ = [
    "CylinderUnwrapSurfaceConfig",
    "DiskPolarSurfaceConfig",
    "PlaneCartSurfaceConfig",
    "SurfaceGrid",
    "build_cylinder_unwrap_surface",
    "build_disk_polar_surface",
    "build_plane_cart_surface",
]
