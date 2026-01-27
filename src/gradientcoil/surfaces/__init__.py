"""Surface definitions for gradient coil optimization."""

from .base import SurfaceGrid
from .disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface

__all__ = ["SurfaceGrid", "DiskPolarSurfaceConfig", "build_disk_polar_surface"]
