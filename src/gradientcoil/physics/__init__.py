"""Physics utilities for gradient coil modeling."""

from .emdm import MU0_DEFAULT, build_A_xyz, dipole_field_matrices, emdm_components
from .roi_sampling import hammersley_sphere, symmetrize_points

__all__ = [
    "MU0_DEFAULT",
    "build_A_xyz",
    "dipole_field_matrices",
    "emdm_components",
    "hammersley_sphere",
    "symmetrize_points",
]
