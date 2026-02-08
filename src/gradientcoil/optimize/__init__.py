"""Optimization routines for gradient coil design."""

from .save_npz import save_linear_bz_npz, save_socp_bz_npz
from .socp_bz import SocpBzResult, SocpBzSpec, solve_socp_bz
from .tikhonov_bz import TikhonovBzResult, TikhonovBzSpec, solve_tikhonov_bz
from .tsvd_bz import TsvdBzResult, TsvdBzSpec, solve_tsvd_bz

__all__ = [
    "SocpBzResult",
    "SocpBzSpec",
    "solve_socp_bz",
    "TikhonovBzResult",
    "TikhonovBzSpec",
    "solve_tikhonov_bz",
    "TsvdBzResult",
    "TsvdBzSpec",
    "solve_tsvd_bz",
    "save_socp_bz_npz",
    "save_linear_bz_npz",
]
