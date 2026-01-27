from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray
BoolArray = np.ndarray
IntArray = np.ndarray


@dataclass
class SurfaceGrid:
    """Structured surface grid with masks, geometry, and plotting coordinates."""

    centers_world_uv: FloatArray
    normals_world_uv: FloatArray
    areas_uv: FloatArray
    scale_u: FloatArray
    scale_v: FloatArray
    X_plot: FloatArray
    Y_plot: FloatArray
    interior_mask: BoolArray
    boundary_mask: BoolArray
    idx_map: IntArray
    coords_int: IntArray
    periodic_u: bool
    periodic_v: bool

    def __post_init__(self) -> None:
        self.centers_world_uv = np.asarray(self.centers_world_uv, dtype=float)
        self.normals_world_uv = np.asarray(self.normals_world_uv, dtype=float)
        self.areas_uv = np.asarray(self.areas_uv, dtype=float)
        self.scale_u = np.asarray(self.scale_u, dtype=float)
        self.scale_v = np.asarray(self.scale_v, dtype=float)
        self.X_plot = np.asarray(self.X_plot, dtype=float)
        self.Y_plot = np.asarray(self.Y_plot, dtype=float)

        self.interior_mask = np.asarray(self.interior_mask)
        if self.interior_mask.dtype != bool:
            raise ValueError("interior_mask must be boolean dtype.")
        self.interior_mask = self.interior_mask.astype(bool, copy=False)

        self.boundary_mask = np.asarray(self.boundary_mask)
        if self.boundary_mask.dtype != bool:
            raise ValueError("boundary_mask must be boolean dtype.")
        self.boundary_mask = self.boundary_mask.astype(bool, copy=False)

        self.idx_map = np.asarray(self.idx_map)
        if not np.issubdtype(self.idx_map.dtype, np.integer):
            raise ValueError("idx_map must be integer dtype.")
        self.idx_map = self.idx_map.astype(int, copy=False)

        self.coords_int = np.asarray(self.coords_int)
        if not np.issubdtype(self.coords_int.dtype, np.integer):
            raise ValueError("coords_int must be integer dtype.")
        self.coords_int = self.coords_int.astype(int, copy=False)

        self.periodic_u = bool(self.periodic_u)
        self.periodic_v = bool(self.periodic_v)
        self.validate()

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.centers_world_uv.shape[:2]

    @property
    def Nu(self) -> int:
        return int(self.grid_shape[0])

    @property
    def Nv(self) -> int:
        return int(self.grid_shape[1])

    @property
    def Nint(self) -> int:
        return int(self.coords_int.shape[0])

    def validate(self) -> None:
        """Validate shape, dtype, and mask consistency."""
        if self.centers_world_uv.ndim != 3 or self.centers_world_uv.shape[2] != 3:
            raise ValueError("centers_world_uv must have shape (Nu, Nv, 3).")
        if self.normals_world_uv.shape != self.centers_world_uv.shape:
            raise ValueError("normals_world_uv shape must match centers_world_uv.")

        grid_shape = self.grid_shape
        if self.areas_uv.shape != grid_shape:
            raise ValueError("areas_uv shape must be (Nu, Nv).")
        if self.scale_u.shape != grid_shape:
            raise ValueError("scale_u shape must be (Nu, Nv).")
        if self.scale_v.shape != grid_shape:
            raise ValueError("scale_v shape must be (Nu, Nv).")
        if self.X_plot.shape != grid_shape:
            raise ValueError("X_plot shape must be (Nu, Nv).")
        if self.Y_plot.shape != grid_shape:
            raise ValueError("Y_plot shape must be (Nu, Nv).")
        if self.interior_mask.shape != grid_shape:
            raise ValueError("interior_mask shape must be (Nu, Nv).")
        if self.boundary_mask.shape != grid_shape:
            raise ValueError("boundary_mask shape must be (Nu, Nv).")
        if self.idx_map.shape != grid_shape:
            raise ValueError("idx_map shape must be (Nu, Nv).")

        if np.any(self.interior_mask & self.boundary_mask):
            raise ValueError("interior_mask and boundary_mask must be disjoint.")

        if self.coords_int.ndim != 2 or self.coords_int.shape[1] != 2:
            raise ValueError("coords_int must have shape (Nint, 2).")

        if self.Nint != int(np.count_nonzero(self.interior_mask)):
            raise ValueError("coords_int length must match interior_mask count.")

        if int(np.count_nonzero(self.idx_map >= 0)) != self.Nint:
            raise ValueError("idx_map non-negative count must match Nint.")

        if np.any(self.idx_map[self.boundary_mask] != -1):
            raise ValueError("idx_map must be -1 on boundary_mask.")

        outside_mask = ~(self.interior_mask | self.boundary_mask)
        if np.any(self.idx_map[outside_mask] != -1):
            raise ValueError("idx_map must be -1 on outside_mask.")

        iu = self.coords_int[:, 0]
        iv = self.coords_int[:, 1]
        if np.any(iu < 0) or np.any(iu >= self.Nu) or np.any(iv < 0) or np.any(iv >= self.Nv):
            raise ValueError("coords_int indices out of bounds.")
        if not np.all(self.interior_mask[iu, iv]):
            raise ValueError("coords_int entries must be interior_mask cells.")

        k_vals = self.idx_map[iu, iv]
        if np.any(k_vals < 0):
            raise ValueError("idx_map must be >=0 on coords_int.")
        if not np.array_equal(np.sort(k_vals), np.arange(self.Nint)):
            raise ValueError("idx_map on coords_int must cover 0..Nint-1.")

    def pack(self, S_grid: FloatArray) -> FloatArray:
        """Pack interior values from S_grid into a vector of length Nint."""
        S = np.asarray(S_grid, dtype=float)
        if S.shape != self.grid_shape:
            raise ValueError("S_grid shape must be (Nu, Nv).")
        s_vec = np.empty(self.Nint, dtype=float)
        mask = self.idx_map >= 0
        s_vec[self.idx_map[mask]] = S[mask]
        return s_vec

    def unpack(
        self,
        s_vec: FloatArray,
        *,
        boundary_value: float = 0.0,
        outside_value: float = np.nan,
    ) -> FloatArray:
        """Unpack a vector into S_grid with boundary and outside values filled."""
        s = np.asarray(s_vec, dtype=float).reshape(-1)
        if s.shape[0] != self.Nint:
            raise ValueError("s_vec length must match Nint.")
        S_grid = np.full(self.grid_shape, outside_value, dtype=float)
        S_grid[self.boundary_mask] = boundary_value
        mask = self.idx_map >= 0
        S_grid[mask] = s[self.idx_map[mask]]
        return S_grid
