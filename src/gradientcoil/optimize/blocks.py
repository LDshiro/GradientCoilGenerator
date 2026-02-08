from __future__ import annotations

import numpy as np
from scipy.sparse import block_diag, csr_matrix

from gradientcoil.operators.gradient import build_gradient_operator
from gradientcoil.surfaces.base import SurfaceGrid


def build_gradient_block(
    surfaces: list[SurfaceGrid],
    *,
    rows_mode: str,
    emdm_mode: str,
    scheme: str = "forward",
) -> tuple[csr_matrix, np.ndarray]:
    """Build gradient operator block for all surfaces.

    Returns
    -------
    D : csr_matrix
        Shape (2*nrows, N_unknown). Stacks (du,dv) per row.
    areas : ndarray
        Shape (nrows,). Cell areas corresponding to each (du,dv) pair row.

    Notes
    -----
    emdm_mode:
      - "shared": one unknown vector shared across surfaces (use surfaces[0])
      - "concat": unknowns concatenated per-surface (block diagonal)
    """
    if not surfaces:
        raise ValueError("surfaces must be a non-empty list.")
    if scheme not in {"forward", "central"}:
        raise ValueError("scheme must be 'forward' or 'central' for cell gradients.")
    if emdm_mode not in {"shared", "concat"}:
        raise ValueError("emdm_mode must be 'shared' or 'concat'.")

    if emdm_mode == "shared":
        op = build_gradient_operator(surfaces[0], rows=rows_mode, scheme=scheme)
        areas = surfaces[0].areas_uv[op.row_coords[:, 0], op.row_coords[:, 1]]
        return op.D, np.asarray(areas, dtype=float)

    ops = [build_gradient_operator(surface, rows=rows_mode, scheme=scheme) for surface in surfaces]
    D = block_diag([op.D for op in ops], format="csr")
    area_list = [
        surface.areas_uv[op.row_coords[:, 0], op.row_coords[:, 1]]
        for surface, op in zip(surfaces, ops, strict=True)
    ]
    areas = np.concatenate(area_list) if area_list else np.zeros((0,), dtype=float)
    return D, np.asarray(areas, dtype=float)
