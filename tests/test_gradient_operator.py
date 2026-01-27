from __future__ import annotations

import numpy as np

from gradientcoil.operators.gradient import build_gradient_operator
from gradientcoil.surfaces.cylinder_unwrap import (
    CylinderUnwrapSurfaceConfig,
    build_cylinder_unwrap_surface,
)
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface


def make_random_grid(surface, rng) -> np.ndarray:
    S = np.zeros(surface.grid_shape, dtype=float)
    S[surface.interior_mask] = rng.standard_normal(surface.Nint)
    S[surface.boundary_mask] = 0.0
    outside_mask = ~(surface.interior_mask | surface.boundary_mask)
    S[outside_mask] = 0.0
    return S


def reference_gradient(surface, row_coords, S) -> np.ndarray:
    Nu, Nv = surface.grid_shape
    G = np.zeros((row_coords.shape[0] * 2,), dtype=float)
    for ridx, (iu, iv) in enumerate(row_coords):
        iu_n = (iu + 1) % Nu if surface.periodic_u else iu + 1
        iv_n = (iv + 1) % Nv if surface.periodic_v else iv + 1

        S_cur = S[iu, iv]
        if iu_n < 0 or iu_n >= Nu:
            S_u = 0.0
        else:
            S_u = S[iu_n, iv]

        if iv_n < 0 or iv_n >= Nv:
            S_v = 0.0
        else:
            S_v = S[iu, iv_n]

        G[2 * ridx] = (S_u - S_cur) / surface.scale_u[iu, iv]
        G[2 * ridx + 1] = (S_v - S_cur) / surface.scale_v[iu, iv]
    return G


def build_surfaces():
    disk = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.25, NR=6, NT=8))
    plane = build_plane_cart_surface(
        PlaneCartSurfaceConfig(PLANE_HALF=0.2, NX=12, NY=10, R_AP=0.15)
    )
    cyl = build_cylinder_unwrap_surface(
        CylinderUnwrapSurfaceConfig(R_CYL=0.2, H=0.3, NZ=6, NTH=8, dirichlet_z_edges=True)
    )
    return [disk, plane, cyl]


def test_gradient_operator_matches_reference() -> None:
    rng = np.random.default_rng(0)
    for surface in build_surfaces():
        S = make_random_grid(surface, rng)
        s = surface.pack(S)
        for rows_mode in ("interior", "active"):
            op = build_gradient_operator(surface, rows=rows_mode)
            G = op.D @ s
            G_ref = reference_gradient(surface, op.row_coords, S)
            assert np.allclose(G, G_ref)


def test_gradient_operator_periodic_seam() -> None:
    rng = np.random.default_rng(1)
    for surface in [build_surfaces()[0], build_surfaces()[2]]:
        S = make_random_grid(surface, rng)
        s = surface.pack(S)
        op = build_gradient_operator(surface, rows="interior")
        G = op.D @ s

        idx = np.where(op.row_coords[:, 1] == surface.Nv - 1)[0]
        assert idx.size > 0
        ridx = int(idx[0])
        iu, iv = op.row_coords[ridx]
        expected = (S[iu, 0] - S[iu, iv]) / surface.scale_v[iu, iv]
        assert np.isclose(G[2 * ridx + 1], expected)
