from __future__ import annotations

import numpy as np

from gradientcoil.operators.gradient import build_edge_difference_operator
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


def build_surfaces():
    disk = build_disk_polar_surface(DiskPolarSurfaceConfig(R_AP=0.25, NR=6, NT=8))
    plane = build_plane_cart_surface(
        PlaneCartSurfaceConfig(PLANE_HALF=0.2, NX=12, NY=10, R_AP=0.15)
    )
    cyl = build_cylinder_unwrap_surface(
        CylinderUnwrapSurfaceConfig(R_CYL=0.2, H=0.3, NZ=6, NTH=8, dirichlet_z_edges=True)
    )
    return [disk, plane, cyl]


def test_edge_difference_operator_matches_reference() -> None:
    rng = np.random.default_rng(0)
    for surface in build_surfaces():
        S = make_random_grid(surface, rng)
        s = surface.pack(S)
        op = build_edge_difference_operator(surface)
        g = op.D @ s

        S0 = S[op.uv0[:, 0], op.uv0[:, 1]]
        S1 = np.zeros_like(S0)
        has_k1 = op.k1 >= 0
        if np.any(has_k1):
            S1[has_k1] = S[op.uv1[has_k1, 0], op.uv1[has_k1, 1]]
        g_ref = (S1 - S0) * op.inv_h
        assert np.allclose(g, g_ref)


def test_edge_difference_operator_bounds_and_periodic_seam() -> None:
    disk, _, cyl = build_surfaces()
    for surface in [disk, cyl]:
        op = build_edge_difference_operator(surface)
        Nu, Nv = surface.grid_shape
        assert op.uv1.min() >= 0
        assert op.uv1[:, 0].max() < Nu
        assert op.uv1[:, 1].max() < Nv

        seam = (op.edge_dir == 1) & (op.uv0[:, 1] == Nv - 1) & (op.uv1[:, 1] == 0)
        assert np.any(seam)

    plane = build_surfaces()[1]
    op_plane = build_edge_difference_operator(plane)
    assert np.any(op_plane.k1 < 0)
