from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from gradientcoil.optimize.save_npz import save_linear_bz_npz
from gradientcoil.optimize.tsvd_bz import TsvdBzSpec, solve_tsvd_bz
from gradientcoil.physics.roi_sampling import (
    hammersley_sphere,
    sample_sphere_fibonacci,
    symmetrize_points,
)
from gradientcoil.surfaces.cylinder_unwrap import (
    CylinderUnwrapSurfaceConfig,
    build_cylinder_unwrap_surface,
)
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface
from gradientcoil.targets.bz_shim import BzShimTargetSpec, standard_shim_terms


def _parse_coeff_list(items: list[str]) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid coeff '{item}', must be NAME=VALUE.")
        name, val = item.split("=", 1)
        coeffs[name.strip()] = float(val)
    return coeffs


def _dedup_points_with_weights(points: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if eps <= 0.0:
        raise ValueError("roi_dedup_eps must be positive.")
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3), np.zeros((0,), dtype=float)
    keys = np.round(pts / eps).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    p_out = np.zeros((uniq.shape[0], 3), dtype=float)
    w_out = np.bincount(inverse, minlength=uniq.shape[0]).astype(float)
    for idx in range(uniq.shape[0]):
        p_out[idx] = np.mean(pts[inverse == idx], axis=0)
    return p_out, w_out


def _build_surfaces(args) -> list:
    if args.surface == "disk_polar":
        base = DiskPolarSurfaceConfig(R_AP=args.r_ap, NR=args.nr, NT=args.nt, z0=args.z0)
        if args.two_planes:
            top = build_disk_polar_surface(
                DiskPolarSurfaceConfig(
                    R_AP=args.r_ap, NR=args.nr, NT=args.nt, z0=args.z0 + 0.5 * args.gap
                )
            )
            bot = build_disk_polar_surface(
                DiskPolarSurfaceConfig(
                    R_AP=args.r_ap, NR=args.nr, NT=args.nt, z0=args.z0 - 0.5 * args.gap
                )
            )
            return [top, bot]
        return [build_disk_polar_surface(base)]

    if args.surface == "plane_cart":
        base = PlaneCartSurfaceConfig(
            PLANE_HALF=args.plane_half, NX=args.nx, NY=args.ny, z0=args.z0
        )
        if args.two_planes:
            top = build_plane_cart_surface(
                PlaneCartSurfaceConfig(
                    PLANE_HALF=args.plane_half, NX=args.nx, NY=args.ny, z0=args.z0 + 0.5 * args.gap
                )
            )
            bot = build_plane_cart_surface(
                PlaneCartSurfaceConfig(
                    PLANE_HALF=args.plane_half, NX=args.nx, NY=args.ny, z0=args.z0 - 0.5 * args.gap
                )
            )
            return [top, bot]
        return [build_plane_cart_surface(base)]

    if args.surface == "cylinder_unwrap":
        if args.two_planes:
            raise ValueError("two-planes is not supported for cylinder_unwrap.")
        surface = build_cylinder_unwrap_surface(
            CylinderUnwrapSurfaceConfig(
                R_CYL=args.r_cyl,
                H=args.h,
                NZ=args.nz,
                NTH=args.nth,
                z_center=args.z_center,
                dirichlet_z_edges=not args.dirichlet_top_only,
            )
        )
        if args.dirichlet_top_only:
            boundary_mask = np.zeros((args.nz, args.nth), dtype=bool)
            if args.nz > 0:
                boundary_mask[-1, :] = True
            interior_mask = np.ones((args.nz, args.nth), dtype=bool)
            interior_mask[-1, :] = False
            surface.interior_mask = interior_mask
            surface.boundary_mask = boundary_mask
            surface.coords_int = np.argwhere(interior_mask)
            surface.idx_map = -np.ones((args.nz, args.nth), dtype=int)
            for k, (iu, iv) in enumerate(surface.coords_int):
                surface.idx_map[iu, iv] = k
            surface.validate()
        return [surface]

    raise ValueError(f"Unknown surface: {args.surface}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Solve Bz inverse problem with TSVD.")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument(
        "--surface", choices=["disk_polar", "plane_cart", "cylinder_unwrap"], default="plane_cart"
    )
    parser.add_argument("--two-planes", action="store_true")
    parser.add_argument("--gap", type=float, default=0.0)

    parser.add_argument("--plane-half", type=float, default=0.2)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--z0", type=float, default=0.0)

    parser.add_argument("--r-ap", type=float, default=0.2)
    parser.add_argument("--nr", type=int, default=8)
    parser.add_argument("--nt", type=int, default=16)

    parser.add_argument("--r-cyl", type=float, default=0.2)
    parser.add_argument("--h", type=float, default=0.3)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--nth", type=int, default=12)
    parser.add_argument("--z-center", type=float, default=0.0)
    parser.add_argument("--dirichlet-top-only", action="store_true")

    parser.add_argument("--roi-radius", type=float, default=0.1)
    parser.add_argument("--roi-n", type=int, default=64)
    parser.add_argument("--roi-sampler", choices=["hammersley", "fibonacci"], default="hammersley")
    parser.add_argument("--roi-sym-axes", default="", help="Comma-separated axes, e.g. x,y,z")
    parser.add_argument("--roi-dedup", action="store_true")
    parser.add_argument("--roi-dedup-eps", type=float, default=1e-12)

    parser.add_argument("--shim-max-order", type=int, default=1)
    parser.add_argument("--coeff", action="append", default=[])
    parser.add_argument("--scale-policy", default="T_per_m")
    parser.add_argument("--L-ref", default="auto")

    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--svd-method", choices=["full", "svds"], default="full")
    parser.add_argument("--emdm-mode", choices=["shared", "concat"], default="shared")
    parser.add_argument("--cache-dir", default=None)

    args = parser.parse_args(argv)
    surfaces = _build_surfaces(args)

    if args.roi_sampler == "fibonacci":
        roi_points = sample_sphere_fibonacci(args.roi_n, args.roi_radius, rotate=False)
    else:
        roi_points = hammersley_sphere(args.roi_n, args.roi_radius, rotate=False)

    if args.roi_sym_axes:
        axes = tuple(
            ax.strip().lower()
            for ax in args.roi_sym_axes.split(",")
            if ax.strip().lower() in {"x", "y", "z"}
        )
        if axes:
            roi_points = symmetrize_points(roi_points, axes=axes)

    if args.roi_dedup:
        roi_points, roi_weights = _dedup_points_with_weights(roi_points, eps=args.roi_dedup_eps)
    else:
        roi_weights = np.ones((roi_points.shape[0],), dtype=float)

    coeffs = _parse_coeff_list(args.coeff)
    L_ref = args.roi_radius if args.L_ref == "auto" else float(args.L_ref)
    terms = list(standard_shim_terms(max_order=int(args.shim_max_order)).keys())
    target_spec = BzShimTargetSpec(
        coeffs=coeffs,
        terms=terms,
        scale_policy=args.scale_policy,
        L_ref=float(L_ref),
    )
    bz_target = target_spec.evaluate(roi_points)

    spec = TsvdBzSpec(
        k=int(args.k),
        svd_method=str(args.svd_method),
        emdm_mode=str(args.emdm_mode),
    )
    result = solve_tsvd_bz(
        roi_points,
        bz_target,
        surfaces,
        spec,
        roi_weights=roi_weights,
        cache_dir=args.cache_dir,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"tsvd_{stamp}.npz"

    config = {
        "solver": "tsvd",
        "surface": args.surface,
        "two_planes": bool(args.two_planes),
        "gap": float(args.gap),
        "roi_n": int(args.roi_n),
        "roi_radius": float(args.roi_radius),
        "roi_sampler": args.roi_sampler,
        "roi_sym_axes": args.roi_sym_axes,
        "roi_dedup": bool(args.roi_dedup),
        "roi_dedup_eps": float(args.roi_dedup_eps),
        "coeffs": coeffs,
        "max_order": int(args.shim_max_order),
        "scale_policy": args.scale_policy,
        "L_ref": float(L_ref) if args.L_ref != "auto" else "auto",
        "k": int(args.k),
        "svd_method": args.svd_method,
        "emdm_mode": args.emdm_mode,
    }
    save_linear_bz_npz(
        out_path,
        result_status=result.status,
        objective=result.objective,
        s_opt=result.s_opt,
        S_grids=result.S_grids,
        surfaces=surfaces,
        roi_points=roi_points,
        roi_weights=roi_weights,
        bz_target=bz_target,
        config=config,
        solver_stats=result.solver_stats,
        method="tsvd",
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
