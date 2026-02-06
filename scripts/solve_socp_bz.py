from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from gradientcoil.optimize.save_npz import save_socp_bz_npz
from gradientcoil.optimize.socp_bz import SocpBzSpec, solve_socp_bz
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
from gradientcoil.targets.target_bz_source import (
    MeasuredTargetBz,
    ShimBasisTargetBz,
)


def _parse_coeff_list(items: list[str]) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid coeff '{item}', must be NAME=VALUE.")
        name, val = item.split("=", 1)
        coeffs[name.strip()] = float(val)
    return coeffs


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

    parser = argparse.ArgumentParser(description="Solve Bz SOCP with Clarabel.")
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
    parser.add_argument(
        "--roi-sampler",
        choices=["hammersley", "fibonacci"],
        default="hammersley",
    )
    parser.add_argument(
        "--roi-sym-axes",
        default="",
        help="Comma-separated axes to mirror ROI points (e.g., x,y,z).",
    )

    parser.add_argument("--shim-max-order", type=int, default=1)
    parser.add_argument("--coeff", action="append", default=[])
    parser.add_argument("--scale-policy", default="T_per_m")
    parser.add_argument("--L-ref", default="auto")
    parser.add_argument("--target-source", choices=["basis", "measured"], default="basis")
    parser.add_argument("--measured-path", default=None)

    parser.add_argument("--emdm-mode", choices=["shared", "concat"], default="shared")
    parser.add_argument("--use-pitch", action="store_true")
    parser.add_argument("--delta-S", type=float, default=0.0020)
    parser.add_argument("--pitch-min", type=float, default=0.0001)
    parser.add_argument("--J-max", type=float, default=None)
    parser.add_argument("--use-tv", action="store_true")
    parser.add_argument("--lambda-tv", type=float, default=5.00e-8)
    parser.add_argument("--use-power", action="store_true")
    parser.add_argument("--lambda-pwr", type=float, default=3.00e-2)
    parser.add_argument("--r-sheet", type=float, default=0.000492)
    parser.add_argument("--use-tgv", action="store_true")
    parser.add_argument("--alpha1-tgv", type=float, default=1e-6)
    parser.add_argument("--alpha0-tgv", type=float, default=1e-6)
    tgv_group = parser.add_mutually_exclusive_group()
    tgv_group.add_argument("--tgv-area-weights", action="store_true", default=True)
    tgv_group.add_argument("--no-tgv-area-weights", action="store_false", dest="tgv_area_weights")
    parser.add_argument("--use-curv-r1", action="store_true")
    parser.add_argument("--lambda-curv-r1", type=float, default=0.0)
    parser.add_argument("--use-curv-en", action="store_true")
    parser.add_argument("--lambda-curv-en", type=float, default=0.0)
    parser.add_argument(
        "--grad-scheme",
        choices=["forward", "central", "edge"],
        default=None,
        help="Gradient scheme for pitch/tv/power (default: plane_cart=central, others=forward).",
    )
    parser.add_argument("--grad-scheme-pitch", choices=["forward", "central", "edge"], default=None)
    parser.add_argument("--grad-scheme-tv", choices=["forward", "central", "edge"], default=None)
    parser.add_argument("--grad-scheme-power", choices=["forward", "central", "edge"], default=None)
    parser.add_argument("--grad-scheme-tgv", choices=["forward", "central", "edge"], default=None)
    parser.add_argument("--grad-scheme-curv", choices=["forward", "central", "edge"], default=None)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None)
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
    roi_weights = np.ones((roi_points.shape[0],), dtype=float)

    L_ref = args.roi_radius if args.L_ref == "auto" else float(args.L_ref)
    coeffs = _parse_coeff_list(args.coeff)
    if args.target_source == "basis":
        target_source = ShimBasisTargetBz(
            max_order=int(args.shim_max_order),
            coeffs=coeffs,
            L_ref=float(L_ref),
            scale_policy=args.scale_policy,
        )
    else:
        if args.measured_path is None:
            raise ValueError("--measured-path is required for measured target.")
        target_source = MeasuredTargetBz(path=str(args.measured_path))
    bz_target = target_source.evaluate(roi_points)

    if args.J_max is not None:
        J_max = float(args.J_max)
    elif args.delta_S > 0.0 and args.pitch_min > 0.0:
        J_max = float(args.delta_S / args.pitch_min)
    else:
        J_max = 0.0

    default_scheme = "central" if args.surface == "plane_cart" else "forward"
    grad_scheme = args.grad_scheme or default_scheme
    grad_scheme_pitch = args.grad_scheme_pitch or grad_scheme
    grad_scheme_tv = args.grad_scheme_tv or grad_scheme
    grad_scheme_power = args.grad_scheme_power or grad_scheme
    grad_scheme_tgv = args.grad_scheme_tgv or grad_scheme
    grad_scheme_curv = args.grad_scheme_curv or grad_scheme

    spec = SocpBzSpec(
        use_tv=args.use_tv,
        lambda_tv=float(args.lambda_tv),
        use_pitch=args.use_pitch,
        J_max=float(J_max),
        use_power=args.use_power,
        lambda_pwr=float(args.lambda_pwr),
        r_sheet=float(args.r_sheet),
        use_tgv=args.use_tgv,
        alpha1_tgv=float(args.alpha1_tgv),
        alpha0_tgv=float(args.alpha0_tgv),
        tgv_area_weights=bool(args.tgv_area_weights),
        use_curv_r1=bool(args.use_curv_r1),
        lambda_curv_r1=float(args.lambda_curv_r1),
        use_curv_en=bool(args.use_curv_en),
        lambda_curv_en=float(args.lambda_curv_en),
        gradient_scheme_curv=grad_scheme_curv,
        gradient_scheme_tgv=grad_scheme_tgv,
        gradient_scheme_pitch=grad_scheme_pitch,
        gradient_scheme_tv=grad_scheme_tv,
        gradient_scheme_power=grad_scheme_power,
        emdm_mode=args.emdm_mode,
        verbose=args.verbose,
        max_iter=args.max_iter,
        time_limit=args.time_limit,
    )

    result = solve_socp_bz(
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
    out_path = out_dir / f"opt_{stamp}.npz"

    config = {
        "surface": args.surface,
        "two_planes": bool(args.two_planes),
        "gap": float(args.gap),
        "roi_n": int(args.roi_n),
        "roi_radius": float(args.roi_radius),
        "roi_sampler": args.roi_sampler,
        "roi_sym_axes": args.roi_sym_axes,
        "source_kind": args.target_source,
        "coeffs": coeffs,
        "max_order": int(args.shim_max_order),
        "scale_policy": args.scale_policy,
        "L_ref": float(L_ref) if args.L_ref != "auto" else "auto",
        "measured_path": args.measured_path,
        "emdm_mode": args.emdm_mode,
        "use_pitch": bool(args.use_pitch),
        "J_max": float(J_max),
        "use_tv": bool(args.use_tv),
        "lambda_tv": float(args.lambda_tv),
        "use_power": bool(args.use_power),
        "lambda_pwr": float(args.lambda_pwr),
        "r_sheet": float(args.r_sheet),
        "use_tgv": bool(args.use_tgv),
        "alpha1_tgv": float(args.alpha1_tgv),
        "alpha0_tgv": float(args.alpha0_tgv),
        "tgv_area_weights": bool(args.tgv_area_weights),
        "use_curv_r1": bool(args.use_curv_r1),
        "lambda_curv_r1": float(args.lambda_curv_r1),
        "use_curv_en": bool(args.use_curv_en),
        "lambda_curv_en": float(args.lambda_curv_en),
        "grad_scheme": args.grad_scheme,
        "gradient_scheme_curv": grad_scheme_curv,
        "gradient_scheme_tgv": grad_scheme_tgv,
        "gradient_scheme_pitch": grad_scheme_pitch,
        "gradient_scheme_tv": grad_scheme_tv,
        "gradient_scheme_power": grad_scheme_power,
        "solver": "CLARABEL",
        "max_iter": args.max_iter,
        "time_limit": args.time_limit,
    }

    save_socp_bz_npz(
        out_path,
        result=result,
        surfaces=surfaces,
        roi_points=roi_points,
        roi_weights=roi_weights,
        bz_target=bz_target,
        config=config,
        target_source=target_source,
    )

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
