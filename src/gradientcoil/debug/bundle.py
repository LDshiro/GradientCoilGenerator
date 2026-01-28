from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from gradientcoil.debug.report import (
    plot_emdm_sanity,
    plot_surface_masks,
    plot_target_slices,
    summarize_emdm,
    summarize_roi,
    summarize_surfaces,
    summarize_target,
    write_summary_json,
    write_summary_md,
)
from gradientcoil.physics.emdm import build_A_xyz
from gradientcoil.physics.roi_sampling import (
    dedup_points_with_weights,
    hammersley_sphere,
    symmetrize_points,
)
from gradientcoil.surfaces.cylinder_unwrap import (
    CylinderUnwrapSurfaceConfig,
    build_cylinder_unwrap_surface,
)
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface
from gradientcoil.targets.bz_shim import BzShimTargetSpec, standard_shim_terms


@dataclass
class DebugBundleConfig:
    out_dir: Path = Path("runs")
    surface_type: str = "plane_cart"
    gap: float = 0.0
    plane_half: float = 0.2
    nx: int = 20
    ny: int = 16
    z0: float = 0.0
    r_ap: float = 0.2
    nr: int = 8
    nt: int = 16
    r_cyl: float = 0.2
    h: float = 0.3
    nz: int = 6
    nth: int = 12
    z_center: float = 0.0
    dirichlet_top_only: bool = False
    roi_radius: float = 0.1
    roi_n: int = 64
    sym_axes: tuple[str, ...] = ("x", "y", "z")
    roi_dedup: bool = False
    roi_dedup_eps: float = 1e-12
    shim_max_order: int = 2
    coeffs: dict[str, float] = field(default_factory=dict)
    scale_policy: str = "T_per_m"
    L_ref: float | str = "auto"
    build_Az: bool = True
    build_Axyz: bool = False
    chunk: int = 4096
    cache_dir: Path | None = None
    skip_A: bool = False


def _unique_bundle_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"debug_bundle_{stamp}"
    if not base.exists():
        return base
    for k in range(1, 100):
        cand = out_dir / f"debug_bundle_{stamp}_{k:02d}"
        if not cand.exists():
            return cand
    raise RuntimeError("Failed to create unique bundle directory.")


def _reset_masks(surface, interior_mask: np.ndarray, boundary_mask: np.ndarray) -> None:
    interior = np.asarray(interior_mask, dtype=bool)
    boundary = np.asarray(boundary_mask, dtype=bool)
    coords_int = np.argwhere(interior)
    idx_map = -np.ones(interior.shape, dtype=int)
    for k, (iu, iv) in enumerate(coords_int):
        idx_map[iu, iv] = k
    surface.interior_mask = interior
    surface.boundary_mask = boundary
    surface.coords_int = coords_int
    surface.idx_map = idx_map
    surface.validate()


def _build_surfaces(cfg: DebugBundleConfig) -> list:
    if cfg.surface_type == "disk_polar":
        base = DiskPolarSurfaceConfig(R_AP=cfg.r_ap, NR=cfg.nr, NT=cfg.nt, z0=cfg.z0)
        if cfg.gap > 0:
            top = build_disk_polar_surface(
                DiskPolarSurfaceConfig(
                    R_AP=cfg.r_ap, NR=cfg.nr, NT=cfg.nt, z0=cfg.z0 + 0.5 * cfg.gap
                )
            )
            bot = build_disk_polar_surface(
                DiskPolarSurfaceConfig(
                    R_AP=cfg.r_ap, NR=cfg.nr, NT=cfg.nt, z0=cfg.z0 - 0.5 * cfg.gap
                )
            )
            return [top, bot]
        return [build_disk_polar_surface(base)]
    if cfg.surface_type == "plane_cart":
        base = PlaneCartSurfaceConfig(PLANE_HALF=cfg.plane_half, NX=cfg.nx, NY=cfg.ny, z0=cfg.z0)
        if cfg.gap > 0:
            top = build_plane_cart_surface(
                PlaneCartSurfaceConfig(
                    PLANE_HALF=cfg.plane_half, NX=cfg.nx, NY=cfg.ny, z0=cfg.z0 + 0.5 * cfg.gap
                )
            )
            bot = build_plane_cart_surface(
                PlaneCartSurfaceConfig(
                    PLANE_HALF=cfg.plane_half, NX=cfg.nx, NY=cfg.ny, z0=cfg.z0 - 0.5 * cfg.gap
                )
            )
            return [top, bot]
        return [build_plane_cart_surface(base)]
    if cfg.surface_type == "cylinder_unwrap":
        surface = build_cylinder_unwrap_surface(
            CylinderUnwrapSurfaceConfig(
                R_CYL=cfg.r_cyl,
                H=cfg.h,
                NZ=cfg.nz,
                NTH=cfg.nth,
                z_center=cfg.z_center,
                dirichlet_z_edges=not cfg.dirichlet_top_only,
            )
        )
        if cfg.dirichlet_top_only:
            boundary_mask = np.zeros((cfg.nz, cfg.nth), dtype=bool)
            if cfg.nz > 0:
                boundary_mask[-1, :] = True
            interior_mask = np.ones((cfg.nz, cfg.nth), dtype=bool)
            interior_mask[-1, :] = False
            _reset_masks(surface, interior_mask, boundary_mask)
            surface.dirichlet_top_only = True
        return [surface]
    raise ValueError(f"Unknown surface_type: {cfg.surface_type}")


def _parse_coeff_list(items: Iterable[str]) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid coeff '{item}', must be NAME=VALUE.")
        name, val = item.split("=", 1)
        coeffs[name.strip()] = float(val)
    return coeffs


def generate_debug_bundle(cfg: DebugBundleConfig) -> Path:
    out_dir = _unique_bundle_dir(Path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    surfaces = _build_surfaces(cfg)

    surface_data = {}
    for idx, surface in enumerate(surfaces):
        surface_data[f"X_plot_{idx}"] = surface.X_plot
        surface_data[f"Y_plot_{idx}"] = surface.Y_plot
        surface_data[f"interior_mask_{idx}"] = surface.interior_mask
        surface_data[f"boundary_mask_{idx}"] = surface.boundary_mask
        surface_data[f"centers_world_uv_{idx}"] = surface.centers_world_uv
        surface_data[f"normals_world_uv_{idx}"] = surface.normals_world_uv
        surface_data[f"areas_uv_{idx}"] = surface.areas_uv
        surface_data[f"scale_u_{idx}"] = surface.scale_u
        surface_data[f"scale_v_{idx}"] = surface.scale_v
    np.savez(out_dir / "surface.npz", **surface_data)

    roi_points = hammersley_sphere(cfg.roi_n, cfg.roi_radius, rotate=False)
    roi_points_raw = symmetrize_points(roi_points, axes=cfg.sym_axes)
    if cfg.roi_dedup:
        roi_points, roi_weights = dedup_points_with_weights(roi_points_raw, cfg.roi_dedup_eps)
    else:
        roi_points = roi_points_raw
        roi_weights = np.ones((roi_points.shape[0],), dtype=float)
    np.savez(
        out_dir / "roi.npz",
        points_raw=roi_points_raw,
        points=roi_points,
        weights=roi_weights,
        dedup_enabled=cfg.roi_dedup,
        dedup_eps=float(cfg.roi_dedup_eps),
    )

    L_ref = cfg.roi_radius if cfg.L_ref == "auto" else float(cfg.L_ref)
    terms_map = standard_shim_terms(max_order=cfg.shim_max_order)
    terms = list(terms_map.keys())
    target_spec = BzShimTargetSpec(
        coeffs=cfg.coeffs,
        terms=terms,
        scale_policy=cfg.scale_policy,
        L_ref=float(L_ref),
    )
    Bz_target = target_spec.evaluate(roi_points)

    y_line = np.linspace(-cfg.roi_radius, cfg.roi_radius, 201)
    x_line = np.linspace(-cfg.roi_radius, cfg.roi_radius, 201)
    y_points = np.column_stack([np.zeros_like(y_line), y_line, np.zeros_like(y_line)])
    x_points = np.column_stack([x_line, np.zeros_like(x_line), np.zeros_like(x_line)])
    Bz_y = target_spec.evaluate(y_points)
    Bz_x = target_spec.evaluate(x_points)

    import json

    coeffs_json = json.dumps(cfg.coeffs, ensure_ascii=False)
    coeff_names = np.asarray(list(cfg.coeffs.keys()), dtype="<U32")
    coeff_values = np.asarray(list(cfg.coeffs.values()), dtype=float)
    coeffs_json_arr = np.asarray(coeffs_json, dtype=f"<U{len(coeffs_json) + 1}")

    np.savez(
        out_dir / "target.npz",
        coeffs_json=coeffs_json_arr,
        coeff_names=coeff_names,
        coeff_values=coeff_values,
        L_ref=float(L_ref),
        scale_policy=cfg.scale_policy,
        Bz_target=Bz_target,
        roi_points=roi_points,
        roi_weights=roi_weights,
        y_line=y_line,
        Bz_y=Bz_y,
        x_line=x_line,
        Bz_x=Bz_x,
    )

    Az = None
    Bz_dummy = None
    Bz_dummy_y = None
    if not cfg.skip_A and cfg.build_Az:
        Ax, Ay, Az = build_A_xyz(
            roi_points,
            surfaces,
            mode="concat",
            cache_dir=cfg.cache_dir,
        )
        if not cfg.build_Axyz:
            Ax = None
            Ay = None
        s = np.ones((Az.shape[1],), dtype=float)
        Bz_dummy = Az @ s

        _, _, Az_y = build_A_xyz(
            y_points,
            surfaces,
            mode="concat",
            cache_dir=cfg.cache_dir,
        )
        Bz_dummy_y = Az_y @ s

        save_kwargs = {"Az": Az}
        if Ax is not None and Ay is not None:
            save_kwargs.update({"Ax": Ax, "Ay": Ay})
        np.savez(out_dir / "A_emdm.npz", **save_kwargs)

    try:
        from gradientcoil.viz.plotly_setup import make_problem_setup_figure_plotly

        fig = make_problem_setup_figure_plotly(
            surfaces,
            roi_points,
            cfg.roi_radius,
            show_boundary=True,
            show_normals=True,
        )
        fig.write_html(out_dir / "setup_3d.html")
    except ModuleNotFoundError:
        from gradientcoil.post.setup_viz3d import plot_problem_setup_3d

        fig, _ = plot_problem_setup_3d(
            surfaces,
            roi_radius=cfg.roi_radius,
            roi_points=roi_points,
            show_boundary=True,
            show_normals=True,
        )
        fig.savefig(out_dir / "setup_3d.png")

    plot_surface_masks(surfaces, out_dir / "surface_2d.png")
    plot_target_slices(y_line, Bz_y, x_line, Bz_x, out_dir / "target_bz_slices.png")
    if Bz_dummy_y is not None:
        plot_emdm_sanity(y_line, Bz_dummy_y, out_dir / "emdm_sanity.png")

    summary = {
        "surface": summarize_surfaces(surfaces),
        "roi": summarize_roi(
            roi_points,
            points_raw=roi_points_raw,
            weights=roi_weights,
            dedup_enabled=cfg.roi_dedup,
            dedup_eps=float(cfg.roi_dedup_eps),
        ),
        "target": summarize_target(
            cfg.coeffs, float(L_ref), cfg.scale_policy, Bz_target, y_line, Bz_y, x_line, Bz_x
        ),
        "emdm": summarize_emdm(
            Az, Bz_dummy, y_line if Bz_dummy_y is not None else None, Bz_dummy_y
        ),
    }
    write_summary_json(out_dir / "summary.json", summary)
    write_summary_md(out_dir / "summary.md", summary)

    zip_path = out_dir.with_suffix(".zip")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_dir))

    return out_dir


def parse_args(argv: list[str] | None = None) -> DebugBundleConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a debug bundle for validation.")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument(
        "--surface", choices=["disk_polar", "plane_cart", "cylinder_unwrap"], default="plane_cart"
    )
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
    parser.add_argument("--sym-axes", default="x,y,z")
    parser.add_argument("--roi-dedup", action="store_true")
    parser.add_argument("--roi-dedup-eps", type=float, default=1e-12)

    parser.add_argument("--shim-max-order", type=int, default=2)
    parser.add_argument("--coeff", action="append", default=[])
    parser.add_argument("--scale-policy", default="T_per_m")
    parser.add_argument("--L-ref", default="auto")

    parser.add_argument("--build-Axyz", action="store_true")
    parser.add_argument("--skip-A", action="store_true")
    parser.add_argument("--cache-dir", default=None)

    args = parser.parse_args(argv)

    coeffs = _parse_coeff_list(args.coeff)
    sym_axes = tuple(ax.strip() for ax in args.sym_axes.split(",") if ax.strip())
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    return DebugBundleConfig(
        out_dir=Path(args.out_dir),
        surface_type=args.surface,
        gap=args.gap,
        plane_half=args.plane_half,
        nx=args.nx,
        ny=args.ny,
        z0=args.z0,
        r_ap=args.r_ap,
        nr=args.nr,
        nt=args.nt,
        r_cyl=args.r_cyl,
        h=args.h,
        nz=args.nz,
        nth=args.nth,
        z_center=args.z_center,
        dirichlet_top_only=args.dirichlet_top_only,
        roi_radius=args.roi_radius,
        roi_n=args.roi_n,
        sym_axes=sym_axes or ("x", "y", "z"),
        roi_dedup=args.roi_dedup,
        roi_dedup_eps=args.roi_dedup_eps,
        shim_max_order=args.shim_max_order,
        coeffs=coeffs,
        scale_policy=args.scale_policy,
        L_ref=args.L_ref,
        build_Axyz=args.build_Axyz,
        skip_A=args.skip_A,
        cache_dir=cache_dir,
    )
