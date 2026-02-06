from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from gradientcoil.optimize.socp_bz import SocpBzSpec, solve_socp_bz
from gradientcoil.physics.roi_sampling import (
    hammersley_sphere,
    sample_sphere_fibonacci,
    symmetrize_points,
)
from gradientcoil.runs.run_bundle import create_run_dir, save_run_bundle
from gradientcoil.surfaces.cylinder_unwrap import (
    CylinderUnwrapSurfaceConfig,
    build_cylinder_unwrap_surface,
)
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface
from gradientcoil.targets.target_bz_source import ShimBasisTargetBz, target_source_from_dict


def _progress(progress_cb: Callable[[str, str], None] | None, stage: str, info: str) -> None:
    if progress_cb is not None:
        progress_cb(stage, info)


def _build_surfaces(config: dict) -> list:
    surface_type = config["surface_type"]
    params = config["surface_params"]
    use_two_planes = bool(config.get("use_two_planes", False))
    gap = float(config.get("gap", 0.0))
    flip_second = bool(config.get("flip_second_normals", False))

    if surface_type == "disk_polar":
        base = DiskPolarSurfaceConfig(
            R_AP=params["R_AP"],
            NR=params["NR"],
            NT=params["NT"],
            z0=params["z0"],
        )
        if use_two_planes:
            top = build_disk_polar_surface(
                DiskPolarSurfaceConfig(
                    R_AP=params["R_AP"],
                    NR=params["NR"],
                    NT=params["NT"],
                    z0=params["z0"] + 0.5 * gap,
                )
            )
            bot = build_disk_polar_surface(
                DiskPolarSurfaceConfig(
                    R_AP=params["R_AP"],
                    NR=params["NR"],
                    NT=params["NT"],
                    z0=params["z0"] - 0.5 * gap,
                )
            )
            if flip_second:
                bot.normals_world_uv = -bot.normals_world_uv
            return [top, bot]
        return [build_disk_polar_surface(base)]

    if surface_type == "plane_cart":
        base = PlaneCartSurfaceConfig(
            PLANE_HALF=params["PLANE_HALF"],
            NX=params["NX"],
            NY=params["NY"],
            R_AP=params.get("R_AP"),
            z0=params["z0"],
        )
        if use_two_planes:
            top = build_plane_cart_surface(
                PlaneCartSurfaceConfig(
                    PLANE_HALF=params["PLANE_HALF"],
                    NX=params["NX"],
                    NY=params["NY"],
                    R_AP=params.get("R_AP"),
                    z0=params["z0"] + 0.5 * gap,
                )
            )
            bot = build_plane_cart_surface(
                PlaneCartSurfaceConfig(
                    PLANE_HALF=params["PLANE_HALF"],
                    NX=params["NX"],
                    NY=params["NY"],
                    R_AP=params.get("R_AP"),
                    z0=params["z0"] - 0.5 * gap,
                )
            )
            if flip_second:
                bot.normals_world_uv = -bot.normals_world_uv
            return [top, bot]
        return [build_plane_cart_surface(base)]

    if surface_type == "cylinder_unwrap":
        if use_two_planes:
            raise ValueError("two_planes is not supported for cylinder_unwrap.")
        cfg = CylinderUnwrapSurfaceConfig(
            R_CYL=params["R_CYL"],
            H=params["H"],
            NZ=params["NZ"],
            NTH=params["NTH"],
            z_center=params["z_center"],
            dirichlet_z_edges=params["dirichlet_z_edges"],
        )
        return [build_cylinder_unwrap_surface(cfg)]

    raise ValueError(f"Unknown surface type: {surface_type}")


def run_optimization_pipeline(
    config: dict,
    *,
    progress_cb: Callable[[str, str], None] | None = None,
) -> tuple[Path, dict]:
    _progress(progress_cb, "start", "building surfaces")
    surfaces = _build_surfaces(config)

    roi_cfg = config["roi"]
    roi_sampler = roi_cfg.get("sampler", "hammersley")
    if roi_sampler == "fibonacci":
        roi_points = sample_sphere_fibonacci(
            roi_cfg["roi_n"],
            roi_cfg["roi_radius"],
            rotate=bool(roi_cfg.get("roi_rotate", False)),
            seed=0,
        )
    else:
        roi_points = hammersley_sphere(
            roi_cfg["roi_n"],
            roi_cfg["roi_radius"],
            rotate=bool(roi_cfg.get("roi_rotate", False)),
            seed=0,
        )
    roi_axes_raw = roi_cfg.get("sym_axes", ())
    if isinstance(roi_axes_raw, str):
        roi_axes_raw = [ax.strip() for ax in roi_axes_raw.split(",") if ax.strip()]
    roi_axes = tuple(str(ax).lower() for ax in roi_axes_raw) if roi_axes_raw else ()
    if roi_axes:
        roi_points = symmetrize_points(roi_points, axes=roi_axes)
    roi_weights = np.ones((roi_points.shape[0],), dtype=float)

    _progress(progress_cb, "target", "building Bz target")
    tgt_cfg = config["target"]
    source_kind = tgt_cfg.get("source_kind", "basis")
    if source_kind == "basis":
        L_ref_val = roi_cfg["roi_radius"] if tgt_cfg["L_ref"] == "auto" else float(tgt_cfg["L_ref"])
        target_source = ShimBasisTargetBz(
            max_order=int(tgt_cfg["max_order"]),
            coeffs=dict(tgt_cfg["coeffs"]),
            L_ref=float(L_ref_val),
            scale_policy=str(tgt_cfg["scale_policy"]),
        )
    else:
        target_source = target_source_from_dict(tgt_cfg)
    bz_target = target_source.evaluate(roi_points)
    config["target_resolved"] = target_source.to_dict()

    _progress(progress_cb, "solve", "running SOCP solver")
    default_scheme = "central" if config["surface_type"] == "plane_cart" else "forward"
    base_scheme = config["spec"].get("grad_scheme", default_scheme)
    scheme_pitch = config["spec"].get("gradient_scheme_pitch", base_scheme)
    scheme_tv = config["spec"].get("gradient_scheme_tv", base_scheme)
    scheme_power = config["spec"].get("gradient_scheme_power", base_scheme)
    scheme_tgv = config["spec"].get("gradient_scheme_tgv", base_scheme)

    spec = SocpBzSpec(
        use_tv=config["spec"]["use_tv"],
        lambda_tv=float(config["spec"]["lambda_tv"]),
        use_pitch=config["spec"]["use_pitch"],
        J_max=float(config["spec"]["J_max"]),
        use_power=config["spec"]["use_power"],
        lambda_pwr=float(config["spec"]["lambda_pwr"]),
        r_sheet=float(config["spec"]["r_sheet"]),
        use_tgv=config["spec"].get("use_tgv", False),
        alpha1_tgv=float(config["spec"].get("alpha1_tgv", 1e-6)),
        alpha0_tgv=float(config["spec"].get("alpha0_tgv", 1e-6)),
        tgv_area_weights=bool(config["spec"].get("tgv_area_weights", True)),
        gradient_rows_tgv=str(config["spec"].get("gradient_rows_tgv", "interior")),
        gradient_scheme_tgv=scheme_tgv,
        use_curv_r1=bool(config["spec"].get("use_curv_r1", False)),
        lambda_curv_r1=float(config["spec"].get("lambda_curv_r1", 0.0)),
        use_curv_en=bool(config["spec"].get("use_curv_en", False)),
        lambda_curv_en=float(config["spec"].get("lambda_curv_en", 0.0)),
        gradient_scheme_curv=str(config["spec"].get("gradient_scheme_curv", base_scheme)),
        gradient_scheme_pitch=scheme_pitch,
        gradient_scheme_tv=scheme_tv,
        gradient_scheme_power=scheme_power,
        emdm_mode=config["spec"]["emdm_mode"],
        verbose=config["solver"]["verbose"],
        max_iter=config["solver"]["max_iter"],
        time_limit=config["solver"]["time_limit"],
    )

    result = solve_socp_bz(
        roi_points,
        bz_target,
        surfaces,
        spec,
        roi_weights=roi_weights,
        cache_dir=config.get("cache_dir"),
    )

    _progress(progress_cb, "save", "saving run bundle")
    run_dir = create_run_dir(Path(config["out_dir"]), prefix="opt", config=config)
    npz_payload: dict[str, object] = {
        "roi_points_used": roi_points,
        "roi_weights_used": roi_weights,
        "bz_target": bz_target,
        "config_json": json.dumps(config, ensure_ascii=False),
        "solver_stats_json": json.dumps(result.solver_stats, ensure_ascii=False),
        "status": result.status,
    }
    for idx, (surface, S_grid) in enumerate(zip(surfaces, result.S_grids, strict=True)):
        npz_payload[f"S_grid_{idx}"] = S_grid
        npz_payload[f"X_plot_{idx}"] = surface.X_plot
        npz_payload[f"Y_plot_{idx}"] = surface.Y_plot
    if len(surfaces) == 1:
        npz_payload["S_grid"] = result.S_grids[0]
        npz_payload["X_plot"] = surfaces[0].X_plot
        npz_payload["Y_plot"] = surfaces[0].Y_plot

    log_text = f"status={result.status}\nobjective={result.objective}\n"
    save_run_bundle(
        run_dir,
        npz_payload=npz_payload,
        config=config,
        solver=result.solver_stats,
        log_text=log_text,
    )

    summary = {
        "status": result.status,
        "objective": result.objective,
        "run_dir": str(run_dir),
    }
    return run_dir, summary
