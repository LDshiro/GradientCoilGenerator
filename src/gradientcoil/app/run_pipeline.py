from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from gradientcoil.optimize.socp_bz import SocpBzSpec, solve_socp_bz
from gradientcoil.physics.roi_sampling import (
    dedup_points_with_weights,
    hammersley_sphere,
    symmetrize_points,
)
from gradientcoil.runs.run_bundle import create_run_dir, save_run_bundle
from gradientcoil.surfaces.cylinder_unwrap import (
    CylinderUnwrapSurfaceConfig,
    build_cylinder_unwrap_surface,
)
from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface
from gradientcoil.targets.bz_shim import BzShimTargetSpec, standard_shim_terms


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
    roi_points = hammersley_sphere(
        roi_cfg["roi_n"],
        roi_cfg["roi_radius"],
        rotate=bool(roi_cfg.get("roi_rotate", False)),
        seed=0,
    )
    roi_points_raw = symmetrize_points(roi_points, axes=tuple(roi_cfg["sym_axes"]))
    if roi_cfg["roi_dedup"]:
        roi_points, roi_weights = dedup_points_with_weights(
            roi_points_raw, roi_cfg["roi_dedup_eps"]
        )
    else:
        roi_points = roi_points_raw
        roi_weights = np.ones((roi_points.shape[0],), dtype=float)

    _progress(progress_cb, "target", "building Bz target")
    tgt_cfg = config["target"]
    L_ref = roi_cfg["roi_radius"] if tgt_cfg["L_ref"] == "auto" else float(tgt_cfg["L_ref"])
    terms = list(standard_shim_terms(max_order=tgt_cfg["shim_max_order"]).keys())
    target_spec = BzShimTargetSpec(
        coeffs=tgt_cfg["coeffs"],
        terms=terms,
        scale_policy=tgt_cfg["scale_policy"],
        L_ref=float(L_ref),
    )
    bz_target = target_spec.evaluate(roi_points)

    _progress(progress_cb, "solve", "running SOCP solver")
    spec = SocpBzSpec(
        use_tv=config["spec"]["use_tv"],
        lambda_tv=float(config["spec"]["lambda_tv"]),
        use_pitch=config["spec"]["use_pitch"],
        J_max=float(config["spec"]["J_max"]),
        use_power=config["spec"]["use_power"],
        lambda_pwr=float(config["spec"]["lambda_pwr"]),
        r_sheet=float(config["spec"]["r_sheet"]),
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
