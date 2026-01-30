from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gradientcoil.optimize.socp_bz import SocpBzResult
from gradientcoil.targets.target_bz_source import TargetBzSource


def _json_array(text: str) -> np.ndarray:
    return np.asarray(text, dtype=f"<U{len(text) + 1}")


def _solver_stats_json(stats: dict) -> np.ndarray:
    return _json_array(json.dumps(stats, ensure_ascii=False))


def save_socp_bz_npz(
    path: str | Path,
    *,
    result: SocpBzResult,
    surfaces: list,
    roi_points: np.ndarray,
    roi_weights: np.ndarray,
    bz_target: np.ndarray,
    config: dict,
    target_source: TargetBzSource | None = None,
) -> Path:
    save_path = Path(path)
    config_json = _json_array(json.dumps(config, ensure_ascii=False))

    data: dict[str, np.ndarray] = {
        "roi_points_used": np.asarray(roi_points, dtype=float),
        "roi_weights_used": np.asarray(roi_weights, dtype=float),
        "bz_target": np.asarray(bz_target, dtype=float),
        "config_json": config_json,
        "solver_stats_json": _solver_stats_json(result.solver_stats),
        "status": _json_array(str(result.status)),
    }
    if result.objective is not None:
        data["objective"] = np.asarray(result.objective, dtype=float)

    for idx, (surface, S_grid) in enumerate(zip(surfaces, result.S_grids, strict=True)):
        data[f"S_grid_{idx}"] = np.asarray(S_grid, dtype=float)
        data[f"X_plot_{idx}"] = np.asarray(surface.X_plot, dtype=float)
        data[f"Y_plot_{idx}"] = np.asarray(surface.Y_plot, dtype=float)

    if len(surfaces) == 1:
        data["S_grid"] = np.asarray(result.S_grids[0], dtype=float)
        data["X_plot"] = np.asarray(surfaces[0].X_plot, dtype=float)
        data["Y_plot"] = np.asarray(surfaces[0].Y_plot, dtype=float)

    if target_source is not None:
        target_json = _json_array(json.dumps(target_source.to_dict(), ensure_ascii=False))
        data["target_json"] = target_json
        if hasattr(target_source, "coeffs"):
            coeffs = dict(target_source.coeffs)
            coeffs_json = _json_array(json.dumps(coeffs, ensure_ascii=False))
            coeff_names = np.asarray(list(coeffs.keys()), dtype="<U32")
            coeff_values = np.asarray(list(coeffs.values()), dtype=float)
            data.update(
                {
                    "coeffs_json": coeffs_json,
                    "coeff_names": coeff_names,
                    "coeff_values": coeff_values,
                }
            )
        if hasattr(target_source, "L_ref"):
            data["L_ref"] = np.asarray(float(target_source.L_ref), dtype=float)
        if hasattr(target_source, "scale_policy"):
            data["scale_policy"] = _json_array(str(target_source.scale_policy))
        if hasattr(target_source, "max_order"):
            data["max_order"] = np.asarray(int(target_source.max_order), dtype=int)

    np.savez(save_path, **data)
    return save_path
