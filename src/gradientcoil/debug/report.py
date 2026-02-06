from __future__ import annotations

from pathlib import Path

import numpy as np

from gradientcoil.surfaces.base import SurfaceGrid

Array = np.ndarray


def summarize_surfaces(surfaces: list[SurfaceGrid]) -> list[dict]:
    summaries = []
    for surface in surfaces:
        normals = surface.normals_world_uv[surface.interior_mask]
        norms = np.linalg.norm(normals, axis=1) if normals.size else np.array([], dtype=float)
        areas = surface.areas_uv[surface.interior_mask]
        scale_u = surface.scale_u[surface.interior_mask]
        scale_v = surface.scale_v[surface.interior_mask]
        summaries.append(
            {
                "Nu": surface.Nu,
                "Nv": surface.Nv,
                "Nint": surface.Nint,
                "Nboundary": int(np.count_nonzero(surface.boundary_mask)),
                "area_min": float(np.min(areas)) if areas.size else 0.0,
                "area_max": float(np.max(areas)) if areas.size else 0.0,
                "scale_u_min": float(np.min(scale_u)) if scale_u.size else 0.0,
                "scale_u_max": float(np.max(scale_u)) if scale_u.size else 0.0,
                "scale_v_min": float(np.min(scale_v)) if scale_v.size else 0.0,
                "scale_v_max": float(np.max(scale_v)) if scale_v.size else 0.0,
                "normal_norm_min": float(np.min(norms)) if norms.size else 0.0,
                "normal_norm_max": float(np.max(norms)) if norms.size else 0.0,
            }
        )
    return summaries


def summarize_roi(
    points: Array,
    *,
    sampler: str | None = None,
) -> dict:
    if points.size == 0:
        return {"count": 0}
    r = np.linalg.norm(points, axis=1)
    summary = {
        "count": int(points.shape[0]),
        "roi_sampler": sampler,
        "radius_min": float(np.min(r)),
        "radius_max": float(np.max(r)),
        "radius_mean": float(np.mean(r)),
    }
    return summary


def summarize_target(
    coeffs: dict[str, float],
    L_ref: float,
    scale_policy: str,
    Bz_target: Array,
    y_line: Array,
    Bz_y: Array,
    x_line: Array,
    Bz_x: Array,
) -> dict:
    return {
        "coeffs": dict(coeffs),
        "L_ref": float(L_ref),
        "scale_policy": scale_policy,
        "Bz_min": float(np.min(Bz_target)) if Bz_target.size else 0.0,
        "Bz_max": float(np.max(Bz_target)) if Bz_target.size else 0.0,
        "y_line": y_line.tolist(),
        "Bz_y": Bz_y.tolist(),
        "x_line": x_line.tolist(),
        "Bz_x": Bz_x.tolist(),
    }


def summarize_emdm(
    Az: Array | None,
    Bz_dummy: Array | None,
    y_line: Array | None,
    Bz_dummy_y: Array | None,
) -> dict:
    if Az is None:
        return {"skipped": True}

    finite_ratio = float(np.isfinite(Az).mean()) if Az.size else 0.0
    col_norms = np.linalg.norm(Az, axis=0) if Az.size else np.array([], dtype=float)
    stats = {
        "skipped": False,
        "Az_shape": list(Az.shape),
        "finite_ratio": finite_ratio,
        "colnorm_min": float(np.min(col_norms)) if col_norms.size else 0.0,
        "colnorm_median": float(np.median(col_norms)) if col_norms.size else 0.0,
        "colnorm_max": float(np.max(col_norms)) if col_norms.size else 0.0,
    }
    if Bz_dummy is not None:
        stats["Bz_dummy_min"] = float(np.min(Bz_dummy))
        stats["Bz_dummy_max"] = float(np.max(Bz_dummy))
    if y_line is not None and Bz_dummy_y is not None:
        stats["y_line"] = y_line.tolist()
        stats["Bz_dummy_y"] = Bz_dummy_y.tolist()
    return stats


def write_summary_json(path: Path, summary: dict) -> None:
    import json

    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def write_summary_md(path: Path, summary: dict) -> None:
    lines = ["# Debug Bundle Summary", ""]
    lines.append("## Surface")
    for idx, s in enumerate(summary.get("surface", [])):
        lines.append(f"- surface[{idx}]: Nu={s['Nu']} Nv={s['Nv']} Nint={s['Nint']}")
    lines.append("")
    lines.append("## ROI")
    roi = summary.get("roi", {})
    lines.append(f"- count: {roi.get('count', 0)}")
    lines.append(f"- roi_sampler: {roi.get('roi_sampler')}")
    lines.append("")
    lines.append("## Target")
    tgt = summary.get("target", {})
    lines.append(f"- L_ref: {tgt.get('L_ref')}")
    lines.append(f"- scale_policy: {tgt.get('scale_policy')}")
    lines.append("")
    lines.append("## EMDM")
    emdm = summary.get("emdm", {})
    lines.append(f"- skipped: {emdm.get('skipped')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_surface_masks(surfaces: list[SurfaceGrid], out_path: Path) -> None:
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(surfaces)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for idx, surface in enumerate(surfaces):
        ax = axes[0, idx]
        interior = surface.interior_mask.astype(float)
        boundary = surface.boundary_mask.astype(float) * 2.0
        mask = interior + boundary
        ax.pcolormesh(surface.X_plot, surface.Y_plot, mask, shading="auto")
        ax.set_title(f"surface {idx}")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_target_slices(
    y_line: Array,
    Bz_y: Array,
    x_line: Array,
    Bz_x: Array,
    out_path: Path,
) -> None:
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(y_line, Bz_y, label="y-line")
    ax[0].set_title("Bz target along y (x=0,z=0)")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel("Bz [T]")
    ax[0].grid(True)

    ax[1].plot(x_line, Bz_x, label="x-line")
    ax[1].set_title("Bz target along x (y=0,z=0)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Bz [T]")
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_emdm_sanity(y_line: Array, Bz_dummy_y: Array, out_path: Path) -> None:
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(y_line, Bz_dummy_y)
    ax.set_title("EMDM sanity: Bz along y")
    ax.set_xlabel("y")
    ax.set_ylabel("Bz [T]")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
