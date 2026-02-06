from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

MU0_BIOT_DEFAULT = 4.0 * math.pi * 1e-7 * 1e3 * 2.0


def load_contours_resampled(path: Path | str) -> tuple[list[dict], dict]:
    loops: list[dict] = []
    meta: dict = {}
    with np.load(Path(path), allow_pickle=True) as npz:
        meta = {
            "delta_s": float(npz["delta_s"]) if "delta_s" in npz.files else None,
            "source_npz": _decode_text(npz["source_npz"]) if "source_npz" in npz.files else None,
            "source_config_json": (
                _decode_text(npz["source_config_json"])
                if "source_config_json" in npz.files
                else None
            ),
        }
        loops_raw = npz["loops_resampled"] if "loops_resampled" in npz.files else None
        if loops_raw is not None:
            if isinstance(loops_raw, np.ndarray):
                items = loops_raw.tolist()
            else:
                items = list(loops_raw)
            for item in items:
                rec = dict(item)
                if "points" not in rec:
                    continue
                rec["points"] = np.asarray(rec["points"], float)
                rec.setdefault("sign", 1)
                rec.setdefault("surface", 0)
                loops.append(rec)
    return loops, meta


def parse_source_config(meta: dict) -> dict | None:
    raw = meta.get("source_config_json")
    if not raw:
        return None
    if isinstance(raw, str):
        text = raw
    else:
        text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def infer_surface_z_positions(config: dict | None, loops: Iterable[dict]) -> dict[int, float]:
    surface_indices = {int(loop.get("surface", 0)) for loop in loops}
    if not surface_indices:
        surface_indices = {0}
    if not config:
        return {idx: 0.0 for idx in surface_indices}

    params = config.get("surface_params", {})
    z0 = float(params.get("z0", 0.0))
    use_two = bool(config.get("use_two_planes", params.get("use_two_planes", False)))
    gap = float(config.get("gap", 0.0))
    if gap == 0.0 and "z_offset" in params:
        gap = 2.0 * float(params.get("z_offset", 0.0))

    if use_two:
        return {0: z0 + 0.5 * gap, 1: z0 - 0.5 * gap}
    return {idx: z0 for idx in surface_indices}


def infer_roi_radius(config: dict | None) -> float | None:
    if not config:
        return None
    roi = config.get("roi", {})
    if "roi_radius" in roi:
        return float(roi["roi_radius"])
    return None


def infer_aperture_radius(config: dict | None) -> float | None:
    if not config:
        return None
    params = config.get("surface_params", {})
    if "R_AP" in params and params["R_AP"] is not None:
        return float(params["R_AP"])
    return None


def infer_g_target(config: dict | None) -> float | None:
    if not config:
        return None
    target = config.get("target_resolved") or config.get("target") or {}
    if not isinstance(target, dict):
        return None
    coeffs = target.get("coeffs")
    if isinstance(coeffs, dict) and "Y" in coeffs:
        try:
            return float(coeffs["Y"])
        except (TypeError, ValueError):
            return None
    return None


def fallback_roi_radius_from_loops(loops: Iterable[dict]) -> float | None:
    max_r2 = None
    for loop in loops:
        pts = np.asarray(loop.get("points", []), float)
        if pts.size == 0:
            continue
        r2 = np.max(np.sum(pts[:, :2] ** 2, axis=1))
        max_r2 = r2 if max_r2 is None else max(max_r2, r2)
    if max_r2 is None:
        return None
    return float(np.sqrt(max_r2))


def biot_savart_Bz_from_loops(
    loops: Iterable[dict],
    z_by_surface: dict[int, float],
    points: np.ndarray,
    *,
    I_per_turn: float,
    mu0: float = MU0_BIOT_DEFAULT,
    chunk_points: int = 5000,
    eps: float = 1e-12,
) -> np.ndarray:
    pts_eval = np.asarray(points, float)
    if pts_eval.ndim != 2 or pts_eval.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    Bz = np.zeros((pts_eval.shape[0],), dtype=float)

    for loop in loops:
        pts2d = np.asarray(loop.get("points", []), float)
        if pts2d.shape[0] < 2:
            continue
        sign = float(loop.get("sign", 1.0))
        surface_idx = int(loop.get("surface", 0))
        z_plane = float(z_by_surface.get(surface_idx, 0.0))

        P3 = np.column_stack([pts2d, np.full(pts2d.shape[0], z_plane)])
        Pn = np.roll(P3, -1, axis=0)
        dl = Pn - P3
        mid = 0.5 * (P3 + Pn)

        for s0 in range(0, pts_eval.shape[0], chunk_points):
            s1 = min(pts_eval.shape[0], s0 + chunk_points)
            R = pts_eval[s0:s1, None, :] - mid[None, :, :]
            R2 = np.sum(R * R, axis=2) + eps
            R32 = R2 * np.sqrt(R2)
            dB = (
                (mu0 / (4.0 * math.pi))
                * (I_per_turn * sign)
                * np.cross(dl[None, :, :], R, axis=2)
                / R32[..., None]
            )
            Bz[s0:s1] += np.sum(dB[..., 2], axis=1)

    return Bz


def _decode_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)
