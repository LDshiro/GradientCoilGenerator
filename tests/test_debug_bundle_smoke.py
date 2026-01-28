from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gradientcoil.debug.bundle import DebugBundleConfig, generate_debug_bundle


def test_debug_bundle_smoke(tmp_path: Path) -> None:
    cfg = DebugBundleConfig(
        out_dir=tmp_path,
        surface_type="plane_cart",
        plane_half=0.1,
        nx=8,
        ny=8,
        roi_radius=0.05,
        roi_n=10,
        roi_dedup=True,
        roi_dedup_eps=1e-12,
        shim_max_order=1,
        coeffs={"Y": 0.02},
        skip_A=True,
    )
    out_dir = generate_debug_bundle(cfg)
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir.with_suffix(".zip")).exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    roi_summary = summary.get("roi", {})
    assert "roi_count_raw" in roi_summary
    assert "roi_count_used" in roi_summary
    assert "roi_weights_sum" in roi_summary
    target_npz = np.load(out_dir / "target.npz", allow_pickle=False)
    assert "coeffs_json" in target_npz
    assert "coeff_names" in target_npz
    assert "coeff_values" in target_npz
    roi_npz = np.load(out_dir / "roi.npz", allow_pickle=False)
    assert "points_raw" in roi_npz
    assert "weights" in roi_npz
