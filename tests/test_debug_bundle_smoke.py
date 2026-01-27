from __future__ import annotations

from pathlib import Path

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
        shim_max_order=1,
        coeffs={"Y": 0.02},
        skip_A=True,
    )
    out_dir = generate_debug_bundle(cfg)
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir.with_suffix(".zip")).exists()
