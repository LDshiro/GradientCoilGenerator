from __future__ import annotations

from pathlib import Path

import numpy as np

from gradientcoil.debug.bundle import DebugBundleConfig, generate_debug_bundle
from gradientcoil.targets.bz_shim import coeffs_from_npz


def test_target_npz_serialization(tmp_path: Path) -> None:
    cfg = DebugBundleConfig(
        out_dir=tmp_path,
        surface_type="plane_cart",
        plane_half=0.1,
        nx=6,
        ny=6,
        roi_radius=0.05,
        roi_n=8,
        shim_max_order=1,
        coeffs={"Y": 0.02},
        skip_A=True,
    )
    out_dir = generate_debug_bundle(cfg)
    target_npz = np.load(out_dir / "target.npz", allow_pickle=False)
    coeffs = coeffs_from_npz(target_npz)
    assert coeffs == {"Y": 0.02}
    assert "coeffs_json" in target_npz
    assert "coeff_names" in target_npz
    assert "coeff_values" in target_npz
