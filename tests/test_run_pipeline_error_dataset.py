from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gradientcoil.app.run_pipeline import run_optimization_pipeline


def _base_config(tmp_path: Path, solver_kind: str) -> dict:
    return {
        "out_dir": str(tmp_path),
        "solver_kind": solver_kind,
        "surface_type": "plane_cart",
        "surface_params": {
            "PLANE_HALF": 0.1,
            "NX": 6,
            "NY": 6,
            "R_AP": 0.09,
            "z0": 0.0,
        },
        "use_two_planes": False,
        "gap": 0.0,
        "flip_second_normals": False,
        "roi": {
            "roi_radius": 0.05,
            "roi_n": 8,
            "roi_rotate": False,
            "sampler": "hammersley",
            "sym_axes": [],
        },
        "target": {
            "source_kind": "basis",
            "max_order": 1,
            "coeffs": {"Y": 0.02},
            "scale_policy": "T_per_m",
            "L_ref": "auto",
            "measured_path": "",
        },
        "spec": {
            "emdm_mode": "shared",
            "r_sheet": 1.0,
        },
        "solver": {
            "verbose": False,
            "max_iter": 100,
            "time_limit": None,
        },
        "tikhonov": {
            "lambda_reg": 1.0e-2,
            "reg_operator": "grad",
            "gradient_rows_reg": "interior",
            "r_sheet": 1.0,
            "cg_tol": 1.0e-8,
            "cg_maxiter": 300,
            "emdm_mode": "shared",
        },
        "tsvd": {
            "k": 4,
            "svd_method": "full",
            "emdm_mode": "shared",
        },
    }


@pytest.mark.parametrize("solver_kind", ["tikhonov", "tsvd"])
def test_run_pipeline_saves_error_dataset(tmp_path: Path, solver_kind: str) -> None:
    config = _base_config(tmp_path, solver_kind)
    run_dir, _ = run_optimization_pipeline(config)
    run_npz = run_dir / "run.npz"
    assert run_npz.exists()

    with np.load(run_npz, allow_pickle=False) as npz:
        required = {
            "roi_points_used",
            "bz_target",
            "bz_pred",
            "bz_error_abs",
            "bz_error_valid_mask",
            "bz_error_hist_edges",
            "bz_error_hist_counts_weighted",
            "bz_error_hist_counts_unweighted",
        }
        assert required.issubset(set(npz.files))

        p = int(npz["roi_points_used"].shape[0])
        assert npz["bz_target"].shape == (p,)
        assert npz["bz_pred"].shape == (p,)
        assert npz["bz_error_abs"].shape == (p,)
        assert npz["bz_error_valid_mask"].shape == (p,)

        edges = np.asarray(npz["bz_error_hist_edges"], dtype=float)
        counts_w = np.asarray(npz["bz_error_hist_counts_weighted"], dtype=float)
        counts_u = np.asarray(npz["bz_error_hist_counts_unweighted"], dtype=float)
        assert edges.ndim == 1
        assert counts_w.ndim == 1
        assert counts_u.ndim == 1
        assert counts_w.shape[0] == edges.shape[0] - 1
        assert counts_u.shape[0] == edges.shape[0] - 1

        valid_mask = np.asarray(npz["bz_error_valid_mask"], dtype=bool)
        roi_weights = np.asarray(npz["roi_weights_used"], dtype=float)
        assert np.isclose(np.sum(counts_u), np.sum(valid_mask), rtol=1e-12, atol=1e-12)
        assert np.isclose(
            np.sum(counts_w),
            np.sum(roi_weights[valid_mask]),
            rtol=1e-12,
            atol=1e-12,
        )
