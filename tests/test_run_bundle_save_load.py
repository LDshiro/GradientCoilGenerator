from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gradientcoil.runs.listing import RunEntry, load_run_config
from gradientcoil.runs.run_bundle import create_run_dir, save_run_bundle


def test_run_bundle_save_load(tmp_path: Path) -> None:
    cfg = {"surface": "plane_cart", "nx": 4}
    solver = {"status": "ok", "iter": 3}
    payload = {
        "S_grid": np.zeros((2, 2), dtype=float),
        "config_json": json.dumps(cfg),
    }

    run_dir = create_run_dir(tmp_path, config=cfg)
    save_run_bundle(
        run_dir,
        npz_payload=payload,
        config=cfg,
        solver=solver,
        log_text="ok",
    )

    assert (run_dir / "config.json").exists()
    assert (run_dir / "solver.json").exists()
    assert (run_dir / "log.txt").exists()
    assert (run_dir / "run.npz").exists()

    with np.load(run_dir / "run.npz", allow_pickle=False) as npz:
        assert "S_grid" in npz

    entry = RunEntry(
        kind="folder",
        path=run_dir,
        mtime=(run_dir / "run.npz").stat().st_mtime,
        label=run_dir.name,
    )
    assert load_run_config(entry) == cfg
