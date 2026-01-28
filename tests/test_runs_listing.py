from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from gradientcoil.runs.listing import list_runs, load_run_config, load_run_npz, load_run_solver


def test_list_runs_and_loaders(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    folder = runs_dir / "opt_test"
    folder.mkdir()
    cfg = {"alpha": 1.0}
    solver = {"status": "ok"}
    (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (folder / "solver.json").write_text(json.dumps(solver), encoding="utf-8")
    np.savez(folder / "run.npz", dummy=np.array([1.0]), config_json=json.dumps(cfg))

    legacy = runs_dir / "legacy.npz"
    np.savez(
        legacy,
        dummy=np.array([2.0]),
        config_json=json.dumps({"beta": 2}),
        solver_stats_json=json.dumps({"iter": 3}),
    )

    os.utime(folder / "run.npz", (10, 10))
    os.utime(legacy, (20, 20))

    entries = list_runs(runs_dir)
    labels = [entry.label for entry in entries]
    assert labels == ["legacy.npz", "opt_test"]

    folder_entry = next(entry for entry in entries if entry.kind == "folder")
    legacy_entry = next(entry for entry in entries if entry.kind == "npz")

    assert load_run_config(folder_entry) == cfg
    assert load_run_solver(folder_entry) == solver
    assert load_run_config(legacy_entry) == {"beta": 2}
    assert load_run_solver(legacy_entry) == {"iter": 3}

    with load_run_npz(folder_entry) as npz:
        assert "dummy" in npz
