from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ContourField:
    name: str
    X: np.ndarray
    Y: np.ndarray
    S: np.ndarray


def list_npz_runs(runs_dir: Path) -> list[Path]:
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []
    files = [p for p in runs_path.iterdir() if p.is_file() and p.suffix.lower() == ".npz"]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(Path(path), allow_pickle=True)


def _add_field(
    fields: list[ContourField],
    npz: np.lib.npyio.NpzFile,
    *,
    name: str,
    x_key: str,
    y_key: str,
    s_key: str,
) -> None:
    if x_key not in npz or y_key not in npz or s_key not in npz:
        return
    X = np.asarray(npz[x_key])
    Y = np.asarray(npz[y_key])
    S = np.asarray(npz[s_key])
    if X.shape != Y.shape or X.shape != S.shape:
        return
    fields.append(ContourField(name=name, X=X, Y=Y, S=S))


def extract_contour_fields(npz: np.lib.npyio.NpzFile) -> list[ContourField]:
    fields: list[ContourField] = []

    _add_field(fields, npz, name="S_grid", x_key="X_plot", y_key="Y_plot", s_key="S_grid")
    _add_field(fields, npz, name="S_polar", x_key="Xp", y_key="Yp", s_key="S_polar")

    if "XX" in npz and "YY" in npz:
        if "S_top" in npz:
            _add_field(fields, npz, name="S_top", x_key="XX", y_key="YY", s_key="S_top")
        if "S_bottom" in npz:
            _add_field(fields, npz, name="S_bottom", x_key="XX", y_key="YY", s_key="S_bottom")

    return fields
