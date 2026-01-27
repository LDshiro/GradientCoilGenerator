from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from gradientcoil.io.results_npz import extract_contour_fields, list_npz_runs, load_npz


def _make_grid(shape=(3, 4)):
    y = np.linspace(-1.0, 1.0, shape[0])
    x = np.linspace(-1.0, 1.0, shape[1])
    return np.meshgrid(x, y)


def test_extract_contour_fields_patterns(tmp_path: Path) -> None:
    X, Y = _make_grid()
    S = X + Y
    path_a = tmp_path / "a.npz"
    np.savez(path_a, X_plot=X, Y_plot=Y, S_grid=S)
    with load_npz(path_a) as npz:
        fields = extract_contour_fields(npz)
    assert len(fields) == 1
    assert fields[0].name == "S_grid"
    assert fields[0].S.shape == X.shape

    path_b = tmp_path / "b.npz"
    np.savez(path_b, Xp=X, Yp=Y, S_polar=S)
    with load_npz(path_b) as npz:
        fields = extract_contour_fields(npz)
    assert len(fields) == 1
    assert fields[0].name == "S_polar"

    path_c = tmp_path / "c.npz"
    np.savez(path_c, XX=X, YY=Y, S_top=S, S_bottom=-S)
    with load_npz(path_c) as npz:
        fields = extract_contour_fields(npz)
    names = {f.name for f in fields}
    assert names == {"S_top", "S_bottom"}


def test_list_npz_runs_sorted(tmp_path: Path) -> None:
    p1 = tmp_path / "old.npz"
    p2 = tmp_path / "new.npz"
    np.savez(p1, dummy=np.array([1.0]))
    np.savez(p2, dummy=np.array([2.0]))

    os.utime(p1, (10, 10))
    os.utime(p2, (20, 20))

    runs = list_npz_runs(tmp_path)
    assert runs[0] == p2
    assert runs[1] == p1
