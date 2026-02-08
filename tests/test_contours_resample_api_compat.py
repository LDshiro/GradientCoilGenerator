from __future__ import annotations

import matplotlib.path as mpath
import numpy as np

from gradientcoil.postprocess.contours_resample import _iter_contour_paths


class _DummyCollection:
    def __init__(self, paths: list[mpath.Path]) -> None:
        self._paths = paths

    def get_paths(self) -> list[mpath.Path]:
        return self._paths


class _DummyCSCollections:
    def __init__(self, levels: np.ndarray, paths: list[mpath.Path]) -> None:
        self.levels = levels
        self.collections = [_DummyCollection(paths)]


class _DummyCSAllSegs:
    def __init__(self, levels: np.ndarray, segs: list[list[np.ndarray]]) -> None:
        self.levels = levels
        self.allsegs = segs


def test_iter_contour_paths_with_collections() -> None:
    levels = np.array([0.5], float)
    path = mpath.Path(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], float))
    cs = _DummyCSCollections(levels, [path])
    out = _iter_contour_paths(cs, levels)
    assert len(out) == 1
    assert out[0][0] == 0.5
    assert len(out[0][1]) == 1


def test_iter_contour_paths_with_allsegs_only() -> None:
    levels = np.array([0.5], float)
    seg = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], float)
    cs = _DummyCSAllSegs(levels, [[seg]])
    out = _iter_contour_paths(cs, levels)
    assert len(out) == 1
    assert out[0][0] == 0.5
    assert len(out[0][1]) == 1
    path = out[0][1][0]
    assert isinstance(path, mpath.Path)
    assert path.vertices.shape == (3, 2)
