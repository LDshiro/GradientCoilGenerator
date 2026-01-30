from __future__ import annotations

import numpy as np

from gradientcoil.targets.shim_basis import eval_shim_basis, list_shim_terms


def test_list_shim_terms_counts() -> None:
    assert len(list_shim_terms(1)) == 3
    assert len(list_shim_terms(2)) == 8
    assert len(list_shim_terms(3)) == 15


def test_eval_shape() -> None:
    points = np.zeros((5, 3), dtype=float)
    coeffs = {"Y": 1.0}
    out = eval_shim_basis(points, coeffs, max_order=1, L_ref=1.0)
    assert out.shape == (5,)


def test_eval_parity_y() -> None:
    points = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float)
    coeffs = {"Y": 2.0}
    out = eval_shim_basis(points, coeffs, max_order=1, L_ref=1.0)
    assert np.allclose(out, np.array([2.0, -2.0], dtype=float))
