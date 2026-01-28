from __future__ import annotations

import numpy as np

from gradientcoil.viz.mesh import periodic_theta_extend


def test_periodic_theta_extend_shapes_and_wrap() -> None:
    nr, nt = 4, 8
    X = np.arange(nr * nt, dtype=float).reshape(nr, nt)
    Y = X + 100.0
    Z = X + 200.0

    X_ext, Y_ext, Z_ext = periodic_theta_extend(X, Y, Z)
    assert X_ext.shape == (nr, nt + 1)
    assert Y_ext.shape == (nr, nt + 1)
    assert Z_ext.shape == (nr, nt + 1)
    assert np.allclose(X_ext[:, 0], X_ext[:, -1])
    assert np.allclose(Y_ext[:, 0], Y_ext[:, -1])
    assert np.allclose(Z_ext[:, 0], Z_ext[:, -1])
