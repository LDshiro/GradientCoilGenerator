from __future__ import annotations

import numpy as np

from gradientcoil.physics.emdm import MU0_DEFAULT, emdm_components


def test_emdm_single_dipole_on_axis() -> None:
    points = np.array([[0.0, 0.0, 1.0]])
    centers = np.array([[0.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    areas = np.array([1.0])

    Ax, Ay, Az = emdm_components(points, centers, normals, areas)
    expected_bz = MU0_DEFAULT / (2.0 * np.pi)
    assert np.allclose(Ax[0, 0], 0.0, atol=1e-12)
    assert np.allclose(Ay[0, 0], 0.0, atol=1e-12)
    assert np.allclose(Az[0, 0], expected_bz, rtol=1e-6)
