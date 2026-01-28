from __future__ import annotations

import numpy as np

from gradientcoil.physics.roi_sampling import dedup_points_with_weights


def test_roi_dedup_weights() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    uniq, weights = dedup_points_with_weights(points, eps=0.0)
    assert uniq.shape[0] == 3
    assert np.isclose(weights.sum(), points.shape[0])

    uniq2, weights2 = dedup_points_with_weights(points, eps=1e-12)
    assert uniq2.shape[0] == 3
    assert np.isclose(weights2.sum(), points.shape[0])
