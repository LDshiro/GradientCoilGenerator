from __future__ import annotations

import numpy as np


def periodic_theta_extend(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extend polar grid data by duplicating the first theta column at the end."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError("X, Y, Z must have the same shape.")
    if X.ndim != 2:
        raise ValueError("X, Y, Z must be 2D arrays.")

    X_ext = np.concatenate([X, X[:, :1]], axis=1)
    Y_ext = np.concatenate([Y, Y[:, :1]], axis=1)
    Z_ext = np.concatenate([Z, Z[:, :1]], axis=1)
    return X_ext, Y_ext, Z_ext
