from __future__ import annotations

from typing import Protocol

import numpy as np


class BzTarget(Protocol):
    """Protocol for scalar Bz target generators."""

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Return Bz field at points with shape (P, 3)."""
        ...
