"""Linear operators for SurfaceGrid."""

from .gradient import (
    EdgeDifferenceOperator,
    GradientOperator,
    build_edge_difference_operator,
    build_gradient_operator,
)

__all__ = [
    "EdgeDifferenceOperator",
    "GradientOperator",
    "build_edge_difference_operator",
    "build_gradient_operator",
]
