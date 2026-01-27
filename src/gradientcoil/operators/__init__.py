"""Linear operators for SurfaceGrid."""

from .gradient import GradientOperator, build_gradient_operator

__all__ = ["GradientOperator", "build_gradient_operator"]
