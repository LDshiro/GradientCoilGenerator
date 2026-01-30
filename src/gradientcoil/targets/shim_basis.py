from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ShimTerm:
    name: str
    order: int

    def eval_raw(self, points: Array) -> Array:
        raise NotImplementedError("ShimTerm.eval_raw must be implemented per term.")


def list_shim_terms(max_order: int) -> list[ShimTerm]:
    if max_order < 0:
        raise ValueError("max_order must be >= 0.")

    class _Term(ShimTerm):
        def __init__(self, name: str, order: int, fn) -> None:
            super().__init__(name=name, order=order)
            self._fn = fn

        def eval_raw(self, points: Array) -> Array:
            return self._fn(points)

    terms: list[ShimTerm] = []

    def add(term: ShimTerm) -> None:
        if term.order <= max_order:
            terms.append(term)

    add(_Term("X", 1, lambda p: p[:, 0]))
    add(_Term("Y", 1, lambda p: p[:, 1]))
    add(_Term("Z", 1, lambda p: p[:, 2]))

    add(_Term("Z2", 2, lambda p: 2 * p[:, 2] ** 2 - p[:, 0] ** 2 - p[:, 1] ** 2))
    add(_Term("XZ", 2, lambda p: p[:, 0] * p[:, 2]))
    add(_Term("YZ", 2, lambda p: p[:, 1] * p[:, 2]))
    add(_Term("X2-Y2", 2, lambda p: p[:, 0] ** 2 - p[:, 1] ** 2))
    add(_Term("XY", 2, lambda p: p[:, 0] * p[:, 1]))

    add(
        _Term(
            "Z3",
            3,
            lambda p: p[:, 2] * (2 * p[:, 2] ** 2 - 3 * p[:, 0] ** 2 - 3 * p[:, 1] ** 2),
        )
    )
    add(
        _Term(
            "XZ2",
            3,
            lambda p: p[:, 0] * (4 * p[:, 2] ** 2 - p[:, 0] ** 2 - p[:, 1] ** 2),
        )
    )
    add(
        _Term(
            "YZ2",
            3,
            lambda p: p[:, 1] * (4 * p[:, 2] ** 2 - p[:, 0] ** 2 - p[:, 1] ** 2),
        )
    )
    add(_Term("Z(X2-Y2)", 3, lambda p: p[:, 2] * (p[:, 0] ** 2 - p[:, 1] ** 2)))
    add(_Term("XYZ", 3, lambda p: p[:, 0] * p[:, 1] * p[:, 2]))
    add(_Term("X3-3XY2", 3, lambda p: p[:, 0] * (p[:, 0] ** 2 - 3 * p[:, 1] ** 2)))
    add(_Term("3X2Y-Y3", 3, lambda p: p[:, 1] * (3 * p[:, 0] ** 2 - p[:, 1] ** 2)))

    return terms


def _scale_term(raw: Array, order: int, L_ref: float, scale_policy: str) -> Array:
    if scale_policy != "T_per_m":
        raise ValueError("scale_policy must be 'T_per_m'.")
    if L_ref <= 0.0:
        raise ValueError("L_ref must be positive.")
    if order <= 1:
        return raw
    return raw / (L_ref ** (order - 1))


def eval_shim_basis(
    points: Array,
    coeffs: dict[str, float],
    max_order: int,
    L_ref: float,
    scale_policy: str = "T_per_m",
) -> Array:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    out = np.zeros((P.shape[0],), dtype=float)
    for term in list_shim_terms(max_order):
        coef = float(coeffs.get(term.name, 0.0))
        if coef == 0.0:
            continue
        raw = term.eval_raw(P)
        phi = _scale_term(raw, term.order, L_ref, scale_policy)
        out += coef * phi
    return out
