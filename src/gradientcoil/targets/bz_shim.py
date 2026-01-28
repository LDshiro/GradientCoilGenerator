from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .base import BzTarget

Array = np.ndarray


@dataclass(frozen=True)
class ShimTerm:
    name: str
    order: int
    display_name: str
    latex: str
    eval_poly: Callable[[Array], Array]


def standard_shim_terms(max_order: int = 3) -> dict[str, ShimTerm]:
    """Return standard harmonic Cartesian shim basis terms up to max_order."""
    terms: dict[str, ShimTerm] = {}

    def add(term: ShimTerm) -> None:
        if term.order <= max_order:
            terms[term.name] = term

    add(
        ShimTerm(
            name="Z0",
            order=0,
            display_name="Z0",
            latex="1",
            eval_poly=lambda p: np.ones((p.shape[0],), dtype=float),
        )
    )
    add(
        ShimTerm(
            name="X",
            order=1,
            display_name="X",
            latex="x",
            eval_poly=lambda p: p[:, 0],
        )
    )
    add(
        ShimTerm(
            name="Y",
            order=1,
            display_name="Y",
            latex="y",
            eval_poly=lambda p: p[:, 1],
        )
    )
    add(
        ShimTerm(
            name="Z",
            order=1,
            display_name="Z",
            latex="z",
            eval_poly=lambda p: p[:, 2],
        )
    )
    add(
        ShimTerm(
            name="Z2",
            order=2,
            display_name="Z2",
            latex="z^2 - 0.5(x^2+y^2)",
            eval_poly=lambda p: p[:, 2] ** 2 - 0.5 * (p[:, 0] ** 2 + p[:, 1] ** 2),
        )
    )
    add(
        ShimTerm(
            name="ZX",
            order=2,
            display_name="ZX",
            latex="zx",
            eval_poly=lambda p: p[:, 2] * p[:, 0],
        )
    )
    add(
        ShimTerm(
            name="ZY",
            order=2,
            display_name="ZY",
            latex="zy",
            eval_poly=lambda p: p[:, 2] * p[:, 1],
        )
    )
    add(
        ShimTerm(
            name="X2Y2",
            order=2,
            display_name="X2-Y2",
            latex="x^2-y^2",
            eval_poly=lambda p: p[:, 0] ** 2 - p[:, 1] ** 2,
        )
    )
    add(
        ShimTerm(
            name="XY",
            order=2,
            display_name="XY",
            latex="xy",
            eval_poly=lambda p: p[:, 0] * p[:, 1],
        )
    )
    add(
        ShimTerm(
            name="Z3",
            order=3,
            display_name="Z3",
            latex="z^3 - 1.5z(x^2+y^2)",
            eval_poly=lambda p: p[:, 2] ** 3 - 1.5 * p[:, 2] * (p[:, 0] ** 2 + p[:, 1] ** 2),
        )
    )
    add(
        ShimTerm(
            name="Z2X",
            order=3,
            display_name="Z2X",
            latex="xz^2 - 0.25x(x^2+y^2)",
            eval_poly=lambda p: p[:, 0] * p[:, 2] ** 2
            - 0.25 * p[:, 0] * (p[:, 0] ** 2 + p[:, 1] ** 2),
        )
    )
    add(
        ShimTerm(
            name="Z2Y",
            order=3,
            display_name="Z2Y",
            latex="yz^2 - 0.25y(x^2+y^2)",
            eval_poly=lambda p: p[:, 1] * p[:, 2] ** 2
            - 0.25 * p[:, 1] * (p[:, 0] ** 2 + p[:, 1] ** 2),
        )
    )
    add(
        ShimTerm(
            name="ZX2Y2",
            order=3,
            display_name="ZX2-ZY2",
            latex="z(x^2-y^2)",
            eval_poly=lambda p: p[:, 2] * (p[:, 0] ** 2 - p[:, 1] ** 2),
        )
    )
    add(
        ShimTerm(
            name="ZXY",
            order=3,
            display_name="ZXY",
            latex="zxy",
            eval_poly=lambda p: p[:, 2] * p[:, 0] * p[:, 1],
        )
    )
    add(
        ShimTerm(
            name="X3",
            order=3,
            display_name="X3-3XY2",
            latex="x(x^2-3y^2)",
            eval_poly=lambda p: p[:, 0] * (p[:, 0] ** 2 - 3.0 * p[:, 1] ** 2),
        )
    )
    add(
        ShimTerm(
            name="Y3",
            order=3,
            display_name="3X2Y-Y3",
            latex="y(3x^2-y^2)",
            eval_poly=lambda p: p[:, 1] * (3.0 * p[:, 0] ** 2 - p[:, 1] ** 2),
        )
    )
    return terms


def _scale_phi(phi: Array, order: int, L_ref: float, policy: str) -> Array:
    if policy == "native":
        return phi
    if policy != "T_per_m":
        raise ValueError("scale_policy must be 'T_per_m' or 'native'.")
    if L_ref <= 0.0:
        raise ValueError("L_ref must be positive.")
    if order == 0:
        return phi * L_ref
    if order == 1:
        return phi
    return phi / (L_ref ** (order - 1))


@dataclass
class BzShimTargetSpec(BzTarget):
    coeffs: dict[str, float]
    terms: list[str]
    scale_policy: str = "T_per_m"
    L_ref: float = 1.0
    origin: Array = field(default_factory=lambda: np.zeros((3,), dtype=float))
    convention: str = "harmonic_cartesian_v1"

    def __post_init__(self) -> None:
        if self.L_ref <= 0.0:
            raise ValueError("L_ref must be positive.")
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)

    def design_matrix(self, points: Array) -> Array:
        P = np.asarray(points, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must have shape (P, 3).")

        terms_map = standard_shim_terms(max_order=3)
        coords = P - self.origin[None, :]
        cols = []
        for name in self.terms:
            if name not in terms_map:
                raise ValueError(f"Unknown shim term: {name}")
            term = terms_map[name]
            phi = term.eval_poly(coords)
            psi = _scale_phi(phi, term.order, self.L_ref, self.scale_policy)
            cols.append(psi.reshape(-1, 1))
        if not cols:
            return np.zeros((P.shape[0], 0), dtype=float)
        return np.hstack(cols)

    def evaluate(self, points: Array) -> Array:
        Phi = self.design_matrix(points)
        if Phi.shape[1] == 0:
            return np.zeros((Phi.shape[0],), dtype=float)
        alpha = np.array([self.coeffs.get(name, 0.0) for name in self.terms], dtype=float)
        return Phi @ alpha

    def to_dict(self) -> dict:
        return {
            "coeffs": dict(self.coeffs),
            "terms": list(self.terms),
            "scale_policy": self.scale_policy,
            "L_ref": float(self.L_ref),
            "origin": self.origin.tolist(),
            "convention": self.convention,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BzShimTargetSpec:
        return cls(
            coeffs=dict(data.get("coeffs", {})),
            terms=list(data.get("terms", [])),
            scale_policy=str(data.get("scale_policy", "T_per_m")),
            L_ref=float(data.get("L_ref", 1.0)),
            origin=np.asarray(data.get("origin", [0.0, 0.0, 0.0]), dtype=float),
            convention=str(data.get("convention", "harmonic_cartesian_v1")),
        )


def coeffs_from_npz(npz: dict) -> dict[str, float]:
    """Recover coeffs from a target.npz-like dict."""
    if "coeffs_json" in npz:
        try:
            raw = npz["coeffs_json"]
            if isinstance(raw, np.ndarray):
                raw = raw.item() if raw.shape == () else raw[0]
            return dict(json.loads(str(raw)))
        except Exception:
            pass
    if "coeff_names" in npz and "coeff_values" in npz:
        names = [str(x) for x in npz["coeff_names"]]
        values = np.asarray(npz["coeff_values"], dtype=float)
        return dict(zip(names, values, strict=False))
    if "coeffs" in npz:
        try:
            return dict(npz["coeffs"].item())
        except Exception:
            return {}
    return {}
