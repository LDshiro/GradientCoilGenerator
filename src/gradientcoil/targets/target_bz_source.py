from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .shim_basis import eval_shim_basis


class TargetBzSource(Protocol):
    def evaluate(self, points_xyz: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class ShimBasisTargetBz:
    max_order: int
    coeffs: dict[str, float]
    L_ref: float
    scale_policy: str = "T_per_m"
    convention: str = "NMR"
    source_kind: str = "basis"

    def evaluate(self, points_xyz: np.ndarray) -> np.ndarray:
        return eval_shim_basis(
            points_xyz,
            coeffs=self.coeffs,
            max_order=self.max_order,
            L_ref=self.L_ref,
            scale_policy=self.scale_policy,
        )

    def to_dict(self) -> dict:
        return {
            "source_kind": self.source_kind,
            "max_order": int(self.max_order),
            "coeffs": dict(self.coeffs),
            "L_ref": float(self.L_ref),
            "scale_policy": str(self.scale_policy),
            "convention": str(self.convention),
        }

    @staticmethod
    def from_dict(data: dict) -> ShimBasisTargetBz:
        return ShimBasisTargetBz(
            max_order=int(data.get("max_order", 1)),
            coeffs=dict(data.get("coeffs", {})),
            L_ref=float(data.get("L_ref", 1.0)),
            scale_policy=str(data.get("scale_policy", "T_per_m")),
            convention=str(data.get("convention", "NMR")),
        )


@dataclass(frozen=True)
class MeasuredTargetBz:
    path: str
    fmt: str | None = None
    source_kind: str = "measured"

    def evaluate(self, points_xyz: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Measured target not implemented yet.")

    def to_dict(self) -> dict:
        return {
            "source_kind": self.source_kind,
            "path": str(self.path),
            "format": self.fmt,
        }

    @staticmethod
    def from_dict(data: dict) -> MeasuredTargetBz:
        return MeasuredTargetBz(
            path=str(data.get("path", "")),
            fmt=data.get("format"),
        )


def target_source_from_dict(data: dict) -> TargetBzSource:
    kind = str(data.get("source_kind", "basis"))
    if kind == "basis":
        return ShimBasisTargetBz.from_dict(data)
    if kind == "measured":
        return MeasuredTargetBz.from_dict(data)
    raise ValueError(f"Unknown target source_kind: {kind}")
