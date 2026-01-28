from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gradientcoil.operators.gradient import build_gradient_operator  # noqa: E402
from gradientcoil.surfaces.plane_cart import (  # noqa: E402
    PlaneCartSurfaceConfig,
    build_plane_cart_surface,
)


@dataclass
class Metrics:
    l2: float
    linf: float
    rel_l2: float
    rel_linf: float


def _metrics(diff: np.ndarray, ref: np.ndarray) -> Metrics:
    l2 = float(np.linalg.norm(diff))
    linf = float(np.max(np.abs(diff)))
    ref_l2 = float(np.linalg.norm(ref))
    ref_linf = float(np.max(np.abs(ref)))
    rel_l2 = l2 / ref_l2 if ref_l2 > 0 else float("nan")
    rel_linf = linf / ref_linf if ref_linf > 0 else float("nan")
    return Metrics(l2=l2, linf=linf, rel_l2=rel_l2, rel_linf=rel_linf)


def _gaussian_field(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    r2 = X * X + Y * Y
    return np.exp(-r2 / (2.0 * sigma * sigma))


def _gaussian_gradients(
    X: np.ndarray, Y: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    S = _gaussian_field(X, Y, sigma)
    inv = 1.0 / (sigma * sigma)
    dSdx = -X * inv * S
    dSdy = -Y * inv * S
    return dSdy, dSdx


def _compute_grad(surface, s_vec: np.ndarray, rows_mode: str, scheme: str) -> np.ndarray:
    op = build_gradient_operator(surface, rows=rows_mode, scheme=scheme)
    g = op.D @ s_vec
    return g.reshape(-1, 2), op.row_coords


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare forward/central gradient schemes.")
    parser.add_argument("--plane-half", type=float, default=0.2)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--rows", choices=["interior", "active", "both"], default="both")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    surface = build_plane_cart_surface(
        PlaneCartSurfaceConfig(
            PLANE_HALF=args.plane_half,
            NX=args.nx,
            NY=args.ny,
            R_AP=None,
            z0=0.0,
        )
    )

    X = surface.X_plot
    Y = surface.Y_plot
    S = _gaussian_field(X, Y, args.sigma)
    S = np.asarray(S, dtype=float)
    S[surface.boundary_mask] = 0.0
    s_vec = surface.pack(S)

    rows_modes = ["interior", "active"] if args.rows == "both" else [args.rows]
    for rows_mode in rows_modes:
        g_fwd, coords = _compute_grad(surface, s_vec, rows_mode, "forward")
        g_ctr, _ = _compute_grad(surface, s_vec, rows_mode, "central")

        diff = g_ctr - g_fwd
        stats = _metrics(diff, g_fwd)
        print(f"[{rows_mode}] central-forward diff:")
        print(f"  L2={stats.l2:.4e}  Linf={stats.linf:.4e}")
        print(f"  rel_L2={stats.rel_l2:.4e}  rel_Linf={stats.rel_linf:.4e}")

        if rows_mode == "interior":
            dSdy, dSdx = _gaussian_gradients(X, Y, args.sigma)
            ref = np.column_stack(
                [dSdy[coords[:, 0], coords[:, 1]], dSdx[coords[:, 0], coords[:, 1]]]
            )
            stats_fwd = _metrics(g_fwd - ref, ref)
            stats_ctr = _metrics(g_ctr - ref, ref)
            print("  vs analytic (interior only):")
            print(f"    forward  rel_L2={stats_fwd.rel_l2:.4e} rel_Linf={stats_fwd.rel_linf:.4e}")
            print(f"    central  rel_L2={stats_ctr.rel_l2:.4e} rel_Linf={stats_ctr.rel_linf:.4e}")

        if args.plot:
            try:
                import matplotlib.pyplot as plt

                diff_norm = np.linalg.norm(diff, axis=1)
                fig, ax = plt.subplots(figsize=(5, 4))
                sc = ax.scatter(
                    X[coords[:, 0], coords[:, 1]], Y[coords[:, 0], coords[:, 1]], c=diff_norm
                )
                fig.colorbar(sc, ax=ax, label="|central-forward|")
                ax.set_title(f"{rows_mode}: |central-forward|")
                ax.set_aspect("equal", adjustable="box")
                plt.show()
            except Exception as exc:
                print(f"plot skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
