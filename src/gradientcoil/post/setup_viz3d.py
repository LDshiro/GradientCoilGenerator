from __future__ import annotations

import numpy as np

from gradientcoil.surfaces.base import SurfaceGrid


def plot_problem_setup_3d(
    surfaces: list[SurfaceGrid],
    *,
    roi_radius: float,
    roi_points: np.ndarray,
    show_normals: bool = True,
    normals_stride: int = 4,
    show_boundary: bool = True,
    show_roi_wireframe: bool = True,
):
    """Plot surfaces, normals, boundary hulls, and ROI points in 3D."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for surface in surfaces:
        centers = surface.centers_world_uv[surface.interior_mask]
        if centers.size == 0:
            continue
        step = max(1, int(normals_stride))
        subs = centers[::step]
        ax.scatter(subs[:, 0], subs[:, 1], subs[:, 2], s=6, alpha=0.6)

        if show_normals:
            normals = surface.normals_world_uv[surface.interior_mask][::step]
            ax.quiver(
                subs[:, 0],
                subs[:, 1],
                subs[:, 2],
                normals[:, 0],
                normals[:, 1],
                normals[:, 2],
                length=roi_radius * 0.15,
                normalize=True,
                linewidth=0.5,
                alpha=0.6,
            )

        if show_boundary:
            boundary_centers = surface.centers_world_uv[surface.boundary_mask]
            if boundary_centers.shape[0] >= 3:
                try:
                    from scipy.spatial import ConvexHull

                    xy = boundary_centers[:, :2]
                    hull = ConvexHull(xy)
                    hull_pts = xy[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[:1]])
                    z_mean = float(np.mean(boundary_centers[:, 2]))
                    ax.plot(
                        hull_pts[:, 0],
                        hull_pts[:, 1],
                        z_mean * np.ones(hull_pts.shape[0]),
                        color="red",
                        linewidth=1.2,
                        alpha=0.8,
                    )
                except Exception:
                    ax.scatter(
                        boundary_centers[:, 0],
                        boundary_centers[:, 1],
                        boundary_centers[:, 2],
                        color="red",
                        s=8,
                        alpha=0.6,
                    )

    if roi_points.size:
        roi_pts = np.asarray(roi_points, dtype=float)
        ax.scatter(roi_pts[:, 0], roi_pts[:, 1], roi_pts[:, 2], s=8, alpha=0.5)

    if show_roi_wireframe:
        u = np.linspace(0.0, 2 * np.pi, 24)
        v = np.linspace(0.0, np.pi, 12)
        x = roi_radius * np.outer(np.cos(u), np.sin(v))
        y = roi_radius * np.outer(np.sin(u), np.sin(v))
        z = roi_radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", linewidth=0.4, alpha=0.4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1.0, 1.0, 1.0])
    fig.tight_layout()
    return fig, ax
