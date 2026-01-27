from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from gradientcoil.surfaces.base import SurfaceGrid


def _add_line_segments(
    fig,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    color: str,
    width: int,
    name: str,
) -> None:
    import plotly.graph_objects as go

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for p0, p1 in zip(starts, ends, strict=True):
        xs.extend([p0[0], p1[0], None])
        ys.extend([p0[1], p1[1], None])
        zs.extend([p0[2], p1[2], None])
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": color, "width": width},
            name=name,
            showlegend=False,
        )
    )


def _sphere_wireframe(
    radius: float, *, n_meridians: int = 12, n_parallels: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2 * np.pi, n_meridians, endpoint=False)
    v = np.linspace(0.0, np.pi, n_parallels)

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    for ui in u:
        x = radius * np.cos(ui) * np.sin(v)
        y = radius * np.sin(ui) * np.sin(v)
        z = radius * np.cos(v)
        xs.extend(x.tolist() + [None])
        ys.extend(y.tolist() + [None])
        zs.extend(z.tolist() + [None])

    for vi in v[1:-1]:
        x = radius * np.cos(u) * np.sin(vi)
        y = radius * np.sin(u) * np.sin(vi)
        z = np.full_like(u, radius * np.cos(vi))
        xs.extend(x.tolist() + [None])
        ys.extend(y.tolist() + [None])
        zs.extend(z.tolist() + [None])

    return np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(zs, dtype=float)


def make_problem_setup_figure_plotly(
    surfaces: Iterable[SurfaceGrid],
    roi_points: np.ndarray | None,
    roi_radius: float | None,
    *,
    show_boundary: bool = True,
    show_normals: bool = True,
    normals_stride: int = 10,
    centers_stride: int = 10,
    normal_scale: float = 0.02,
):
    """Create a Plotly figure for surfaces, normals, boundary hulls, and ROI."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for surface in surfaces:
        centers = surface.centers_world_uv[surface.interior_mask]
        if centers.size == 0:
            continue

        c_step = max(1, int(centers_stride))
        centers_sub = centers[::c_step]
        fig.add_trace(
            go.Scatter3d(
                x=centers_sub[:, 0],
                y=centers_sub[:, 1],
                z=centers_sub[:, 2],
                mode="markers",
                marker={"size": 3, "opacity": 0.7, "color": "#1f77b4"},
                name="centers",
                showlegend=False,
            )
        )

        if show_normals:
            n_step = max(1, int(normals_stride))
            centers_n = centers[::n_step]
            normals = surface.normals_world_uv[surface.interior_mask][::n_step]
            ends = centers_n + float(normal_scale) * normals
            _add_line_segments(
                fig,
                centers_n,
                ends,
                color="rgba(44,160,44,0.8)",
                width=3,
                name="normals",
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
                    fig.add_trace(
                        go.Scatter3d(
                            x=hull_pts[:, 0],
                            y=hull_pts[:, 1],
                            z=np.full(hull_pts.shape[0], z_mean),
                            mode="lines",
                            line={"color": "red", "width": 4},
                            name="boundary",
                            showlegend=False,
                        )
                    )
                except Exception:
                    fig.add_trace(
                        go.Scatter3d(
                            x=boundary_centers[:, 0],
                            y=boundary_centers[:, 1],
                            z=boundary_centers[:, 2],
                            mode="markers",
                            marker={"size": 3, "opacity": 0.7, "color": "red"},
                            name="boundary",
                            showlegend=False,
                        )
                    )

    if roi_points is not None:
        roi_pts = np.asarray(roi_points, dtype=float)
        if roi_pts.size:
            fig.add_trace(
                go.Scatter3d(
                    x=roi_pts[:, 0],
                    y=roi_pts[:, 1],
                    z=roi_pts[:, 2],
                    mode="markers",
                    marker={"size": 3, "opacity": 0.6, "color": "#ff7f0e"},
                    name="ROI",
                    showlegend=False,
                )
            )

    if roi_radius is not None and roi_radius > 0:
        xs, ys, zs = _sphere_wireframe(float(roi_radius))
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": "rgba(120,120,120,0.6)", "width": 2},
                name="ROI sphere",
                showlegend=False,
            )
        )

    fig.update_layout(
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    return fig
