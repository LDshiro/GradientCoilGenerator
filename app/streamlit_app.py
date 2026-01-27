from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _build_surface(surface_type: str, params: dict) -> object:
    from gradientcoil.surfaces.cylinder_unwrap import (
        CylinderUnwrapSurfaceConfig,
        build_cylinder_unwrap_surface,
    )
    from gradientcoil.surfaces.disk_polar import DiskPolarSurfaceConfig, build_disk_polar_surface
    from gradientcoil.surfaces.plane_cart import PlaneCartSurfaceConfig, build_plane_cart_surface

    if surface_type == "disk_polar":
        cfg = DiskPolarSurfaceConfig(
            R_AP=params["R_AP"],
            NR=params["NR"],
            NT=params["NT"],
            z0=params["z0"],
        )
        return build_disk_polar_surface(cfg)
    if surface_type == "plane_cart":
        cfg = PlaneCartSurfaceConfig(
            PLANE_HALF=params["PLANE_HALF"],
            NX=params["NX"],
            NY=params["NY"],
            z0=params["z0"],
        )
        return build_plane_cart_surface(cfg)
    if surface_type == "cylinder_unwrap":
        cfg = CylinderUnwrapSurfaceConfig(
            R_CYL=params["R_CYL"],
            H=params["H"],
            NZ=params["NZ"],
            NTH=params["NTH"],
            z_center=params["z_center"],
            dirichlet_z_edges=params["dirichlet_z_edges"],
        )
        return build_cylinder_unwrap_surface(cfg)
    raise ValueError(f"Unknown surface type: {surface_type}")


def _build_surfaces(surface_type: str, params: dict) -> list[object]:
    if surface_type in {"disk_polar", "plane_cart"} and params.get("use_two_planes"):
        z_center = float(params.get("z0", 0.0))
        z_offset = float(params.get("z_offset", 0.0))
        top_params = dict(params)
        bottom_params = dict(params)
        top_params["z0"] = z_center + z_offset
        bottom_params["z0"] = z_center - z_offset
        top = _build_surface(surface_type, top_params)
        bottom = _build_surface(surface_type, bottom_params)
        if params.get("flip_second_normals"):
            bottom.normals_world_uv = -bottom.normals_world_uv
        return [top, bottom]
    return [_build_surface(surface_type, params)]


def _param_panel(surface_type: str) -> dict:
    import streamlit as st

    if surface_type == "disk_polar":
        return {
            "R_AP": st.number_input("R_AP", min_value=1e-4, value=0.2, format="%.4f"),
            "NR": int(st.number_input("NR", min_value=2, value=8, step=1)),
            "NT": int(st.number_input("NT", min_value=3, value=16, step=1)),
            "z0": st.number_input("z0", value=0.0, format="%.4f"),
            "use_two_planes": st.checkbox("use_two_planes", value=False),
            "z_offset": st.number_input("z_offset", value=0.05, format="%.4f"),
            "flip_second_normals": st.checkbox("flip_second_normals", value=False),
        }
    if surface_type == "plane_cart":
        return {
            "PLANE_HALF": st.number_input("PLANE_HALF", min_value=1e-4, value=0.2, format="%.4f"),
            "NX": int(st.number_input("NX", min_value=2, value=20, step=1)),
            "NY": int(st.number_input("NY", min_value=2, value=16, step=1)),
            "z0": st.number_input("z0", value=0.0, format="%.4f"),
            "use_two_planes": st.checkbox("use_two_planes", value=False),
            "z_offset": st.number_input("z_offset", value=0.05, format="%.4f"),
            "flip_second_normals": st.checkbox("flip_second_normals", value=False),
        }
    return {
        "R_CYL": st.number_input("R_CYL", min_value=1e-4, value=0.2, format="%.4f"),
        "H": st.number_input("H", min_value=1e-4, value=0.3, format="%.4f"),
        "NZ": int(st.number_input("NZ", min_value=1, value=6, step=1)),
        "NTH": int(st.number_input("NTH", min_value=3, value=12, step=1)),
        "z_center": st.number_input("z_center", value=0.0, format="%.4f"),
        "dirichlet_z_edges": st.checkbox("dirichlet_z_edges", value=True),
    }


def main() -> None:
    import streamlit as st

    from gradientcoil.io.results_npz import extract_contour_fields, list_npz_runs, load_npz
    from gradientcoil.physics.roi_sampling import hammersley_sphere
    from gradientcoil.viz.plotly_setup import make_problem_setup_figure_plotly

    st.set_page_config(page_title="GradientCoil GUI", layout="wide")

    tab_conditions, tab_results = st.tabs(["条件", "結果"])

    with tab_conditions:
        col_plot, col_params = st.columns([3, 1])
        with col_params:
            st.subheader("パラメータ")
            surface_type = st.selectbox(
                "surface",
                ["disk_polar", "plane_cart", "cylinder_unwrap"],
                index=0,
            )
            params = _param_panel(surface_type)

            st.markdown("### ROI")
            roi_radius = st.number_input("ROI radius", min_value=0.0, value=0.1, format="%.4f")
            roi_points_n = int(st.number_input("ROI points", min_value=0, value=64, step=1))
            roi_rotate = st.checkbox("ROI rotate", value=False)

            st.markdown("### 表示")
            centers_stride = int(st.number_input("centers_stride", min_value=1, value=10))
            normals_stride = int(st.number_input("normals_stride", min_value=1, value=10))
            normal_scale = st.number_input("normal_scale", min_value=0.0, value=0.02, format="%.4f")

            st.markdown("### まとめ")
            summary = {
                "surface": surface_type,
                **params,
                "roi_radius": roi_radius,
                "roi_points": roi_points_n,
                "roi_rotate": roi_rotate,
                "centers_stride": centers_stride,
                "normals_stride": normals_stride,
                "normal_scale": normal_scale,
            }
            st.json(summary)

        with col_plot:
            st.subheader("3D 設定ビュー")
            try:
                surfaces = _build_surfaces(surface_type, params)
                if roi_points_n > 0 and roi_radius > 0:
                    roi_points = hammersley_sphere(
                        roi_points_n, roi_radius, rotate=roi_rotate, seed=0
                    )
                else:
                    roi_points = None
                fig = make_problem_setup_figure_plotly(
                    surfaces,
                    roi_points,
                    roi_radius if roi_radius > 0 else None,
                    show_boundary=True,
                    show_normals=True,
                    normals_stride=normals_stride,
                    centers_stride=centers_stride,
                    normal_scale=normal_scale,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"表示に失敗しました: {exc}")

    with tab_results:
        runs_dir = ROOT / "runs"
        runs_dir.mkdir(exist_ok=True)
        npz_paths = list_npz_runs(runs_dir)
        if not npz_paths:
            st.info("runs/ に npz がありません。")
            return

        labels = [p.name for p in npz_paths]
        selected = st.selectbox("NPZを選択", labels)
        path = npz_paths[labels.index(selected)]
        with load_npz(path) as npz:
            fields = extract_contour_fields(npz)

        if not fields:
            st.warning("表示できる等高線データが見つかりません。")
            return

        import matplotlib.pyplot as plt

        for field in fields:
            fig, ax = plt.subplots(figsize=(5, 4))
            cs = ax.contourf(field.X, field.Y, field.S, levels=30)
            fig.colorbar(cs, ax=ax)
            ax.set_title(field.name)
            ax.set_aspect("equal", adjustable="box")
            st.pyplot(fig)


if __name__ == "__main__":
    main()
