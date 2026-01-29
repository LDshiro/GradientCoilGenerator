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
            R_AP=params.get("R_AP"),
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
            "R_AP": st.number_input("R_AP", min_value=1e-4, value=0.3000, format="%.4f"),
            "NR": int(st.number_input("NR", min_value=2, value=32, step=1)),
            "NT": int(st.number_input("NT", min_value=3, value=48, step=1)),
            "z0": st.number_input("z0", value=0.0, format="%.4f"),
            "use_two_planes": st.checkbox("use_two_planes", value=True),
            "z_offset": st.number_input("z_offset", value=0.1400, format="%.4f"),
            "flip_second_normals": st.checkbox("flip_second_normals", value=False),
        }
    if surface_type == "plane_cart":
        return {
            "PLANE_HALF": st.number_input(
                "PLANE_HALF", min_value=1e-4, value=0.3000, format="%.4f"
            ),
            "NX": int(st.number_input("NX", min_value=2, value=20, step=1)),
            "NY": int(st.number_input("NY", min_value=2, value=20, step=1)),
            "R_AP": st.number_input("R_AP", min_value=1e-4, value=0.1, format="%.4f"),
            "z0": st.number_input("z0", value=0.0, format="%.4f"),
            "use_two_planes": st.checkbox("use_two_planes", value=True),
            "z_offset": st.number_input("z_offset", value=0.1500, format="%.4f"),
            "flip_second_normals": st.checkbox("flip_second_normals", value=False),
        }
    return {
        "R_CYL": st.number_input("R_CYL", min_value=1e-4, value=0.2, format="%.4f"),
        "H": st.number_input("H", min_value=1e-4, value=0.6000, format="%.4f"),
        "NZ": int(st.number_input("NZ", min_value=1, value=32, step=1)),
        "NTH": int(st.number_input("NTH", min_value=3, value=48, step=1)),
        "z_center": st.number_input("z_center", value=0.0, format="%.4f"),
        "dirichlet_z_edges": st.checkbox("dirichlet_z_edges", value=True),
    }


def main() -> None:
    import streamlit as st

    from gradientcoil.app.run_pipeline import run_optimization_pipeline
    from gradientcoil.io.results_npz import ContourField, extract_contour_fields
    from gradientcoil.physics.roi_sampling import (
        hammersley_sphere,
        sample_sphere_fibonacci,
        sample_sphere_sym_hammersley,
    )
    from gradientcoil.runs.listing import list_runs, load_run_config, load_run_npz, load_run_solver
    from gradientcoil.viz.mesh import periodic_theta_extend
    from gradientcoil.viz.plotly_setup import make_problem_setup_figure_plotly

    st.set_page_config(page_title="GradientCoil GUI", layout="wide")
    st.markdown("# Gradient Coil Designer")

    tab_conditions, tab_results = st.tabs(["条件", "結果"])

    with tab_conditions:
        run_clicked = False
        col_plot, col_params = st.columns([5, 1])
        with col_params:
            st.subheader("パラメータ")
            surface_type = st.selectbox(
                "surface",
                ["disk_polar", "plane_cart", "cylinder_unwrap"],
                index=1,
            )
            params = _param_panel(surface_type)

            st.markdown("### ROI")
            roi_radius = st.number_input("ROI radius", min_value=0.0, value=0.1, format="%.4f")
            roi_points_n = int(st.number_input("ROI points", min_value=0, value=128, step=1))
            roi_rotate = st.checkbox("ROI rotate", value=False)
            sym_axes = st.multiselect(
                "ROI symmetry axes",
                options=["x", "y", "z"],
                default=["x", "y", "z"],
            )
            roi_sampler = st.selectbox(
                "ROI sampler",
                options=["hammersley", "fibonacci", "sym_hammersley"],
                index=0,
                format_func={
                    "hammersley": "Hammersley (quasi-uniform)",
                    "fibonacci": "Fibonacci (quasi-uniform)",
                    "sym_hammersley": "Symmetric Hammersley (axis-symmetric)",
                }.get,
            )
            roi_dedup = st.checkbox("ROI dedup", value=False)
            roi_dedup_eps = st.number_input(
                "ROI dedup eps", min_value=0.0, value=1e-12, format="%.2e"
            )

            st.markdown("### Target")
            shim_max_order = int(st.number_input("shim_max_order", min_value=0, value=1, step=1))
            coeff_text = st.text_input("coeffs (NAME=VALUE, comma separated)", value="Y=0.02")
            scale_policy = st.selectbox("scale_policy", ["T_per_m", "native"], index=0)
            L_ref = st.text_input("L_ref", value="auto")

            st.markdown("### Regularizers")
            use_pitch = st.checkbox("use_pitch", value=False)
            delta_s = st.number_input("delta_S", min_value=0.0, value=0.0, format="%.4f")
            pitch_min = st.number_input("pitch_min", min_value=0.0, value=0.0, format="%.4f")
            use_tv_default = surface_type == "disk_polar"
            lambda_tv_default = 1.00e-6 if surface_type == "disk_polar" else 0.0
            use_tv = st.checkbox("use_tv", value=use_tv_default)
            lambda_tv = st.number_input(
                "lambda_tv", min_value=0.0, value=lambda_tv_default, format="%.2e"
            )
            use_power_default = surface_type in {"cylinder_unwrap", "disk_polar"}
            lambda_pwr_default = 2.00e-2 if surface_type == "cylinder_unwrap" else 4.00e-2
            use_power = st.checkbox("use_power", value=use_power_default)
            lambda_pwr = st.number_input(
                "lambda_pwr", min_value=0.0, value=lambda_pwr_default, format="%.2e"
            )
            r_sheet = st.number_input("r_sheet", min_value=0.0, value=0.000492, format="%.6f")
            default_scheme = "central" if surface_type == "plane_cart" else "forward"
            grad_scheme = st.selectbox(
                "gradient_scheme",
                options=["forward", "central", "edge"],
                index=0 if default_scheme == "forward" else 1,
            )
            emdm_mode = st.selectbox("emdm_mode", ["shared", "concat"], index=0)

            st.markdown("### Solver")
            max_iter = st.number_input("max_iter", min_value=1, value=100, step=1)
            solver_verbose = st.checkbox("verbose", value=True)

            st.markdown("### 可視化")
            normal_scale = st.number_input("normal_scale", min_value=0.0, value=0.02, format="%.4f")

            st.markdown("### まとめ")
            summary = {
                "surface": surface_type,
                **params,
                "roi_radius": roi_radius,
                "roi_points": roi_points_n,
                "roi_rotate": roi_rotate,
                "sym_axes": sym_axes,
                "roi_sampler": roi_sampler,
                "roi_dedup": roi_dedup,
                "roi_dedup_eps": roi_dedup_eps,
                "shim_max_order": shim_max_order,
                "coeffs": coeff_text,
                "scale_policy": scale_policy,
                "L_ref": L_ref,
                "use_pitch": use_pitch,
                "delta_S": delta_s,
                "pitch_min": pitch_min,
                "use_tv": use_tv,
                "lambda_tv": lambda_tv,
                "use_power": use_power,
                "lambda_pwr": lambda_pwr,
                "r_sheet": r_sheet,
                "gradient_scheme": grad_scheme,
                "emdm_mode": emdm_mode,
                "max_iter": max_iter,
                "solver_verbose": solver_verbose,
                "normal_scale": normal_scale,
            }
            st.json(summary)

            st.markdown("### Run Optimization")
            run_clicked = st.button("Run Optimization")

        with col_plot:
            st.subheader("3D 可視化")
            try:
                surfaces = _build_surfaces(surface_type, params)
                if roi_points_n > 0 and roi_radius > 0:
                    if roi_sampler == "fibonacci":
                        roi_points = sample_sphere_fibonacci(
                            roi_points_n,
                            roi_radius,
                            rotate=roi_rotate,
                            seed=0,
                        )
                    elif roi_sampler == "sym_hammersley":
                        roi_points = sample_sphere_sym_hammersley(
                            roi_points_n, roi_radius, sym_axes=tuple(sym_axes)
                        )
                    else:
                        roi_points = hammersley_sphere(
                            roi_points_n,
                            roi_radius,
                            rotate=roi_rotate,
                            seed=0,
                        )
                else:
                    roi_points = None
                fig = make_problem_setup_figure_plotly(
                    surfaces,
                    roi_points,
                    roi_radius if roi_radius > 0 else None,
                    show_boundary=True,
                    show_normals=True,
                    normals_stride=1,
                    centers_stride=1,
                    normal_scale=normal_scale,
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"可視化に失敗しました: {exc}")

        if run_clicked:
            status_box = st.empty()

            def _progress(stage: str, info: str) -> None:
                status_box.info(f"{stage}: {info}")

            coeffs: dict[str, float] = {}
            if coeff_text.strip():
                for item in coeff_text.split(","):
                    if not item.strip():
                        continue
                    if "=" in item:
                        name, val = item.split("=", 1)
                        coeffs[name.strip()] = float(val)

            gap = 2.0 * float(params.get("z_offset", 0.0)) if params.get("use_two_planes") else 0.0
            J_max = float(delta_s / pitch_min) if (use_pitch and pitch_min > 0.0) else 0.0
            config = {
                "out_dir": str(ROOT / "runs"),
                "surface_type": surface_type,
                "surface_params": params,
                "use_two_planes": bool(params.get("use_two_planes")),
                "gap": gap,
                "flip_second_normals": bool(params.get("flip_second_normals")),
                "roi": {
                    "roi_radius": float(roi_radius),
                    "roi_n": int(roi_points_n),
                    "roi_rotate": bool(roi_rotate),
                    "sym_axes": tuple(sym_axes),
                    "sampler": roi_sampler,
                    "roi_dedup": bool(roi_dedup),
                    "roi_dedup_eps": float(roi_dedup_eps),
                },
                "target": {
                    "shim_max_order": int(shim_max_order),
                    "coeffs": coeffs,
                    "scale_policy": scale_policy,
                    "L_ref": L_ref,
                },
                "spec": {
                    "use_tv": bool(use_tv),
                    "lambda_tv": float(lambda_tv),
                    "use_pitch": bool(use_pitch),
                    "J_max": float(J_max),
                    "use_power": bool(use_power),
                    "lambda_pwr": float(lambda_pwr),
                    "r_sheet": float(r_sheet),
                    "grad_scheme": grad_scheme,
                    "gradient_scheme_pitch": grad_scheme,
                    "gradient_scheme_tv": grad_scheme,
                    "gradient_scheme_power": grad_scheme,
                    "emdm_mode": emdm_mode,
                },
                "solver": {
                    "verbose": bool(solver_verbose),
                    "max_iter": int(max_iter),
                    "time_limit": None,
                },
            }
            with st.spinner("Running optimization..."):
                try:
                    run_dir, summary_out = run_optimization_pipeline(config, progress_cb=_progress)
                    st.success(f"Saved: {run_dir}")
                    st.session_state["selected_run_label"] = Path(run_dir).name
                    st.json(summary_out)
                except Exception as exc:
                    st.error(f"Optimization failed: {exc}")

    with tab_results:
        runs_dir = ROOT / "runs"
        runs_dir.mkdir(exist_ok=True)
        entries = list_runs(runs_dir)
        if not entries:
            st.info("runs/ に結果がありません。")
            return

        labels = [entry.label for entry in entries]
        selected_label = st.session_state.get("selected_run_label")
        default_index = labels.index(selected_label) if selected_label in labels else 0

        selected = st.selectbox("Run", labels, index=default_index)
        entry = entries[labels.index(selected)]

        config = load_run_config(entry)
        solver = load_run_solver(entry)

        with load_run_npz(entry) as npz:
            fields = extract_contour_fields(npz)

        st.caption(f"path: {entry.path}")
        if config:
            with st.expander("config.json", expanded=False):
                st.json(config)
        if solver:
            with st.expander("solver.json", expanded=False):
                st.json(solver)

        if not fields:
            st.warning("可視化できるデータが見つかりません。")
            return

        import matplotlib.pyplot as plt

        def _extend_field(field: ContourField) -> ContourField:
            X_ext, Y_ext, S_ext = periodic_theta_extend(field.X, field.Y, field.S)
            return ContourField(name=field.name, X=X_ext, Y=Y_ext, S=S_ext)

        surface_type = config.get("surface_type") or config.get("surface")
        if surface_type == "disk_polar":
            fields = [_extend_field(field) for field in fields]
        else:
            fields = [
                _extend_field(field) if field.name == "S_polar" else field for field in fields
            ]

        for field in fields:
            fig, ax = plt.subplots(figsize=(5, 4))
            cs = ax.contourf(field.X, field.Y, field.S, levels=30)
            ax.contour(field.X, field.Y, field.S, levels=30, colors="k", linewidths=0.6)
            fig.colorbar(cs, ax=ax)
            ax.set_title(field.name)
            ax.set_aspect("equal", adjustable="box")
            st.pyplot(fig)


if __name__ == "__main__":
    main()
