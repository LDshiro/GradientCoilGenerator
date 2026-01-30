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
    import numpy as np
    import pandas as pd
    import streamlit as st

    from gradientcoil.app.run_pipeline import run_optimization_pipeline
    from gradientcoil.io.results_npz import ContourField, extract_contour_fields
    from gradientcoil.optimize.socp_bz import SocpBzSpec, estimate_socp_bz_problem_size
    from gradientcoil.physics.roi_sampling import (
        dedup_points_with_weights,
        hammersley_sphere,
        sample_sphere_fibonacci,
        sample_sphere_sym_hammersley,
        symmetrize_points,
    )
    from gradientcoil.postprocess.contours_resample import (
        ContourFilterConfig,
        ContourLevelConfig,
        PlotConfig,
        PostprocessConfig,
        ResampleConfig,
        TriangulationConfig,
        process_run,
        save_resampled_npz,
    )
    from gradientcoil.runs.listing import list_runs, load_run_config, load_run_npz, load_run_solver
    from gradientcoil.targets.shim_basis import list_shim_terms
    from gradientcoil.targets.target_bz_source import ShimBasisTargetBz
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
            target_source = st.radio(
                "target_source",
                ["basis", "measured"],
                format_func={
                    "basis": "Shim basis (NMR)",
                    "measured": "Measured (CSV/NPZ) [future]",
                }.get,
                horizontal=True,
            )
            target_display = st.radio(
                "target_display",
                ["plane", "sphere"],
                format_func={
                    "plane": "Plane contour (Z=0)",
                    "sphere": "Sphere surface (ROI)",
                }.get,
                horizontal=True,
            )
            max_order = int(st.slider("max_order", min_value=0, max_value=3, value=1, step=1))
            L_ref_auto = st.checkbox("L_ref auto", value=True)
            if L_ref_auto:
                L_ref = "auto"
                st.number_input(
                    "L_ref (auto)", min_value=0.0, value=0.0, format="%.4f", disabled=True
                )
            else:
                L_ref = st.number_input("L_ref", min_value=0.0, value=roi_radius, format="%.4f")
            scale_policy = st.selectbox("scale_policy", ["T_per_m"], index=0)
            coeffs: dict[str, float] = {}
            measured_path = ""
            if target_source == "basis":
                terms = list_shim_terms(max_order)
                base_coeffs = st.session_state.get("coeff_map", {})
                rows = [
                    {
                        "name": term.name,
                        "order": term.order,
                        "coeff": base_coeffs.get(term.name, 0.0),
                    }
                    for term in terms
                ]
                editor_key = f"coeff_editor_{max_order}"
                data_key = f"{editor_key}_data"
                if data_key not in st.session_state:
                    st.session_state[data_key] = pd.DataFrame(rows)
                if st.button("Preset: Y-gradient"):
                    df = st.session_state[data_key].copy()
                    df["coeff"] = 0.0
                    if "Y" in df["name"].values:
                        df.loc[df["name"] == "Y", "coeff"] = 0.02
                    st.session_state[data_key] = df
                coeff_df = st.data_editor(
                    st.session_state[data_key],
                    key=editor_key,
                    disabled=["name", "order"],
                    use_container_width=True,
                )
                st.caption("order 列は読み取り専用です。次数は上の max_order で変更してください。")
                st.session_state[data_key] = coeff_df
                coeffs = {
                    row["name"]: float(row["coeff"]) for row in coeff_df.to_dict(orient="records")
                }
                st.session_state["coeff_map"] = coeffs
            else:
                measured_path = st.text_input("measured_path", value="")
                st.warning("Measured target is not implemented yet.")

            st.markdown("### Regularizers")
            use_pitch = st.checkbox("use_pitch", value=False)
            delta_s = st.number_input("delta_S", min_value=0.0, value=0.0020, format="%.4f")
            pitch_min = st.number_input("pitch_min", min_value=0.0, value=0.0001, format="%.4f")
            use_tv_default = surface_type == "disk_polar"
            lambda_tv_default = 5.00e-8
            use_tv = st.checkbox("use_tv", value=use_tv_default)
            lambda_tv = st.number_input(
                "lambda_tv", min_value=0.0, value=lambda_tv_default, format="%.2e"
            )
            use_power_default = surface_type in {"cylinder_unwrap", "disk_polar"}
            lambda_pwr_default = 3.00e-2
            use_power = st.checkbox("use_power", value=use_power_default)
            lambda_pwr = st.number_input(
                "lambda_pwr", min_value=0.0, value=lambda_pwr_default, format="%.2e"
            )
            use_tgv = st.checkbox("use_tgv", value=False)
            alpha1_tgv = st.number_input("alpha1_tgv", min_value=0.0, value=1.0e-6, format="%.3e")
            alpha0_tgv = st.number_input("alpha0_tgv", min_value=0.0, value=1.0e-6, format="%.3e")
            tgv_area_weights = st.checkbox("tgv_area_weights", value=True)
            with st.expander("Curvature regularizer (grad-grad)", expanded=False):
                st.caption("隣接勾配差分（曲がり）を抑える。R1はSOCP、ENは二次項。")
                use_curv_r1 = st.checkbox("use_curv_r1", value=False)
                lambda_curv_r1 = st.number_input(
                    "lambda_curv_r1", min_value=0.0, value=0.0, format="%.2e"
                )
                use_curv_en = st.checkbox("use_curv_en", value=False)
                lambda_curv_en = st.number_input(
                    "lambda_curv_en", min_value=0.0, value=0.0, format="%.2e"
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
                "target_source": target_source,
                "target_display": target_display,
                "max_order": max_order,
                "coeffs": coeffs,
                "scale_policy": scale_policy,
                "L_ref": L_ref,
                "measured_path": measured_path,
                "use_pitch": use_pitch,
                "delta_S": delta_s,
                "pitch_min": pitch_min,
                "use_tv": use_tv,
                "lambda_tv": lambda_tv,
                "use_power": use_power,
                "lambda_pwr": lambda_pwr,
                "use_tgv": use_tgv,
                "alpha1_tgv": alpha1_tgv,
                "alpha0_tgv": alpha0_tgv,
                "tgv_area_weights": tgv_area_weights,
                "use_curv_r1": use_curv_r1,
                "lambda_curv_r1": lambda_curv_r1,
                "use_curv_en": use_curv_en,
                "lambda_curv_en": lambda_curv_en,
                "r_sheet": r_sheet,
                "gradient_scheme": grad_scheme,
                "gradient_scheme_curv": grad_scheme,
                "gradient_scheme_tgv": grad_scheme,
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

                if roi_radius > 0 and target_source == "basis":
                    L_ref_val = roi_radius if L_ref == "auto" else float(L_ref)
                    target_spec = ShimBasisTargetBz(
                        max_order=max_order,
                        coeffs=coeffs,
                        L_ref=float(L_ref_val),
                        scale_policy=scale_policy,
                    )
                    grid_n = 60
                    if target_display == "plane":
                        x = np.linspace(-roi_radius, roi_radius, grid_n)
                        y = np.linspace(-roi_radius, roi_radius, grid_n)
                        Xg, Yg = np.meshgrid(x, y, indexing="xy")
                        mask = (Xg**2 + Yg**2) <= (roi_radius**2)
                        Zg = np.full_like(Xg, np.nan, dtype=float)
                        if np.any(mask):
                            pts = np.stack([Xg[mask], Yg[mask], np.zeros_like(Xg[mask])], axis=1)
                            Zg[mask] = target_spec.evaluate(pts)
                    else:
                        theta = np.linspace(0.0, np.pi, grid_n)
                        phi = np.linspace(0.0, 2.0 * np.pi, grid_n + 1)
                        Phi, Theta = np.meshgrid(phi, theta, indexing="xy")
                        Xg = roi_radius * np.sin(Theta) * np.cos(Phi)
                        Yg = roi_radius * np.sin(Theta) * np.sin(Phi)
                        Zg = roi_radius * np.cos(Theta)
                        pts = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)
                        vals = target_spec.evaluate(pts).reshape(Xg.shape)
                    import plotly.graph_objects as go

                    contours = {"z": {"show": True, "color": "black", "width": 1}}
                    opacity = 0.7
                    if target_display == "sphere":
                        contours = None
                        opacity = 1.0
                    fig.add_trace(
                        go.Surface(
                            x=Xg,
                            y=Yg,
                            z=Zg,
                            surfacecolor=Zg if target_display == "plane" else vals,
                            opacity=opacity,
                            colorscale="Viridis",
                            showscale=False,
                            name="target_bz",
                            contours=contours,
                        )
                    )
                st.plotly_chart(fig, use_container_width=True)

                if roi_points is not None:
                    roi_points_raw = symmetrize_points(roi_points, axes=tuple(sym_axes))
                    if roi_dedup:
                        roi_points_used, _ = dedup_points_with_weights(
                            roi_points_raw, roi_dedup_eps
                        )
                    else:
                        roi_points_used = roi_points_raw

                    J_max = float(delta_s / pitch_min) if (use_pitch and pitch_min > 0.0) else 0.0
                    size_spec = SocpBzSpec(
                        use_tv=bool(use_tv),
                        lambda_tv=float(lambda_tv),
                        use_pitch=bool(use_pitch),
                        J_max=float(J_max),
                        use_power=bool(use_power),
                        lambda_pwr=float(lambda_pwr),
                        r_sheet=float(r_sheet),
                        use_tgv=bool(use_tgv),
                        alpha1_tgv=float(alpha1_tgv),
                        alpha0_tgv=float(alpha0_tgv),
                        tgv_area_weights=bool(tgv_area_weights),
                        use_curv_r1=bool(use_curv_r1),
                        lambda_curv_r1=float(lambda_curv_r1),
                        use_curv_en=bool(use_curv_en),
                        lambda_curv_en=float(lambda_curv_en),
                        gradient_scheme_curv=grad_scheme,
                        gradient_scheme_tgv=grad_scheme,
                        gradient_scheme_pitch=grad_scheme,
                        gradient_scheme_tv=grad_scheme,
                        gradient_scheme_power=grad_scheme,
                        emdm_mode=emdm_mode,
                    )
                    size = estimate_socp_bz_problem_size(roi_points_used, surfaces, size_spec)
                    active_terms = [
                        name
                        for name, enabled in {
                            "pitch": use_pitch,
                            "tv": use_tv,
                            "power": use_power,
                            "tgv": use_tgv,
                            "curv_r1": use_curv_r1,
                            "curv_en": use_curv_en,
                        }.items()
                        if enabled
                    ]
                    st.markdown("### 最適化問題サイズ")
                    st.caption(f"有効項目: {', '.join(active_terms) if active_terms else 'なし'}")
                    st.markdown(
                        f"**変数数:** {size.n_variables} "
                        f"(非負制約付き: {size.n_nonneg}) / "
                        f"**制約数:** {size.n_constraints}"
                    )
                    st.json({"variables": size.variables, "constraints": size.constraints})
            except Exception as exc:
                st.error(f"可視化に失敗しました: {exc}")

        if run_clicked:
            status_box = st.empty()

            def _progress(stage: str, info: str) -> None:
                status_box.info(f"{stage}: {info}")

            if target_source == "measured":
                st.error("Measured target is not implemented yet.")
                return

            gap = 2.0 * float(params.get("z_offset", 0.0)) if params.get("use_two_planes") else 0.0
            J_max = float(delta_s / pitch_min) if (use_pitch and pitch_min > 0.0) else 0.0
            if use_tgv and (alpha1_tgv <= 0.0 or alpha0_tgv <= 0.0):
                st.error("TGV requires alpha1_tgv > 0 and alpha0_tgv > 0.")
                return
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
                    "source_kind": target_source,
                    "max_order": int(max_order),
                    "coeffs": coeffs,
                    "scale_policy": scale_policy,
                    "L_ref": L_ref,
                    "measured_path": measured_path,
                },
                "spec": {
                    "use_tv": bool(use_tv),
                    "lambda_tv": float(lambda_tv),
                    "use_pitch": bool(use_pitch),
                    "J_max": float(J_max),
                    "use_power": bool(use_power),
                    "lambda_pwr": float(lambda_pwr),
                    "use_tgv": bool(use_tgv),
                    "alpha1_tgv": float(alpha1_tgv),
                    "alpha0_tgv": float(alpha0_tgv),
                    "tgv_area_weights": bool(tgv_area_weights),
                    "use_curv_r1": bool(use_curv_r1),
                    "lambda_curv_r1": float(lambda_curv_r1),
                    "use_curv_en": bool(use_curv_en),
                    "lambda_curv_en": float(lambda_curv_en),
                    "r_sheet": float(r_sheet),
                    "grad_scheme": grad_scheme,
                    "gradient_scheme_curv": grad_scheme,
                    "gradient_scheme_tgv": grad_scheme,
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
        run_npz_path = entry.path / "run.npz" if entry.kind == "folder" else entry.path

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

        st.subheader("PostProcess")
        with st.expander("PostProcess settings", expanded=False):
            st.caption(f"run_npz: {run_npz_path}")
            pp_config_json = st.text_input("config_json (optional)", value="")
            pp_col1, pp_col2 = st.columns(2)
            with pp_col1:
                pp_level_mode = st.selectbox("level_mode", ["n_levels", "delta_s"], index=0)
                pp_n_levels = int(st.number_input("n_levels_total", min_value=2, value=26, step=2))
                pp_delta_s = st.number_input("delta_s", min_value=0.0, value=1.0, format="%.4f")
                pp_ds_segment = st.number_input(
                    "ds_segment", min_value=0.0, value=0.005, format="%.4f"
                )
                pp_extraction = st.selectbox("extraction_method", ["auto", "grid", "tri"], index=0)
            with pp_col2:
                pp_min_vertices = int(
                    st.number_input("min_vertices", min_value=3, value=12, step=1)
                )
                pp_area_eps = st.number_input("area_eps", min_value=0.0, value=1e-6, format="%.2e")
                pp_close_tol = st.number_input(
                    "close_tol", min_value=0.0, value=1e-10, format="%.2e"
                )
                pp_tri_refine = st.checkbox("tri_refine", value=True)
                pp_flat_ratio = st.number_input(
                    "flat_ratio", min_value=0.0, value=0.02, format="%.3f"
                )
                pp_subdiv = int(st.number_input("subdiv", min_value=0, value=2, step=1))
            pp_show_filled = st.checkbox("show_filled (debug)", value=False)
            pp_show_level_lines = st.checkbox("show_level_lines (debug)", value=False)
            pp_show_loops = st.checkbox("show_extracted_loops (debug)", value=False)
            default_out = runs_dir / f"contours_resampled_{entry.label}.npz"
            pp_out_npz = st.text_input("out_npz", value=str(default_out))

        postprocess_key = f"postprocess_result_{entry.label}"
        if st.button("PostProcess"):
            post_cfg = PostprocessConfig(
                levels=ContourLevelConfig(
                    mode=pp_level_mode,
                    n_levels_total=pp_n_levels,
                    delta_s=pp_delta_s,
                ),
                filters=ContourFilterConfig(
                    min_vertices=pp_min_vertices,
                    area_eps=pp_area_eps,
                    close_tol=pp_close_tol,
                ),
                triangulation=TriangulationConfig(
                    use_refine=pp_tri_refine,
                    flat_ratio=pp_flat_ratio,
                    subdiv=pp_subdiv,
                ),
                plots=PlotConfig(
                    show_filled=pp_show_filled,
                    show_level_lines=pp_show_level_lines,
                    show_extracted_loops=pp_show_loops,
                ),
                resample=ResampleConfig(ds_segment=pp_ds_segment),
                extraction_method=pp_extraction,
            )
            try:
                config_json_path = pp_config_json.strip() or None
                result, run_cfg = process_run(run_npz_path, config_json_path, post_cfg)
                save_resampled_npz(
                    pp_out_npz,
                    result=result,
                    run_cfg=run_cfg,
                    run_npz_path=run_npz_path,
                    cfg=post_cfg,
                )
                st.session_state[postprocess_key] = result
                st.success(f"PostProcess 完了: {pp_out_npz}")
                st.caption(
                    f"loops={len(result.loops_resampled)} segments={result.segments['P'].shape[0]}"
                )
            except Exception as exc:
                st.error(f"PostProcess 失敗: {exc}")

        import io

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

        post_result = st.session_state.get(postprocess_key)

        def _surface_index_from_name(name: str) -> int | None:
            if name == "S_grid":
                return 0
            if name.startswith("S_grid_"):
                try:
                    return int(name.split("_")[-1])
                except ValueError:
                    return None
            return None

        for field in fields:
            idx = _surface_index_from_name(field.name)
            col_left, col_right = st.columns(2)
            with col_left:
                fig, ax = plt.subplots(figsize=(6, 4), dpi=360)
                levels = 60
                cs = ax.contourf(field.X, field.Y, field.S, levels=levels)
                ax.contour(field.X, field.Y, field.S, levels=levels, colors="k", linewidths=0.6)
                fig.colorbar(cs, ax=ax)
                ax.set_title(field.name)
                ax.set_aspect("equal", adjustable="box")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=360, bbox_inches="tight")
                buf.seek(0)
                st.image(buf, width=1000)
                plt.close(fig)

            with col_right:
                if post_result is None or idx is None:
                    st.info("PostProcess 未実行、または対象外フィールドです。")
                    continue
                loops = [loop for loop in post_result.loops_resampled if loop.get("surface") == idx]
                fig, ax = plt.subplots(figsize=(4, 4), dpi=360)
                for loop in loops:
                    pts = np.asarray(loop["points"], float)
                    if pts.size == 0:
                        continue
                    ax.plot(pts[:, 0], pts[:, 1], color="black", lw=0.7)
                ax.set_title(f"PostProcess (surface {idx})")
                ax.set_aspect("equal", adjustable="box")
                ax.set_box_aspect(1)
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.grid(True, alpha=0.3)
                if loops:
                    all_pts = np.vstack([np.asarray(loop["points"], float) for loop in loops])
                    xmin, ymin = np.min(all_pts, axis=0)
                    xmax, ymax = np.max(all_pts, axis=0)
                    pad_x = 0.05 * max(1e-9, xmax - xmin)
                    pad_y = 0.05 * max(1e-9, ymax - ymin)
                    ax.set_xlim(xmin - pad_x, xmax + pad_x)
                    ax.set_ylim(ymin - pad_y, ymax + pad_y)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=360, bbox_inches="tight")
                buf.seek(0)
                st.image(buf, width=1000)
                plt.close(fig)


if __name__ == "__main__":
    main()
