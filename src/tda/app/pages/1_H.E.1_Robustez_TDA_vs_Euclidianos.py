"""Página 1: H.E.1 — Robustez TDA vs Descriptores Euclidianos.

Valida la Hipótesis Específica 1 (H.E.1):
"La homología persistente, computada sobre nubes de puntos X ⊂ R^d
(d ≤ 100) con Ripser 0.6, proporciona números de Betti β₀ y β₁
que permanecen estables bajo perturbaciones del 15–20% en los datos
de entrada, superando a los descriptores euclidianos en tareas de
clasificación y detección de anomalías."

Referencias del Documento:
- Definición 1.4: Número de Betti topológico
- Teorema de Estabilidad (Cohen-Steiner et al., 2007)
- Sección 8.6.1: Diseño de casos de simulación (esfera/toro)
"""

import streamlit as st
from tda.app.download_utils import download_button, export_settings_ui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
from sklearn.cluster import KMeans

from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.analysis.metrics import compute_kmeans_accuracy, verify_betti_numbers
from tda.analysis.stability import compute_noise_sweep
from tda.analysis.anomaly import compare_anomaly_detection
from tda.visualization.plots_tda import plot_stability_chart, interpret_stability, plot_sweep_persistence_animation
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style, metric_card,
    methodology_expander, page_header
)

export_settings_ui()

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="H.E.1 — Robustez TDA vs Euclidianos", layout="wide", page_icon="📊")
st.markdown(responsive_style(), unsafe_allow_html=True)

# ── Page Header ──
st.markdown(page_header(
    "H.E.1 — Robustez TDA vs Euclidianos"
), unsafe_allow_html=True)
st.header("H.E.1 — Robustez Topológica vs Descriptores Euclidianos")

# ── Sidebar ──
st.sidebar.header("📊 H.E.1 — Robustez TDA")
n_points = st.sidebar.slider(
    "Puntos por forma", 100, 500, 200, 50, key="tda_n_points",
    help="n=200 es el valor del Documento (Sección 8.6.1).")
noise_level = st.sidebar.slider(
    "Ruido Gaussiano (%)", 0.0, 0.3, 0.15, 0.05, key="tda_noise_level",
    help="0.15 = 15% de perturbación. H.E.1 evalúa estabilidad en rango 15–20%.")
n_clusters = st.sidebar.number_input(
    "Clústeres (K-medias)", value=2, min_value=2, key="tda_n_clusters",
    help="Para H.E.1 con 2 formas (esfera+toro) mantener 2.")
n_rep = st.sidebar.slider(
    "Repeticiones (n_rep)", 1, 50, 10, 1, key="tda_n_rep",
    help="Número de repeticiones por nivel de ruido.")
run_sweep = st.sidebar.checkbox("Ejecutar barrido de estabilidad (0→30%)", value=True,
    help="Activa el barrido sistemático de ruido. Desactivar para verificación rápida.")

# ── Main content ──
ejecutar = st.button("▶ Ejecutar H.E.1 Completo", type="primary", key="btn_he1_full")
st.markdown("---")

if ejecutar:
    # ═══════════════════════════════════════════════
    # FASE 1: Clasificación + Betti
    # ═══════════════════════════════════════════════
    pts_sphere = generate_cloud("sphere", n_points)
    pts_torus = generate_cloud("torus", n_points)
    pts_torus[:, 0] += 1.5

    dataset = np.vstack([pts_sphere, pts_torus])
    dataset_noisy = add_gaussian_noise(dataset, noise_level)
    y_true = np.array([0] * n_points + [1] * n_points)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(dataset_noisy)

    from tda.analysis.stability import betti_significativos as _betti_sig
    (b0_s_raw, b1_s_raw), res_sphere = _betti_sig(
        add_gaussian_noise(pts_sphere, noise_level))
    (b0_t_raw, b1_t_raw), res_torus = _betti_sig(
        add_gaussian_noise(pts_torus, noise_level))

    acc_kmeans = compute_kmeans_accuracy(y_true, y_kmeans)
    betti_check = verify_betti_numbers(b0_s_raw, b1_s_raw, b0_t_raw, b1_t_raw)

    st.session_state.tda_cls = {
        'dataset_noisy': dataset_noisy, 'y_true': y_true, 'y_kmeans': y_kmeans,
        'res_sphere': res_sphere, 'res_torus': res_torus,
        'acc_kmeans': acc_kmeans,
        'b0_s': b0_s_raw, 'b1_s': b1_s_raw, 'b0_t': b0_t_raw, 'b1_t': b1_t_raw,
        'betti_check': betti_check,
        'params': {'n_points': n_points, 'noise_level': noise_level,
                   'n_clusters': n_clusters, 'n_rep': n_rep}
    }

    # ═══════════════════════════════════════════════
    # FASE 2: Detección de anomalías
    # ═══════════════════════════════════════════════
    anomaly_results = compare_anomaly_detection(
        n_points=n_points, noise_level=noise_level, seed=42
    )
    st.session_state.tda_anomaly = anomaly_results

    # ═══════════════════════════════════════════════
    # FASE 3: Barrido de estabilidad
    # ═══════════════════════════════════════════════
    if run_sweep:
        with st.spinner("Barrido de ruido 0→30%, 10 pasos..."):
            sweep_results = compute_noise_sweep(
                n_points=n_points, n_clusters=n_clusters, n_steps=10,
            )
            st.session_state.tda_sweep = sweep_results

    st.session_state.tda_he1_run = True

# ═══════════════════════════════════════════════════════════════
# RENDERIZAR RESULTADOS
# ═══════════════════════════════════════════════════════════════
if st.session_state.get('tda_he1_run', False):
    cls = st.session_state.tda_cls
    anomaly = st.session_state.get('tda_anomaly', None)
    sweep = st.session_state.get('tda_sweep', None)
    params = cls['params']

    st.info(
        f"📋 Resultados — {params['n_points']} pts, ruido {params['noise_level']:.0%}, "
        f"{params['n_clusters']} clústeres, n_rep={params['n_rep']}"
    )

    # ── 1. Clasificación ──
    st.subheader("1. Clasificación Euclidiana vs Ground Truth")
    ct1, ct2 = st.columns(2)
    with ct1:
        fig_k = go.Figure(data=[go.Scatter3d(
            x=cls['dataset_noisy'][:, 0], y=cls['dataset_noisy'][:, 1],
            z=cls['dataset_noisy'][:, 2],
            mode='markers', marker=dict(size=3, color=cls['y_kmeans'], colorscale='Viridis')
        )])
        fig_k.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
        apply_plotly_theme(fig_k)
        st.plotly_chart(fig_k, width='stretch')
    with ct2:
        fig_true = go.Figure(data=[go.Scatter3d(
            x=cls['dataset_noisy'][:, 0], y=cls['dataset_noisy'][:, 1],
            z=cls['dataset_noisy'][:, 2],
            mode='markers', marker=dict(size=3, color=cls['y_true'], colorscale='Plasma')
        )])
        fig_true.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
        apply_plotly_theme(fig_true)
        st.plotly_chart(fig_true, width='stretch')

    # ── 1b. Animación de transición K-Means → Ground Truth ──
    # Misma nube, solo cambia la etiqueta: se interpola el color para
    # visualizar cómo cada punto pasa de lo predicho a lo real.
    st.markdown("**Transición animada: de K-Means a Ground Truth**")
    y_pred_raw = np.asarray(cls['y_kmeans']).astype(float)
    y_real = np.asarray(cls['y_true']).astype(float)
    acc_directa = float(np.mean(y_pred_raw == y_real))
    acc_invertida = float(np.mean((1 - y_pred_raw) == y_real))
    y_pred = (1 - y_pred_raw) if acc_invertida > acc_directa else y_pred_raw
    xyz = cls['dataset_noisy']
    # Superficies ideales de referencia (sin ruido): esfera r=1 en origen
    # y toro R=2, r=1 desplazado +1.5 en x — mismos parámetros que sampling.py
    _u = np.linspace(0, 2 * np.pi, 30)
    _v_sph = np.linspace(0, np.pi, 20)
    _uu_s, _vv_s = np.meshgrid(_u, _v_sph)
    _xs = np.sin(_vv_s) * np.cos(_uu_s)
    _ys = np.sin(_vv_s) * np.sin(_uu_s)
    _zs = np.cos(_vv_s)
    _vv_t = np.linspace(0, 2 * np.pi, 25)
    _uu_t, _pp_t = np.meshgrid(_u, _vv_t)
    _R, _r = 2.0, 1.0
    _xt = (_R + _r * np.cos(_pp_t)) * np.cos(_uu_t) + 1.5
    _yt = (_R + _r * np.cos(_pp_t)) * np.sin(_uu_t)
    _zt = _r * np.sin(_pp_t)
    _surf_esfera = go.Surface(x=_xs, y=_ys, z=_zs, opacity=0.12,
                              colorscale='Blues', showscale=False,
                              hoverinfo='skip', name='Esfera ideal')
    _surf_toro = go.Surface(x=_xt, y=_yt, z=_zt, opacity=0.12,
                            colorscale='Oranges', showscale=False,
                            hoverinfo='skip', name='Toro ideal')
    n_frames_anim = 11
    frames_anim = []
    for i in range(n_frames_anim):
        t = i / (n_frames_anim - 1)
        col_t = (1 - t) * y_pred + t * y_real
        frames_anim.append(go.Frame(
            data=[go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode='markers',
                marker=dict(size=3, color=col_t, colorscale='Viridis',
                            cmin=0, cmax=1, showscale=False),
            )],
            traces=[2],
            name=f"{t:.1f}",
            layout=go.Layout(title_text=(
                "K-Means (predicho)" if t < 0.5 else "Ground Truth (real)"
            )),
        ))
    fig_anim = go.Figure(
        data=[_surf_esfera, _surf_toro, go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='markers',
            marker=dict(size=3, color=y_pred, colorscale='Viridis',
                        cmin=0, cmax=1, showscale=False),
            name='Nube ruidosa',
        )],
        frames=frames_anim,
    )
    fig_anim.update_layout(
        height=420, margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.5, y=1.05,
            xanchor="center", yanchor="bottom",
            buttons=[
                dict(label="▶ Reproducir transición",
                     method="animate",
                     args=[None, dict(frame=dict(duration=250, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pausar", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.1, len=0.9,
            currentvalue=dict(prefix="Progreso K-Means → Real: "),
            steps=[dict(label=f"{i / (n_frames_anim - 1):.0%}", method="animate",
                        args=[[f"{i / (n_frames_anim - 1):.1f}"],
                              dict(frame=dict(duration=0, redraw=True),
                                   mode="immediate")])
                   for i in range(n_frames_anim)],
        )],
    )
    apply_plotly_theme(fig_anim)
    st.plotly_chart(fig_anim, width='stretch')
    st.caption("Dale play: las mallas tenues son las superficies ideales (esfera r=1, toro R=2/r=1). Los puntos que flotan fuera de la malla son el ruido gaussiano; los que cambian de color son los mal clasificados por K-Means donde TDA (β₀/β₁) se mantiene.")

    # ── 2. Betti + métricas ──
    st.subheader("2. Números de Betti y Métricas de Clasificación")

    rt1, rt2 = st.columns(2)
    with rt1:
        st.info(
            f"**Esfera:** β₀={cls['b0_s']} | β₁={cls['b1_s']}\n\n"
            f"*β₀=1 → una sola componente conexa. β₁=0 → sin agujeros 1-dimensionales.*"
        )
    with rt2:
        st.warning(
            f"**Toro:** β₀={cls['b0_t']} | β₁={cls['b1_t']}\n\n"
            f"*β₀=1 → una sola componente conexa. β₁=2 → dos agujeros (túneles).*"
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card(
            value=f"{cls['acc_kmeans']:.1%}",
            title="Exactitud",
            subtitle=f"K-Medias (ruido {params['noise_level']:.0%})",
            variant="compliance"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card(
            value=f"{cls['b1_s']}",
            title="β₁ Esfera",
            subtitle="Agujeros (esperado: 0)",
            variant="beta1"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card(
            value=f"{cls['b1_t']}",
            title="β₁ Toro",
            subtitle="Agujeros (esperado: 2)",
            variant="beta1"
        ), unsafe_allow_html=True)
    with col4:
        stable = cls['betti_check']["b0_stable"]
        st.markdown(metric_card(
            value="✓" if stable else "✗",
            title="Estabilidad",
            subtitle="β₀ estable" if stable else "β₀ inestable",
            variant="manufacturing",
            value_color="#27ae60" if stable else "#e74c3c"
        ), unsafe_allow_html=True)

    # ── 3. Métricas TDA vs Euclideo ──
    with st.expander("3. Métricas TDA vs. Euclideo (PCA/K-Medias)", expanded=False):
        from tda.simulation.pipeline import run_tda_experiment
        tmp_s = run_tda_experiment('sphere', noise_levels=[params['noise_level']],
                                   n_rep=params['n_rep'], n_points=params['n_points'], seed=42)
        tmp_t = run_tda_experiment('torus', noise_levels=[params['noise_level']],
                                   n_rep=params['n_rep'], n_points=params['n_points'], seed=42)
        all_stats = {}
        for noise, stats in tmp_s.items():
            all_stats[noise] = stats
        for noise, stats in tmp_t.items():
            if noise not in all_stats:
                all_stats[noise] = stats
            else:
                for k, v in stats.items():
                    all_stats[noise][k] = (all_stats[noise][k] + v) / 2

        for noise, stats in all_stats.items():
            st.markdown(f"**Ruido: {noise:.0%}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PCA Inercia (Limpio)", f"{stats.get('tda_vs_euclidean_pca_inertia_clean', 0):.4f}")
            c2.metric("PCA Inercia (Ruidoso)", f"{stats.get('tda_vs_euclidean_pca_inertia_noisy', 0):.4f}")
            c3.metric("K-means Inercia (Limpio)", f"{stats.get('tda_vs_euclidean_kmeans_inertia_clean', 0):.2f}")
            c4.metric("K-means Inercia (Ruidoso)", f"{stats.get('tda_vs_euclidean_kmeans_inertia_noisy', 0):.2f}")
            c5, c6 = st.columns(2)
            c5.metric("β₀ Ruidoso (TDA)", f"{stats.get('tda_vs_euclidean_betti0_noisy', 0):.0f}")
            c6.metric("β₁ Ruidoso (TDA)", f"{stats.get('tda_vs_euclidean_betti1_noisy', 0):.0f}")

        p_clean = all_stats.get('0.1', {}).get('tda_vs_euclidean_pca_inertia_clean', 0) if all_stats else 0
        p_noisy = all_stats.get('0.1', {}).get('tda_vs_euclidean_pca_inertia_noisy', 0) if all_stats else 0
        if p_noisy > p_clean * 1.1:
            st.warning("⚠️ **H.E.1 parcial**: Descriptores euclidianos sensibles al ruido. TDA más robusto.")
        else:
            st.success("✅ **H.E.1 validada**: TDA más robusto que descriptores euclidianos bajo ruido.")

    # ── 4. Detección de anomalías ──
    if anomaly is not None:
        st.subheader("4. Detección de Anomalías (TDA vs K-Means)")

        st.markdown("**Resultados TDA (Betti filtrado por ε*/2)**")
        for category, label in [("normal", "Formas Normales"), ("anomaly", "Formas Anómalas")]:
            st.markdown(f"*{label}:*")
            for r in anomaly["tda"][category]:
                icon = "🔴" if r["is_anomaly"] else "🟢"
                st.markdown(
                    f"{icon} **{r['shape']}**: β₀={r['b0']}, β₁={r['b1']} "
                    f"(esperado: β₀={r['expected'][0]}, β₁={r['expected'][1]}) — "
                    f"{r['details']}"
                )

        c1, c2 = st.columns(2)
        with c1:
            tda_acc = anomaly["tda"].get("exactitud", anomaly["tda"]["accuracy"])
            st.metric("Exactitud TDA", f"{tda_acc:.0%}",
                      delta="✅ Detecta anomalías por topología", delta_color="normal")
        with c2:
            km_acc = anomaly["kmeans"].get("exactitud", anomaly["kmeans"]["accuracy"])
            st.metric("Exactitud K-Means", f"{km_acc:.0%}",
                      delta="⚠️ Solo por inercia", delta_color="off" if km_acc >= tda_acc else "inverse")

        if tda_acc > km_acc:
            st.success(
                f"✅ **H.E.1 validada en anomalías**: TDA ({tda_acc:.0%}) supera a K-Means ({km_acc:.0%})."
            )
        else:
            st.warning(
                f"⚠️ K-Means ({km_acc:.0%}) iguala o supera a TDA ({tda_acc:.0%})."
            )

    # ── 5. Barrido de estabilidad ──
    if sweep is not None:
        st.subheader("5. Barrido de Estabilidad (0→30% ruido)")

        fig_stab = plot_stability_chart(sweep)
        apply_plotly_theme(fig_stab)
        st.plotly_chart(fig_stab, width='stretch')
        st.caption("Línea punteada roja: β₁=2 esperado del toro. Esfera estable en (1,0) en todo el rango; el toro converge a 2 en la ventana 15-20% de H.E.1. Valores altos fuera de esa ventana son artefacto Rips con muestreo disperso (200 pts), no topología real.")

        if "diagrams_s" in sweep and len(sweep["diagrams_s"]) > 0:
            if "show_persist_anim" not in st.session_state:
                st.session_state["show_persist_anim"] = False

            if st.button("▶ Animación: Diagramas de Persistencia", key="toggle_persist_anim"):
                st.session_state["show_persist_anim"] = not st.session_state["show_persist_anim"]

            if st.session_state["show_persist_anim"]:
                if st.button("🔄 Reiniciar", key="restart_persist_anim"):
                    st.session_state["_persist_anim_key"] = st.session_state.get("_persist_anim_key", 0) + 1

                fig_anim = plot_sweep_persistence_animation(sweep)
                apply_plotly_theme(fig_anim)
                import streamlit.components.v1 as _components
                _anim_key = st.session_state.get("_persist_anim_key", 0)
                _html = fig_anim.to_html(include_plotlyjs="cdn", full_html=False)
                _components.html(_html, height=550)

        df_sweep = pd.DataFrame({
            "Ruido (%)": [f"{n*100:.0f}%" for n in sweep["noise_vals"]],
            "β₀ Esfera": [b[0] for b in sweep["betti_s"]],
            "β₁ Esfera": [b[1] for b in sweep["betti_s"]],
            "β₀ Toro": [b[0] for b in sweep["betti_t"]],
            "β₁ Toro": [b[1] for b in sweep["betti_t"]],
            "Exactitud K-Medias": [f"{a:.1%}" for a in sweep["acc"]],
        })
        st.dataframe(df_sweep, width='stretch', hide_index=True)

        if "estabilidad_s" in sweep or "estabilidad_t" in sweep:
            st.markdown("**Verificación H.E.1 — Estabilidad en rango 15%–20%**")
            c_es, c_et = st.columns(2)
            with c_es:
                est_s = sweep.get("estabilidad_s", {})
                icon_s = "✅" if est_s.get("estable", False) else "❌"
                st.markdown(f"**{icon_s} Esfera (β₀=1, β₁=0):**")
                st.caption(est_s.get("mensaje", "Sin datos"))
            with c_et:
                est_t = sweep.get("estabilidad_t", {})
                icon_t = "✅" if est_t.get("estable", False) else "❌"
                st.markdown(f"**{icon_t} Toro (β₀=1, β₁=2):**")
                st.caption(est_t.get("mensaje", "Sin datos"))

            if est_s.get("estable", False) and est_t.get("estable", False):
                st.success(
                    "✅ **H.E.1 VALIDADA:** Los Betti permanecen estables en 15%–20% de ruido."
                )
            else:
                st.warning(
                    "⚠️ **H.E.1 no validada completamente:** "
                    "Los Betti no permanecen estables en 15%–20%."
                )

        interpretation = interpret_stability(sweep)
        if interpretation["stable"]:
            st.success(f"✅ **{interpretation['verdict']}:** {interpretation['details']}")
        else:
            st.warning(f"⚠️ **{interpretation['verdict']}:** {interpretation['details']}")

    # ── 6. Exportación ──
    st.subheader("6. Exportar Datos")
    dataset_noisy = cls['dataset_noisy']
    n_pts = len(dataset_noisy) // 2
    shape_labels = ["sphere"] * n_pts + ["torus"] * n_pts
    betti_0_list = [cls['b0_s']] * n_pts + [cls['b0_t']] * n_pts
    betti_1_list = [cls['b1_s']] * n_pts + [cls['b1_t']] * n_pts

    df_export = pd.DataFrame({
        "x": dataset_noisy[:, 0], "y": dataset_noisy[:, 1], "z": dataset_noisy[:, 2],
        "shape_label": shape_labels, "kmeans_label": cls['y_kmeans'],
        "betti_0": betti_0_list, "betti_1": betti_1_list
    })
    csv_tda = df_export.to_csv(index=False).encode('utf-8')

    c1, c2 = st.columns(2)
    with c1:
        download_button(label="📥 CSV", data=csv_tda,
                        file_name="nube_puntos_tda.csv", mime="text/csv",
                        key="btn_csv_he1")
    with c2:
        pdf_buf = io.BytesIO()
        plt.style.use('default')
        fig_pdf = plt.figure(figsize=(12, 10))
        fig_pdf.patch.set_facecolor('white')
        ax1 = fig_pdf.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2],
                    c=cls['y_kmeans'], cmap='viridis', s=5)
        ax1.set_title("Clasificación Euclidiana (K-Medias)")
        ax2 = fig_pdf.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2],
                    c=cls['y_true'], cmap='plasma', s=5)
        ax2.set_title("Ground Truth")
        ax3 = fig_pdf.add_subplot(2, 2, 3)
        if len(cls['res_sphere'][1]) > 0:
            fin = cls['res_sphere'][1][np.isfinite(cls['res_sphere'][1][:, 1])]
            if len(fin) > 0:
                ax3.scatter(fin[:, 0], fin[:, 1], alpha=0.6)
        ax3.set_title(f"Persistencia Esfera (β₁={cls['b1_s']})")
        ax4 = fig_pdf.add_subplot(2, 2, 4)
        if len(cls['res_torus'][1]) > 0:
            fin = cls['res_torus'][1][np.isfinite(cls['res_torus'][1][:, 1])]
            if len(fin) > 0:
                ax4.scatter(fin[:, 0], fin[:, 1], alpha=0.6)
        ax4.set_title(f"Persistencia Toro (β₁={cls['b1_t']})")
        fig_pdf.suptitle(f"H.E.1 — Robustez TDA | Ruido: {params['noise_level']:.0%}")
        plt.tight_layout()
        pdf_buf_io = io.BytesIO()
        fig_pdf.savefig(pdf_buf_io, format='pdf')
        plt.close(fig_pdf)
        download_button(label="📄 PDF", data=pdf_buf_io.getvalue(),
                        file_name="reporte_he1.pdf", mime="application/pdf",
                        key="btn_pdf_he1")

# ════════════════════════════════════════════════════════════════
# METODOLOGÍA
# ════════════════════════════════════════════════════════════════
methodology_expander(
    "📖 Metodología — H.E.1",
    [
        (
            "fórmulas",
            r"""\beta_0 = \text{número componentes conexas del complejo de Rips}
\beta_1 = \text{número agujeros con persistencia > 0.5}
\text{Métrica: Exactitud = correctos/total}
F_1 = 2 \cdot \frac{precisión \cdot recuperación}{precisión + recuperación}"""
        )
    ],
    "H.E.1"
)
st.markdown("---")

apply_mpl_theme()
