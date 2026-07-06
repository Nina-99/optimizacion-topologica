"""Página 1: TDA vs K-Medias (H.E.1).

Prueba de robustez topológica: homología persistente vs clasificación
euclidiana (K-Medias) bajo ruido gaussiano controlado.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
from sklearn.cluster import KMeans
from ripser import ripser

from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.analysis.metrics import compute_kmeans_accuracy, verify_betti_numbers
from tda.analysis.stability import compute_noise_sweep
from tda.visualization.plots_tda import plot_stability_chart, interpret_stability, plot_sweep_persistence_animation
from tda.app.theme import apply_mpl_theme, apply_plotly_theme, responsive_style

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="TDA vs K-Medias", layout="wide", page_icon="📊")

st.markdown(responsive_style(), unsafe_allow_html=True)
st.header("Robustez Topológica vs Distancia Euclidiana (H.E.1)")

# ── Sidebar ──
st.sidebar.header("📊 TDA vs K-Medias")
n_points = st.sidebar.slider("Puntos por forma", 100, 500, 250, 50, key="tda_n_points",
    help="Cantidad de puntos generados por cada forma geométrica (esfera y toro). Más puntos → mejor representación topológica, pero cómputo más lento.")
noise_level = st.sidebar.slider("Ruido Gaussiano (%)", 0.0, 0.3, 0.15, 0.05, key="tda_noise_level",
    help="Desviación estándar del ruido gaussiano aplicado a las coordenadas de los puntos. 0.15 = 15% de perturbación. A mayor ruido, más se degrada K-Medias mientras TDA se mantiene estable.")
n_clusters = st.sidebar.number_input("Clústeres (K-medias)", value=2, min_value=2, key="tda_n_clusters",
    help="Número de clústeres para K-Medias. Para H.E.1 con 2 formas (esfera+toro) mantener 2. Cambiar solo para probar comportamientos de sobre-segmentación.")

# ── Main content ──
ejecutar_tda = st.button("▶ Ejecutar Comparación", type="primary", key="btn_tda")
st.markdown("---")

if ejecutar_tda:
    with st.spinner("Procesando geometrías..."):
        pts_sphere = generate_cloud("sphere", n_points)
        pts_torus = generate_cloud("torus", n_points)
        pts_torus[:, 0] += 1.5

        dataset = np.vstack([pts_sphere, pts_torus])
        dataset_noisy = add_gaussian_noise(dataset, noise_level)
        y_true = np.array([0]*n_points + [1]*n_points)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(dataset_noisy)

        res_sphere = ripser(add_gaussian_noise(pts_sphere, noise_level), maxdim=1)['dgms']
        res_torus = ripser(add_gaussian_noise(pts_torus, noise_level), maxdim=1)['dgms']

        # Guardar en session state
        st.session_state.tda_dataset_noisy = dataset_noisy
        st.session_state.tda_y_kmeans = y_kmeans
        st.session_state.tda_y_true = y_true
        st.session_state.tda_res_sphere = res_sphere
        st.session_state.tda_res_torus = res_torus
        # Guardar parámetros usados para persistencia al navegar
        st.session_state.tda_params_used = {
            "n_points": n_points,
            "noise_level": noise_level,
            "n_clusters": n_clusters,
        }
        st.session_state.tda_run = True

# Renderizar resultados si existen en session state
if st.session_state.get('tda_run', False):
    dataset_noisy = st.session_state.tda_dataset_noisy
    y_kmeans = st.session_state.tda_y_kmeans
    y_true = st.session_state.tda_y_true
    res_sphere = st.session_state.tda_res_sphere
    res_torus = st.session_state.tda_res_torus

    # Mostrar banner de parámetros usados (persistencia entre páginas)
    params_used = st.session_state.get("tda_params_used", {})
    if params_used:
        st.info(
            f"📋 Mostrando resultados previos — "
            f"Parámetros usados: {params_used['n_points']} pts por forma, "
            f"ruido {params_used['noise_level']:.0%}, "
            f"{params_used['n_clusters']} clústeres. "
            f"Cambiá los parámetros en el sidebar y ejecutá de nuevo para actualizar."
        )
    else:
        st.info("📋 Mostrando resultados previos. Cambiá los parámetros y ejecutá de nuevo.")

    ct1, ct2 = st.columns(2)
    with ct1:
        st.subheader("Clasificación Euclidiana (K-Medias)")
        st.markdown("*Falla al separar geometrías superpuestas*")
        fig_k = go.Figure(data=[go.Scatter3d(
            x=dataset_noisy[:, 0], y=dataset_noisy[:, 1], z=dataset_noisy[:, 2],
            mode='markers', marker=dict(size=3, color=y_kmeans, colorscale='Viridis')
        )])
        fig_k.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
        apply_plotly_theme(fig_k)
        st.plotly_chart(fig_k, use_container_width=True)

    with ct2:
        st.subheader("Clasificación Real (Ground Truth)")
        st.markdown("*Separación perfecta por invariantes topológicos*")
        fig_true = go.Figure(data=[go.Scatter3d(
            x=dataset_noisy[:, 0], y=dataset_noisy[:, 1], z=dataset_noisy[:, 2],
            mode='markers', marker=dict(size=3, color=y_true, colorscale='Plasma')
        )])
        fig_true.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
        apply_plotly_theme(fig_true)
        st.plotly_chart(fig_true, use_container_width=True)

    rt1, rt2 = st.columns(2)
    rt1.info(f"**Esfera:** $\beta_0 = {len(res_sphere[0])}$ | $\beta_1 = {len(res_sphere[1])}$")
    rt2.warning(f"**Toro:** $\beta_0 = {len(res_torus[0])}$ | $\beta_1 = {len(res_torus[1])}$")

    # ── Métrica de clasificación K-Medias ──
    acc_kmeans = compute_kmeans_accuracy(
        st.session_state.tda_y_true,
        st.session_state.tda_y_kmeans
    )
    b0_s = len(st.session_state.tda_res_sphere[0])
    b1_s = len(st.session_state.tda_res_sphere[1])
    b0_t = len(st.session_state.tda_res_torus[0])
    b1_t = len(st.session_state.tda_res_torus[1])
    betti_check = verify_betti_numbers(b0_s, b1_s, b0_t, b1_t)

    col_acc1, col_acc2, col_acc3 = st.columns(3)
    col_acc1.metric("Accuracy K-Medias", f"{acc_kmeans:.1%}",
                    delta=f"Ruido: {noise_level:.0%}",
                    delta_color="off")
    col_acc2.metric("β₁ Esperado (Esfera/Toro)", "0 / 2",
                    delta=betti_check["message"])
    col_acc3.metric("β₀ Esperado (Esfera/Toro)", "1 / 1",
                    delta="✓ Estable" if betti_check["b0_stable"] else "✗ Inestable")

    # Exportación
    st.subheader("Exportar Datos y Reportes de TDA")

    # 1. Crear CSV
    n_pts = len(dataset_noisy) // 2
    shape_labels = ["sphere"] * n_pts + ["torus"] * n_pts
    betti_0_list = [len(res_sphere[0])] * n_pts + [len(res_torus[0])] * n_pts
    betti_1_list = [len(res_sphere[1])] * n_pts + [len(res_torus[1])] * n_pts

    df_tda_export = pd.DataFrame({
        "x": dataset_noisy[:, 0],
        "y": dataset_noisy[:, 1],
        "z": dataset_noisy[:, 2],
        "shape_label": shape_labels,
        "kmeans_label": y_kmeans,
        "betti_0": betti_0_list,
        "betti_1": betti_1_list
    })
    csv_tda = df_tda_export.to_csv(index=False).encode('utf-8')

    # 2. Crear PDF (siempre con fondo blanco para exportación)
    pdf_buf_tda = io.BytesIO()

    # Restablecer estilo a default para exportación (fondo blanco)
    plt.style.use('default')
    fig_pdf_tda = plt.figure(figsize=(12, 10))
    fig_pdf_tda.patch.set_facecolor('white')

    # K-Means 3D
    ax1 = fig_pdf_tda.add_subplot(2, 2, 1, projection='3d')
    ax1.set_facecolor('white')
    ax1.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2], c=y_kmeans, cmap='viridis', s=5)
    ax1.set_title("Clasificación Euclidiana (K-Medias)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Ground Truth 3D
    ax2 = fig_pdf_tda.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('white')
    ax2.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2], c=y_true, cmap='plasma', s=5)
    ax2.set_title("Clasificación Real (Ground Truth)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Sphere persistence diagram
    ax3 = fig_pdf_tda.add_subplot(2, 2, 3)
    ax3.set_facecolor('white')
    if len(res_sphere[0]) > 0:
        finite_h0 = res_sphere[0][np.isfinite(res_sphere[0][:, 1])] if res_sphere[0].ndim > 1 else res_sphere[0]
        if len(finite_h0) > 0:
            ax3.scatter(finite_h0[:, 0], finite_h0[:, 1], label="H0 (Componentes conexas)", color="blue", alpha=0.6)
    if len(res_sphere[1]) > 0:
        finite_h1 = res_sphere[1][np.isfinite(res_sphere[1][:, 1])] if res_sphere[1].ndim > 1 else res_sphere[1]
        if len(finite_h1) > 0:
            ax3.scatter(finite_h1[:, 0], finite_h1[:, 1], label="H1 (Cavidades/Túneles)", color="orange", marker="^", alpha=0.8)

    # diagonal
    max_val_s = 1.0
    for dgm in res_sphere:
        if len(dgm) > 0:
            finite_vals = dgm[np.isfinite(dgm)]
            if len(finite_vals) > 0:
                max_val_s = max(max_val_s, np.max(finite_vals))
    ax3.plot([0, max_val_s * 1.1], [0, max_val_s * 1.1], 'k--', alpha=0.5)
    ax3.set_xlabel("Nacimiento (Birth)")
    ax3.set_ylabel("Muerte (Death)")
    ax3.set_title(f"Diagrama de Persistencia - Esfera\n(b0={len(res_sphere[0])}, b1={len(res_sphere[1])})")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Torus persistence diagram
    ax4 = fig_pdf_tda.add_subplot(2, 2, 4)
    ax4.set_facecolor('white')
    if len(res_torus[0]) > 0:
        finite_h0_t = res_torus[0][np.isfinite(res_torus[0][:, 1])] if res_torus[0].ndim > 1 else res_torus[0]
        if len(finite_h0_t) > 0:
            ax4.scatter(finite_h0_t[:, 0], finite_h0_t[:, 1], label="H0 (Componentes conexas)", color="blue", alpha=0.6)
    if len(res_torus[1]) > 0:
        finite_h1_t = res_torus[1][np.isfinite(res_torus[1][:, 1])] if res_torus[1].ndim > 1 else res_torus[1]
        if len(finite_h1_t) > 0:
            ax4.scatter(finite_h1_t[:, 0], finite_h1_t[:, 1], label="H1 (Cavidades/Túneles)", color="orange", marker="^", alpha=0.8)

    # diagonal
    max_val_t = 1.0
    for dgm in res_torus:
        if len(dgm) > 0:
            finite_vals = dgm[np.isfinite(dgm)]
            if len(finite_vals) > 0:
                max_val_t = max(max_val_t, np.max(finite_vals))
    ax4.plot([0, max_val_t * 1.1], [0, max_val_t * 1.1], 'k--', alpha=0.5)
    ax4.set_xlabel("Nacimiento (Birth)")
    ax4.set_ylabel("Muerte (Death)")
    ax4.set_title(f"Diagrama de Persistencia - Toro\n(b0={len(res_torus[0])}, b1={len(res_torus[1])})")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)

    fig_pdf_tda.suptitle(f"Análisis Robustez Topológica (TDA vs K-Medias)\nRuido: {noise_level:.2f} | K-Medias Clusters: {n_clusters}", fontsize=14)
    plt.tight_layout()
    fig_pdf_tda.savefig(pdf_buf_tda, format='pdf')
    # También exportar como PNG (misma figura, fondo blanco)
    png_buf_tda = io.BytesIO()
    fig_pdf_tda.savefig(png_buf_tda, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_pdf_tda)

    pdf_bytes_tda = pdf_buf_tda.getvalue()
    png_bytes_tda = png_buf_tda.getvalue()
    pdf_buf_tda.close()
    png_buf_tda.close()

    # LaTeX  tabla de métricas
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Métricas de Análisis TDA vs K-Medias}",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "\\textbf{Métrica} & \\textbf{Esfera} & \\textbf{Toro} \\\\ \\hline",
        f"Puntos por forma & {n_points} & {n_points} \\\\",
        f"Nivel de ruido   & {noise_level:.2f} & {noise_level:.2f} \\\\",
        f"\\\\(\\beta_0\\) (TDA) & {b0_s} & {b0_t} \\\\",
        f"\\\\(\\beta_1\\) (TDA) & {b1_s} & {b1_t} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]
    latex_table_tda = "\n".join(latex_lines)
    latex_bytes_tda = latex_table_tda.encode('utf-8')

    col_tda1, col_tda2, col_tda3, col_tda4 = st.columns(4)
    with col_tda1:
        st.download_button(
            label="\U0001f4e5 CSV",
            data=csv_tda,
            file_name="nube_puntos_tda.csv",
            mime="text/csv",
            key="btn_download_tda_csv"
        )
    with col_tda2:
        st.download_button(
            label="\U0001f4c4 PDF",
            data=pdf_bytes_tda,
            file_name="reporte_analisis_tda.pdf",
            mime="application/pdf",
            key="btn_download_tda_pdf"
        )
    with col_tda3:
        st.download_button(
            label="\U0001f5bc\ufe0f PNG",
            data=png_bytes_tda,
            file_name="graficos_analisis_tda.png",
            mime="image/png",
            key="btn_download_tda_png"
        )
    with col_tda4:
        st.download_button(
            label="\U0001f4d0 LaTeX",
            data=latex_bytes_tda,
            file_name="tabla_metricas_tda.tex",
            mime="text/plain",
            key="btn_download_tda_latex"
        )
    # ANÁLISIS DE ESTABILIDAD (Barrido de Ruido)
    # ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 Análisis de Estabilidad — Barrido Sistemático de Ruido", expanded=False):
        st.markdown("""
        **H.E.1**: *"Los números de Betti β₀ y β₁ permanecen estables bajo perturbaciones del 15–20% 
        en los datos de entrada, superando a los descriptores euclidianos (K-Medias)"*
        """)

        st.markdown("**Parámetros del barrido**")
        st.markdown(f"""
        - Formas: Esfera (β₀=1, β₁=0) y Toro (β₀=1, β₁=2)
        - Puntos por forma: {n_points}
        - Rango de ruido: 0% → 30% (10 pasos)
        - Clústeres K-Medias: {n_clusters}
        """)

        resultados_previos = st.session_state.get("tda_sweep_done", False) and st.session_state.get("tda_sweep_results") is not None
        if resultados_previos:
            st.success("✅ Barrido completado. Presione el botón para recalcular.")

        sweep_btn = st.button("▶ Ejecutar Barrido de Ruido", key="btn_run_sweep")
        if sweep_btn:
            st.session_state.tda_sweep_done = False
            n_steps = 10
            pbar_sweep = st.progress(0, text="Computando barrido de ruido (0% → 30%)...")
            status_sweep = st.empty()

            def _report_progress(current, total):
                pbar_sweep.progress(current / total, text=f"Barrido: {current}/{total} — Ruido: {noise_vals[current-1]*100:.0f}%")
                status_sweep.info(f"Procesando nivel de ruido {current}/{total}...")

            noise_vals = np.linspace(0.0, 0.30, n_steps)
            st.session_state.tda_sweep_results = compute_noise_sweep(
                n_points=n_points,
                n_clusters=n_clusters,
                n_steps=n_steps,
                progress_callback=_report_progress,
            )
            pbar_sweep.empty()
            status_sweep.empty()
            st.session_state.tda_sweep_done = True
            st.rerun()

        if st.session_state.get('tda_sweep_done', False) and st.session_state.get('tda_sweep_results') is not None:
            results = st.session_state.tda_sweep_results

            # ── Stability plot ──
            fig_stab = plot_stability_chart(results)
            apply_plotly_theme(fig_stab)
            st.plotly_chart(fig_stab, use_container_width=True)

            # ── Animación: diagramas de persistencia por nivel de ruido ──
            if "diagrams_s" in results and len(results["diagrams_s"]) > 0:
                with st.expander("▶ Animación: Diagramas de Persistencia por Nivel de Ruido", expanded=False):
                    st.markdown(
                        "Usá el slider o el botón ▶ para ver cómo evolucionan los diagramas "
                        "de persistencia (H₀ y H₁) de esfera y toro a medida que aumenta el ruido."
                    )
                    fig_anim = plot_sweep_persistence_animation(results)
                    apply_plotly_theme(fig_anim)
                    st.plotly_chart(fig_anim, use_container_width=True)

            # ── Summary table ──
            st.markdown("**Resultados del barrido**")
            noise_pct = results["noise_vals"] * 100
            df_sweep = pd.DataFrame({
                "Ruido (%)": [f"{n*100:.0f}%" for n in results["noise_vals"]],
                "β₀ Esfera": [b[0] for b in results["betti_s"]],
                "β₁ Esfera": [b[1] for b in results["betti_s"]],
                "β₀ Toro": [b[0] for b in results["betti_t"]],
                "β₁ Toro": [b[1] for b in results["betti_t"]],
                "Accuracy K-Medias": [f"{a:.1%}" for a in results["acc"]],
            })
            st.dataframe(df_sweep, use_container_width=True, hide_index=True)

            # ── Interpretation ──
            interpretation = interpret_stability(results)
            st.markdown("**Interpretación**")
            if interpretation["stable"]:
                st.success(f"""
                ✅ **{interpretation['verdict']}:**
                {interpretation['details']}
                """)
            else:
                st.warning(f"""
                ⚠️ **{interpretation['verdict']}:**
                {interpretation['details']}
                """)

    # Restaurar estilo matplotlib según el tema para las figuras inline
    apply_mpl_theme()
