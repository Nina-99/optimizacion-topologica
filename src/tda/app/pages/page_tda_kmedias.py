"""Página 2: TDA vs K-Medias (H.E.1).

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
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from ripser import ripser
from matplotlib.backends.backend_pdf import PdfPages

from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.analysis.metrics import compute_kmeans_accuracy, verify_betti_numbers
from tda.analysis.stability import compute_noise_sweep
from tda.visualization.plots_tda import plot_stability_chart, interpret_stability


def run():
    """Render the TDA vs K-Medias page (sidebar + content)."""
    # ── Sidebar ──
    st.sidebar.header("📊 TDA vs K-Medias (Pestaña 2)")
    n_points = st.sidebar.slider("Puntos por forma", 100, 500, 250, 50, key="tda_n_points")
    noise_level = st.sidebar.slider("Ruido Gaussiano (%)", 0.0, 0.3, 0.15, 0.05, key="tda_noise_level")
    n_clusters = st.sidebar.number_input("Clústeres (K-medias)", value=2, min_value=2, key="tda_n_clusters")

    # ── Main content ──
    # ------------------------------------------
    # PÁGINA 2: TDA vs K-MEDIAS
    # ------------------------------------------
    st.header("Robustez Topológica vs Distancia Euclidiana (H.E.1)")

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
            st.session_state.tda_run = True

    # Renderizar resultados si existen en session state
    if st.session_state.get('tda_run', False):
        dataset_noisy = st.session_state.tda_dataset_noisy
        y_kmeans = st.session_state.tda_y_kmeans
        y_true = st.session_state.tda_y_true
        res_sphere = st.session_state.tda_res_sphere
        res_torus = st.session_state.tda_res_torus

        ct1, ct2 = st.columns(2)
        with ct1:
            st.subheader("Clasificación Euclidiana (K-Medias)")
            st.markdown("*Falla al separar geometrías superpuestas*")
            fig_k = go.Figure(data=[go.Scatter3d(
                x=dataset_noisy[:, 0], y=dataset_noisy[:, 1], z=dataset_noisy[:, 2],
                mode='markers', marker=dict(size=3, color=y_kmeans, colorscale='Viridis')
            )])
            fig_k.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
            st.plotly_chart(fig_k, use_container_width=True)

        with ct2:
            st.subheader("Clasificación Real (Ground Truth)")
            st.markdown("*Separación perfecta por invariantes topológicos*")
            fig_true = go.Figure(data=[go.Scatter3d(
                x=dataset_noisy[:, 0], y=dataset_noisy[:, 1], z=dataset_noisy[:, 2],
                mode='markers', marker=dict(size=3, color=y_true, colorscale='Plasma')
            )])
            fig_true.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
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
        # dataset_noisy has shape (2*n_points, 3)
        # We need: coordinates, shape labels, K-Means labels, and calculated Betti numbers
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

        # 2. Crear PDF
        pdf_buf_tda = io.BytesIO()
        fig_pdf_tda = plt.figure(figsize=(12, 10))

        # K-Means 3D
        ax1 = fig_pdf_tda.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2], c=y_kmeans, cmap='viridis', s=5)
        ax1.set_title("Clasificación Euclidiana (K-Medias)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # Ground Truth 3D
        ax2 = fig_pdf_tda.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2], c=y_true, cmap='plasma', s=5)
        ax2.set_title("Clasificación Real (Ground Truth)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        # Sphere persistence diagram
        ax3 = fig_pdf_tda.add_subplot(2, 2, 3)
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
        plt.close(fig_pdf_tda)
        pdf_bytes_tda = pdf_buf_tda.getvalue()
        pdf_buf_tda.close()

        col_tda_csv, col_tda_pdf = st.columns(2)
        with col_tda_csv:
            st.download_button(
                label="📥 Descargar Datos Nube de Puntos (CSV)",
                data=csv_tda,
                file_name="nube_puntos_tda.csv",
                mime="text/csv",
                key="btn_download_tda_csv"
            )
        with col_tda_pdf:
            st.download_button(
                label="📥 Descargar Gráficos de Análisis (PDF)",
                data=pdf_bytes_tda,
                file_name="reporte_analisis_tda.pdf",
                mime="application/pdf",
                key="btn_download_tda_pdf"
            )

        # ────────────────────────────────────────────────────────
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

            if st.button("▶ Ejecutar Barrido de Ruido", key="btn_run_sweep"):
                st.session_state.tda_sweep_done = False
                with st.spinner("Computando barrido de ruido (0% → 30%)..."):
                    st.session_state.tda_sweep_results = compute_noise_sweep(
                        n_points=n_points,
                        n_clusters=n_clusters
                    )
                    st.session_state.tda_sweep_done = True
                    st.rerun()

            if st.session_state.get('tda_sweep_done', False) and st.session_state.get('tda_sweep_results') is not None:
                results = st.session_state.tda_sweep_results

                # ── Stability plot ──
                fig_stab = plot_stability_chart(results)
                st.plotly_chart(fig_stab, use_container_width=True)

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
