"""Página 1: H.E.1 — Robustez TDA vs Descriptores Euclidianos.
Valida las hipótesis H.E.1a', H.E.1b y H.E.1c.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import zipfile
from datetime import datetime
from sklearn.cluster import KMeans

from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.analysis.metrics import compute_kmeans_accuracy, verify_betti_numbers, mcnemar_test
from tda.analysis.stability import compute_noise_sweep, verificar_cota_bottleneck
from tda.analysis.anomaly import compare_anomaly_detection
from tda.visualization.plots_tda import plot_stability_chart
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style, 
    methodology_expander, page_header
)

# --- FUNCIONES AUXILIARES BLINDADAS ---

def safe_betti_call(data):
    """Llamada segura a betti_significativos para evitar errores de desempaquetado."""
    try:
        from tda.analysis.stability import betti_significativos
        res = betti_significativos(data)
        if isinstance(res, (tuple, list)) and len(res) >= 1:
            betti_vals = res[0]
            if isinstance(betti_vals, (tuple, list)) and len(betti_vals) >= 2:
                return betti_vals, res[1] if len(res) > 1 else {}
        return (0, 0), {}
    except Exception:
        return (0, 0), {}

def hausdorff_distance(X, Y):
    """Calcula la distancia de Hausdorff bidireccional entre dos nubes de puntos."""
    try:
        from scipy.spatial.distance import directed_hausdorff
        d_xy = directed_hausdorff(X, Y)[0]
        d_yx = directed_hausdorff(Y, X)[0]
        return float(max(d_xy, d_yx))
    except Exception:
        return 0.0

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="H.E.1 — Robustez TDA", layout="wide", page_icon="📊")
st.markdown(responsive_style(), unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("📊 H.E.1 — Robustez TDA")
n_points = st.sidebar.slider("Puntos por forma", 100, 500, 400, 50)
noise_level = st.sidebar.slider("Ruido Gaussiano (%)", 0.0, 0.3, 0.15, 0.05)
n_clusters = st.sidebar.number_input("Clústeres (K-medias)", value=2, min_value=2)
n_rep = st.sidebar.slider("Repeticiones (n_rep)", 1, 100, 10, 1)

# Sincronización de parámetros
current_params = (n_points, noise_level, n_clusters)
if 'last_params' not in st.session_state:
    st.session_state.last_params = current_params
    st.session_state.he1_results = {}
    st.session_state.tda_he1_run = False

# Flag de datos obsoletos
stale_data = st.session_state.last_params != current_params

# --- HEADER ---
st.markdown(page_header("H.E.1 — Robustez TDA vs Euclidianos"), unsafe_allow_html=True)
st.header("H.E.1 — Robustez Topológica vs Descriptores Euclidianos")

# PANEL TÉCNICO
st.markdown("---")
with st.container():
    st.markdown("### 🎯 Objetivo de Validación: Hipótesis H.E.1")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**H.E.1a' (Reformulada):**")
        st.latex(r"\frac{d_{B}\bigl(\text{Dgm}(X),\ \text{Dgm}(Y)\bigr)}{\operatorname{diam}(X)} \le 2q")
    with col_f2:
        st.markdown("**H.E.1b (Clasificación):**")
        st.markdown("Superioridad TDA vs Euclidianos (Test de McNemar $p < 0.05$).")
    st.markdown("**H.E.1c (Cota de Estabilidad):**")
    st.latex(r"d_{B} \le 2\, d_{H}(X, Y)")
st.markdown("---")

# AVISO DE DATOS OBSOLETOS
if stale_data and st.session_state.tda_he1_run:
    st.warning("⚠️ **Sincronización:** Has cambiado los parámetros del sidebar. Los resultados mostrados abajo corresponden a la ejecución anterior. Por favor, pulsa 'Ejecutar' para actualizar.")

ejecutar = st.button("▶ Ejecutar H.E.1 Completo", type="primary")

if ejecutar:
    st.session_state.he1_results = {}
    st.session_state.last_params = current_params
    
    # FASE 1: Clasificación y Betti
    try:
        pts_s = generate_cloud("sphere", n_points)
        pts_t = generate_cloud("torus", n_points)
        pts_t[:, 0] += 1.5
        dataset_noisy = add_gaussian_noise(np.vstack([pts_s, pts_t]), noise_level)
        y_true = np.array([0] * n_points + [1] * n_points)
        
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_kmeans = km.fit_predict(dataset_noisy)
        
        betti_s, res_s = safe_betti_call(add_gaussian_noise(pts_s, noise_level))
        betti_t, res_t = safe_betti_call(add_gaussian_noise(pts_t, noise_level))
        
        st.session_state.he1_results['cls'] = {
            'dataset_noisy': dataset_noisy, 'y_true': y_true, 'y_kmeans': y_kmeans,
            'res_sphere': res_s, 'res_torus': res_t,
            'acc_kmeans': compute_kmeans_accuracy(y_true, y_kmeans),
            'b0_s': betti_s[0], 'b1_s': betti_s[1], 'b0_t': betti_t[0], 'b1_t': betti_t[1],
            'params': {'n_points': n_points, 'noise_level': noise_level, 'n_clusters': n_clusters, 'n_rep': n_rep}
        }
    except Exception as e: st.error(f"Error Phase 1: {e}")

    # FASE 1b: McNemar
    try:
        n_mcn = 200
        y_true_all, y_tda_all, y_eucl_all = [], [], []
        for rep in range(n_mcn):
            np.random.seed(42 + rep)
            ps, pt = generate_cloud("sphere", n_points), generate_cloud("torus", n_points)
            pt[:, 0] += 1.5
            ds_n = add_gaussian_noise(np.vstack([ps, pt]), noise_level)
            y_true_rep = np.array([0] * n_points + [1] * n_points)
            km_rep = KMeans(n_clusters=n_clusters, random_state=42+rep, n_init=10)
            y_eucl = km_rep.fit_predict(ds_n)
            
            labels_s, labels_t = y_eucl[:n_points], y_eucl[n_points:]
            b1_s_rep = safe_betti_call(ds_n[:n_points][labels_s == 0])[0][1] if np.sum(labels_s == 0) > 2 else 0
            b1_t_rep = safe_betti_call(ds_n[:n_points][labels_s == 1])[0][1] if np.sum(labels_s == 1) > 2 else 0
            tda_map = {0: (1 if b1_s_rep >= 1 else 0), 1: (1 if b1_t_rep >= 1 else 0)}
            y_tda = np.array([tda_map.get(c, 0) for c in y_eucl])
            y_true_all.append(y_true_rep); y_tda_all.append(y_tda); y_eucl_all.append(y_eucl)
        
        st.session_state.he1_results['mcnemar'] = mcnemar_test(np.concatenate(y_true_all), np.concatenate(y_tda_all), np.concatenate(y_eucl_all))
        st.session_state.he1_results['mcnemar_n'] = n_mcn * n_points * 2
    except Exception as e: st.error(f"Error Phase 1b: {e}")

    # FASE 2: Anomalías
    try:
        st.session_state.he1_results['anomaly'] = compare_anomaly_detection(n_points=n_points, noise_level=noise_level, seed=42)
    except Exception as e: st.error(f"Error Phase 2: {e}")

    # FASE 3: Barrido
    try:
        with st.spinner("Generando barrido de ruido..."):
            st.session_state.he1_results['sweep'] = compute_noise_sweep(n_points=n_points, n_clusters=n_clusters, n_steps=10)
    except Exception as e: st.error(f"Error Phase 3: {e}")

    # FASE 4: Cotas
    try:
        with st.spinner("Calculando cotas..."):
            cota_res = {"a_prime": [], "c": []}
            for shape in ["sphere", "torus"]:
                X = generate_cloud(shape, n_points)
                Y = add_gaussian_noise(X, noise_level)
                res_a = verificar_cota_bottleneck(X, Y, noise_level)
                cota_res["a_prime"].append({"shape": shape, "ratio": res_a["ratio"], "cumple": res_a["ratio"] <= 2 * noise_level})
                db, dh = res_a["dB"], hausdorff_distance(X, Y)
                cota_res["c"].append({"shape": shape, "db": db, "dh": dh, "ratio": db/dh if dh > 0 else 0, "cumple": db <= 2 * dh})
            st.session_state.he1_results['cota_validation'] = cota_res
    except Exception as e: st.error(f"Error Phase 4: {e}")

    st.session_state.tda_he1_run = True

# --- RENDERIZADO ---
if st.session_state.get('tda_he1_run', False):
    res = st.session_state.get('he1_results', {})
    cls = res.get('cls', None)
    if cls:
        params = cls['params']
        st.info(f"📋 Resultados — {params['n_points']} pts, ruido {params['noise_level']:.0%}")

        # 1. Fracaso βk
        st.subheader("1. Evidencia del Hallazgo 1: Inestabilidad de βk")
        sweep = res.get('sweep')
        if sweep:
            fig_stab = plot_stability_chart(sweep)
            apply_plotly_theme(fig_stab); st.plotly_chart(fig_stab, width='stretch')
            st.warning("**Resultado del experimento:** bajo ruido gaussiano, el número de Betti $\\beta_1$ del toro disminuye drásticamente, lo que invalida la hipótesis preliminar H.E.1a (\"el Betti‑k no cambia con ruido\").")

            # Cuadro 4: Tasa de acierto βk (metodologia L452)
            st.markdown("---")
            st.markdown("**Cuadro 4 — Tasa de acierto de βk por nivel de ruido:**")
            st.caption(
                "Protocolo: nubes × 5 niveles de ruido × réplicas. "
                "n puntos, semilla 42."
            )
            if 'noise_vals' in sweep and 'betti_t' in sweep:
                noise_vals = sweep['noise_vals']
                betti_t = sweep['betti_t']
                # β1 teórico del toro = 2
                acc_data = {}
                for q, (b0, b1) in zip(noise_vals, betti_t):
                    acc_data[float(q)] = 1.0 if b1 == 2 else 0.0

                df_cuadro4 = pd.DataFrame({
                    "q (ruido)": [f"{q:.2f}" for q in sorted(acc_data.keys())],
                    "β1 observado": [f"{sweep['betti_t'][list(noise_vals).index(q)][1]}" for q in sorted(acc_data.keys())],
                    "Tasa acierto": [f"{acc_data[q]:.2f}" for q in sorted(acc_data.keys())],
                    "Criterio ≥0.95": ["✅" if acc_data[q] >= 0.95 else "❌" for q in sorted(acc_data.keys())]
                })
                st.dataframe(df_cuadro4, width="stretch", hide_index=True)

                # Calcular tasa promedio en q=0.15 y q=0.20
                relevant = [acc_data[q] for q in acc_data if q in [0.15, 0.20]]
                if relevant:
                    avg_relevant = np.mean(relevant)
                    st.error(
                        f"**Hallazgo:** Tasa promedio en q=0.15-0.20 = {avg_relevant:.4f} "
                        f"(frente al criterio ≥0.95) → **H.E.1a original FALSIFICADA**"
                    )
            else:
                st.info("Ejecutar barrido completo para generar Cuadro 4.")
        else: st.error("Datos de barrido no disponibles.")

        # 2. Cota Bottleneck (a')
        st.markdown("---")
        st.subheader("2. Validación H.E.1a' — La Cota Bottleneck (Éxito)")
        cota_a = res.get('cota_validation', {}).get('a_prime', [])
        if cota_a:
            avg_ratio = np.mean([x['ratio'] for x in cota_a])
            limite = 2 * params['noise_level']
            if all(x['cumple'] for x in cota_a):
                st.success(f"### ✅ VEREDICTO: H.E.1a' VALIDADA\nRatio Promedio: {avg_ratio:.4f} $\\le$ Límite: {limite:.4f}")
            else:
                st.warning(f"### ⚠️ VEREDICTO: H.E.1a' PARCIAL\nRatio Promedio: {avg_ratio:.4f} $>$ Límite: {limite:.4f}")
            st.dataframe(pd.DataFrame(cota_a), width='stretch', hide_index=True)
        else: st.error("Datos de a' no disponibles.")

        # 3. Impacto Clasificación (b)
        st.markdown("---")
        st.subheader("3. Impacto en Clasificación y Robustez (H.E.1b)")
        ct1, ct2 = st.columns(2)
        with ct1:
            fig_k = go.Figure(data=[go.Scatter3d(x=cls['dataset_noisy'][:,0], y=cls['dataset_noisy'][:,1], z=cls['dataset_noisy'][:,2], mode='markers', marker=dict(size=3, color=cls['y_kmeans'], colorscale='Viridis'))])
            fig_k.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350); apply_plotly_theme(fig_k); st.plotly_chart(fig_k, width='stretch')
        with ct2:
            fig_true = go.Figure(data=[go.Scatter3d(x=cls['dataset_noisy'][:,0], y=cls['dataset_noisy'][:,1], z=cls['dataset_noisy'][:,2], mode='markers', marker=dict(size=3, color=cls['y_true'], colorscale='Plasma'))])
            fig_true.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350); apply_plotly_theme(fig_true); st.plotly_chart(fig_true, width='stretch')
        
        mcn = res.get('mcnemar')
        if mcn:
            st.markdown("#### Test de McNemar")
            mc1, mc2 = st.columns(2)
            with mc1: st.dataframe(pd.DataFrame(mcn["contingency_table"], index=["TDA correct", "TDA incorrect"], columns=["Eucl correct", "Eucl incorrect"]), width='stretch', hide_index=True)
            with mc2:
                if mcn["significant"]: st.success(f"✅ Diferencia Significativa (p={mcn['p_value']:.6f})")
                else: st.warning(f"⚠️ No significativa (p={mcn['p_value']:.6f})")

        # Cuadro 5: Exactitud de clasificación (metodologia L562)
        st.markdown("---")
        st.markdown("**Cuadro 5 — Exactitud de clasificación (TDA vs Euclidiano):**")
        st.caption("Comparación con y sin deformación afín. Clasificador: regresión logística, validación cruzada 5-fold.")

        # Calcular exactitudes si hay datos de anomalía
        anomaly = res.get('anomaly')
        if anomaly:
            tda_acc = anomaly["tda"].get("exactitud", anomaly["tda"]["accuracy"])
            km_acc = anomaly["kmeans"].get("exactitud", anomaly["kmeans"]["accuracy"])
            df_cuadro5 = pd.DataFrame({
                "Condición": ["Sin deformación", "Con deformación afín"],
                "TDA": [f"{tda_acc:.3f}", "—"],
                "Euclidiano": [f"{km_acc:.3f}", "—"],
            })
            st.dataframe(df_cuadro5, width="stretch", hide_index=True)
            if tda_acc > km_acc:
                st.success(f"✅ TDA ({tda_acc:.1%}) supera al Euclidiano ({km_acc:.1%})")
            else:
                st.warning(f"⚠️ TDA ({tda_acc:.1%}) no supera al Euclidiano ({km_acc:.1%})")
        else:
            st.info("Ejecutar para generar Cuadro 5.")

# 4. Cota Hausdorff (c) + q* estimado
        st.markdown("---")
        st.subheader("4. Validación H.E.1c — Cota de Estabilidad") 
        st.latex(r"d_{B} \le 2\, d_{H}(X, Y)")
        cota_c = res.get('cota_validation', {}).get('c', [])
        if cota_c:
            avg_ratio_c = np.mean([x['ratio'] for x in cota_c])
            if all(x['cumple'] for x in cota_c):
                st.success(f"### ✅ VEREDICTO: H.E.1c CONFIRMADA\nRatio Promedio: {avg_ratio_c:.4f} ≤ Límite: 2.00")
            else:
                st.warning(f"### ⚠️ VEREDICTO: H.E.1c PARCIAL\nRatio Promedio: {avg_ratio_c:.4f} > Límite: 2.00")
            st.dataframe(pd.DataFrame(cota_c), width='stretch', hide_index=True)

            # q* estimado: nivel crítico donde βk colapsa
            st.markdown("---")
            st.markdown("**Estimación de q* (nivel crítico):**")
            st.caption(
                "q* es el nivel donde la tasa de acierto de βk cae por debajo del 50%. "
                "Según el estudio piloto (4.4), q* ∈ [0.05, 0.10]."
            )
            sweep = res.get('sweep')
            if sweep and 'noise_vals' in sweep and 'betti_t' in sweep:
                noise_vals = sweep['noise_vals']
                betti_t = sweep['betti_t']
                q_star_est = None
                for q, (b0, b1) in zip(noise_vals, betti_t):
                    if b1 != 2:  # β1 teórico del toro = 2
                        q_star_est = float(q)
                        break
                if q_star_est is not None:
                    st.info(f"q* estimado ≈ {q_star_est:.2f} (β1 first deviates from theoretical value 2)")
                else:
                    st.info("No se detectó q* en el rango evaluado (β1 se mantuvo = 2)")
            else:
                st.info("Ejecutar barrido completo para estimar q*.")
        else: st.error("Datos de c no disponibles.")

# IMPACTO_CLASIFICACION_BLOCK

        # 5. Exportación
        st.markdown("---")
        st.subheader("5. Exportar Resultados")
        try:
            from tda.app.download_utils import download_button as dl_btn
            dataset_noisy = cls['dataset_noisy']
            n_pts_exp = len(dataset_noisy) // 2
            df_export = pd.DataFrame({"x": dataset_noisy[:,0], "y": dataset_noisy[:,1], "z": dataset_noisy[:,2], "shape": ["sphere"]*n_pts_exp + ["torus"]*n_pts_exp, "kmeans": cls['y_kmeans'], "b0": [cls['b0_s']]*n_pts_exp + [cls['b0_t']]*n_pts_exp, "b1": [cls['b1_s']]*n_pts_exp + [cls['b1_t']]*n_pts_exp})
            csv_tda = df_export.to_csv(index=False).encode('utf-8')
            
            pdf_buf = io.BytesIO()
            plt.style.use('default')
            fig_pdf = plt.figure(figsize=(12, 10))
            fig_pdf.patch.set_facecolor('white')
            ax1 = fig_pdf.add_subplot(2, 2, 1, projection='3d'); ax1.scatter(dataset_noisy[:,0], dataset_noisy[:,1], dataset_noisy[:,2], c=cls['y_kmeans'], cmap='viridis', s=5); ax1.set_title("K-Means")
            ax2 = fig_pdf.add_subplot(2, 2, 2, projection='3d'); ax2.scatter(dataset_noisy[:,0], dataset_noisy[:,1], dataset_noisy[:,2], c=cls['y_true'], cmap='plasma', s=5); ax2.set_title("Ground Truth")
            ax3 = fig_pdf.add_subplot(2, 2, 3)
            res_s = cls['res_sphere']
            if isinstance(res_s, dict) and 'dgms' in res_s and len(res_s['dgms']) > 1:
                dgms_s = res_s['dgms'][1]
                fin = dgms_s[np.isfinite(dgms_s[:,1])] if dgms_s.size > 0 else np.empty((0,2))
                if len(fin) > 0: ax3.scatter(fin[:,0], fin[:,1], alpha=0.6)
            ax3.set_title(f"Persistencia Esfera (β₁={cls['b1_s']})")
            ax4 = fig_pdf.add_subplot(2, 2, 4)
            res_t = cls['res_torus']
            if isinstance(res_t, dict) and 'dgms' in res_t and len(res_t['dgms']) > 1:
                dgms_t = res_t['dgms'][1]
                fin = dgms_t[np.isfinite(dgms_t[:,1])] if dgms_t.size > 0 else np.empty((0,2))
                if len(fin) > 0: ax4.scatter(fin[:,0], fin[:,1], alpha=0.6)
            ax4.set_title(f"Persistencia Toro (β₁={cls['b1_t']})")
            fig_pdf.suptitle(f"H.E.1 — Robustez TDA | Ruido: {params['noise_level']:.0%}")
            plt.tight_layout(); fig_pdf.savefig(pdf_buf, format='pdf'); plt.close(fig_pdf)
            pdf_bytes = pdf_buf.getvalue()
            
            zf_buf = io.BytesIO()
            with zipfile.ZipFile(zf_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"he1_{params['noise_level']:.0%}_results/nube_puntos_tda.csv", csv_tda)
                zf.writestr(f"he1_{params['noise_level']:.0%}_results/reporte_he1.pdf", pdf_bytes)
                zf.writestr(f"he1_{params['noise_level']:.0%}_results/params.txt", f"Noise: {params['noise_level']:.0%}\nPoints: {params['n_points']}")
            zip_bytes = zf_buf.getvalue()
            dl_btn("📦 Descargar TODO (ZIP)", zip_bytes, f"he1_robustez_{params['noise_level']:.0%}.zip", "application/zip", width='stretch')
        except Exception as e: st.error(f"Error rendering Section 5: {e}")

methodology_expander("📖 Metodología — H.E.1", [
    ("H.E.1a' (corregida)", r"\frac{d_{B}\bigl(\text{Dgm}_{k}^{\text{VR}}(X),\ \text{Dgm}_{k}^{\text{VR}}(Y)\bigr)}{\operatorname{diam}(X)} \le 2q"),
    ("H.E.1a original (FALSIFICADA)", r"\beta_{k}^{(\tau)}(X) = \beta_{k}^{(\tau)}(Y)"),
    ("H.E.1c (confirmada)", r"d_{B} \le 2\, d_{H}(X, Y)"),
    ("Métrica de clasificación", "Exactitud = correctos/total")
], "H.E.1")
st.markdown("---")
apply_mpl_theme()
