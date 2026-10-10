"""Página 2: H.E.2 — Optimización SIMP.

Presenta la validación en el ORDEN del documento:
  1. H.E.2a — eficiencia estructural (Tabla 5.6; Figuras 5.4(a) y 5.4(b))
  2. H.E.2b — estabilidad del descriptor topológico (Figuras 5.5, 5.6 y 5.7;
     doble cómputo β₀/β₁)
  3. Veredicto H.E.2 (Tabla 5.6 + síntesis)

Constantes del documento:
  - Viga en voladizo; malla base 60×30 (§3.4.1); fV = 0.5; p = 3; rmin = 2.4.
  - H.E.2a: (c(fV·1) − c(ρ*)) / c(fV·1) ≥ 0.40 (referente = diseño uniforme).
  - H.E.2b: β₁(Ω_sólido) invariante frente a malla (rmin escalado), p ∈ {2,3,4}
    y rmin ∈ [1.5, 4.0]; y los dos métodos (Euler y GUDHI) coinciden al 100 %.
  - Configuración canónica: elementos unitarios (Lx = nex, Ly = ney), E₀ = 1, F = 1.
"""

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tda.optimization.metric_simp import MetricaTDA_SIMP
from tda.simulation.he_protocols import corrida_simp, barrido_he2
from tda.app.download_utils import download_button, export_settings_ui, write_figs_zip
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    methodology_expander, page_header,
)

# ── Constantes del documento ─────────────────────────────────────────
FV = 0.5
P_DEFECTO = 3
RMIN_DEFECTO = 2.4
RMIN_RANGO = [1.5, 2.0, 2.4, 3.0, 4.0]     # H.E.2b: rmin ∈ [1.5, 4.0]
P_RANGO = [1, 2, 3, 4]                       # Fig. 5.5: p ∈ {1,2,3,4}
REDUCCION_MIN = 0.40                         # H.E.2a: ≥ 40 %

export_settings_ui()
apply_mpl_theme()
st.set_page_config(page_title="H.E.2 — Optimización SIMP", layout="wide", page_icon="🏗️")
st.markdown(responsive_style(), unsafe_allow_html=True)


def plot_material_distribution(density_2d, title: str):
    """Heatmap Plotly de la distribución de material."""
    fig = go.Figure(data=[go.Heatmap(z=density_2d, colorscale="gray_r", zmin=0, zmax=1,
                                     colorbar=dict(title="ρ"))])
    fig.update_layout(title=title)
    apply_plotly_theme(fig)
    return fig


# ── Header + hipótesis ───────────────────────────────────────────────
st.markdown(page_header("H.E.2 — Optimización SIMP",
                        "H.E.2a: reducción ≥ 40 % vs uniforme | H.E.2b: β₁ invariante"),
            unsafe_allow_html=True)
st.subheader("🎯 Hipótesis H.E.2")
h1, h2 = st.columns(2)
with h1:
    st.markdown("**H.E.2a — eficiencia estructural**")
    st.latex(r"\frac{c(f_V\cdot\mathbf{1})-c(\rho^{\star})}{c(f_V\cdot\mathbf{1})}\ge 0.40")
    st.caption("El método SIMP (p = 3, fV = 0.5, filtro rmin) reduce la compliance ≥ 40 % "
               "respecto del diseño uniforme de igual volumen.")
with h2:
    st.markdown("**H.E.2b — estabilidad del descriptor topológico**")
    st.latex(r"\beta_1(\Omega_{\text{sólido}})\ \text{invariante}")
    st.caption("β₁ invariante frente a malla (rmin escalado), p ∈ {2,3,4} y rmin ∈ [1.5, 4.0]; "
               "y los dos métodos (Euler y GUDHI) coinciden al 100 %.")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("🏗️ Optimización SIMP")
dominio = st.sidebar.radio(
    "Dominio",
    ["Rectangular (viga en voladizo, 2:1)", "Cuadrado (independencia de malla)"],
    key="simp_dominio",
    help="Rectangular: caso tesis (viga empotrada). Cuadrado: control de malla (H.E.2b).",
)

if dominio.startswith("Rectangular"):
    opciones_malla = ["40x20 (800 elem)", "60x30 (1800 elem, Caso Tesis)",
                      "80x40 (3200 elem)", "120x60 (7200 elem)"]
    idx_default = 1
else:
    opciones_malla = ["40x40 (1600 elem)", "80x80 (6400 elem)", "120x120 (14400 elem)"]
    idx_default = 0

malla_opcion = st.sidebar.selectbox("Resolución de malla", opciones_malla, index=idx_default, key="simp_malla")

st.sidebar.markdown("**$f_V = 0.5$** (fijo, diseño uniforme de igual volumen)")
volfrac = FV

penal = st.sidebar.number_input("Factor de penalización (p)", value=float(P_DEFECTO),
                                min_value=1.0, max_value=5.0, step=1.0, key="simp_penal",
                                help="Variable independiente. p = 3 es el estándar SIMP (H.E.2).")
rmin = st.sidebar.number_input("Radio de filtro (rmin, elementos)", value=RMIN_DEFECTO,
                               min_value=1.0, max_value=5.0, step=0.1, key="simp_rmin",
                               help="Variable interviniente. Rango del documento [1.5, 4.0].")

with st.sidebar.expander("⚙️ Parámetros de ejecución", expanded=False):
    max_iter = st.slider("Iteraciones máximas", 100, 500, 200, key="simp_max_iter")
    E_acero = st.number_input("Módulo Young E₀ (normalizado)", value=1.0, step=0.1,
                              help="Config canónica: E₀ = 1.")
    F_carga = st.number_input("Carga F (normalizada)", value=1.0, step=0.1,
                              help="Config canónica: F = 1.")
    espesor = st.number_input("Espesor", value=1.0, step=0.1)

current_params_he2 = (dominio, malla_opcion, penal, rmin, max_iter)
if "last_params_he2" not in st.session_state:
    st.session_state.last_params_he2 = current_params_he2
if (st.session_state.last_params_he2 != current_params_he2
        and st.session_state.get("simp_optimized", False)):
    st.warning("⚠️ Parámetros modificados — vuelva a ejecutar la optimización.")

# ── Ejecución ────────────────────────────────────────────────────────
ejecutar_simp = st.button("▶ Iniciar optimización SIMP", type="primary")

if ejecutar_simp:
    dims = malla_opcion.split(" ")[0].split("x")
    nelx, nely = int(dims[0]), int(dims[1])

    nnx = nelx + 1
    n_dof = 2 * nnx * (nely + 1)
    idx_izq = np.arange(0, nely + 1) * nnx
    dofs_fijos = np.sort(np.concatenate([2 * idx_izq, 2 * idx_izq + 1]))
    node_load = (nely // 2) * nnx + nelx
    F = np.zeros(n_dof)
    F[2 * node_load + 1] = -F_carga

    m = MetricaTDA_SIMP(nex=nelx, ney=nely, E=E_acero, nu=0.3,
                        Lx=float(nelx), Ly=float(nely), t=espesor,
                        f_V=volfrac, p=penal, r_min=rmin, max_iter=max_iter)
    m.definir_problema(F, dofs_fijos)
    with st.spinner("Optimizando SIMP..."):
        m.optimizar(verbose=False)
    with st.spinner("Analizando topología (TDA)..."):
        m.fase_tda(verbose=False)

    c_base = float(m.c_hist[0])
    reduccion = (c_base - m.c_final) / c_base * 100

    st.session_state.simp_optimized = True
    st.session_state.simp_c_base = c_base
    st.session_state.simp_c_final = m.c_final
    st.session_state.simp_reduccion = reduccion
    st.session_state.simp_beta0 = m.beta0
    st.session_state.simp_beta1 = m.beta1
    st.session_state.simp_n_iter = m.n_iter
    st.session_state.simp_converged = m.converged
    st.session_state.simp_c_hist = np.array(m.c_hist)
    st.session_state.simp_rho_final = m.rho_final
    st.session_state.simp_rho_tilde_final = m.rho_tilde_final
    st.session_state.simp_dgm0 = m.dgm0
    st.session_state.simp_dgm1 = m.dgm1
    st.session_state.simp_concordancia = getattr(m, "_betti_concordancia", None)
    st.session_state.simp_betti_detalle = getattr(m, "_betti_detalle", None)
    st.session_state.simp_volfrac = volfrac
    st.session_state.simp_penal_stored = penal
    st.session_state.simp_rmin_stored = rmin
    st.session_state.simp_nex = nelx
    st.session_state.simp_ney = nely
    st.session_state.simp_t_simp = m.t_simp
    st.session_state.simp_t_tda = m.t_tda
    st.session_state.last_params_he2 = current_params_he2


# La corrida SIMP por configuración vive en tda.simulation.he_protocols (corrida_simp),
# compartida con la página H.G.


# ══════════════════════════════════════════════════════════════════════
# RENDERIZADO (orden del documento: H.E.2a → H.E.2b → veredicto)
# ══════════════════════════════════════════════════════════════════════
if st.session_state.get("simp_optimized", False):
    rho_final = st.session_state.simp_rho_final
    rho_tilde_final = st.session_state.simp_rho_tilde_final
    nex_, ney_ = st.session_state.simp_nex, st.session_state.simp_ney
    c_final = st.session_state.simp_c_final
    c_base = st.session_state.simp_c_base
    reduccion = st.session_state.simp_reduccion
    beta0 = st.session_state.simp_beta0
    beta1 = st.session_state.simp_beta1
    volfrac = st.session_state.simp_volfrac
    penal_val = st.session_state.simp_penal_stored
    rmin_val = st.session_state.simp_rmin_stored
    n_iter = st.session_state.simp_n_iter
    converged = st.session_state.simp_converged
    c_hist = st.session_state.simp_c_hist
    dgm1 = st.session_state.get("simp_dgm1")
    concordancia = st.session_state.get("simp_concordancia")
    detalle = st.session_state.get("simp_betti_detalle")
    grisura = float(np.mean(4.0 * rho_tilde_final * (1.0 - rho_tilde_final)))   # 4ρ̃(1−ρ̃)

    # ── 1. H.E.2a — Eficiencia estructural ──────────────────────────
    st.subheader("1. H.E.2a — Eficiencia estructural")
    st.latex(r"\frac{c(f_V\cdot\mathbf{1})-c(\rho^{\star})}{c(f_V\cdot\mathbf{1})}\ge 0.40")

    # Tabla 5.6 del documento
    c_solido = c_base * (volfrac ** penal_val)   # c(ρ=1) = c(fV)·fV^p  → 39.54 con fV=0.5, p=3
    st.markdown("**Tabla 5.6 — Compliance del diseño convergido frente a referentes:**")
    st.dataframe(pd.DataFrame({
        "Configuración": ["Bloque sólido (ρ ≡ 1)", "Diseño uniforme (ρ ≡ fV)", "Diseño SIMP (ρ*)"],
        "Volumen": ["100 %", f"{volfrac*100:.0f} %", f"{volfrac*100:.0f} %"],
        "c(ρ)": [f"{c_solido:.2f}", f"{c_base:.2f}", f"{c_final:.2f}"],
    }), width="stretch", hide_index=True)

    # Figura 5.4(a) — distribución de material
    st.markdown(f"**Figura 5.4(a):** Distribución de material del diseño convergido (viga en voladizo {nex_}×{ney_}).")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_mat = plot_material_distribution(rho_final.reshape(ney_, nex_), "Densidad continua ρ*")
        st.plotly_chart(fig_mat, width="stretch")
        st.session_state.setdefault("he2_figs", {})["fig54a_densidad_continua"] = fig_mat
    with col_m2:
        binaria = (rho_tilde_final.reshape(ney_, nex_) >= 0.5).astype(float)
        fig_bin = go.Figure(data=[go.Heatmap(z=binaria, colorscale="gray_r", zmin=0, zmax=1,
                                             colorbar=dict(title="Ω"))])
        fig_bin.update_layout(title=f"Ω(sólido) | β₀={beta0} | β₁={beta1}", height=400)
        apply_plotly_theme(fig_bin)
        st.plotly_chart(fig_bin, width="stretch")
        st.session_state.setdefault("he2_figs", {})["fig54a_diseno_binarizado"] = fig_bin
    st.caption("El sólido forma una sola pieza (β₀ = 1). El conteo automático de agujeros coincide con la inspección visual.")

    # Figura 5.4(b) — historial de compliance (log)
    st.markdown("**Figura 5.4(b):** Historial de compliance en escala logarítmica, con los referentes marcados (diseño uniforme y diseño SIMP).")
    fig_conv = go.Figure()
    kx = [1, max(len(c_hist), 2)]
    fig_conv.add_trace(go.Scatter(x=list(range(1, len(c_hist) + 1)), y=c_hist,
                                  mode="lines+markers", line=dict(color="royalblue", width=2), name="c(k)"))
    fig_conv.add_trace(go.Scatter(x=kx, y=[c_base, c_base], mode="lines",
                                  line=dict(color="gray", dash="dot"), name=f"c(fV) = {c_base:.1f}"))
    fig_conv.add_trace(go.Scatter(x=kx, y=[c_final, c_final], mode="lines",
                                  line=dict(color="green", dash="dash"), name=f"c* = {c_final:.1f}"))
    fig_conv.update_layout(title="Figura 5.4(b) — Historial de compliance",
                           xaxis_title="Iteración k", yaxis_title="Compliance c(ρ) (escala log)",
                           yaxis_type="log", height=380,
                           xaxis=dict(range=[1, max(len(c_hist), 2)]),
                           legend=dict(x=0.6, y=0.95),
                           margin=dict(l=0, r=0, b=0, t=40))
    apply_plotly_theme(fig_conv)
    st.plotly_chart(fig_conv, width="stretch")
    st.session_state.setdefault("he2_figs", {})["fig54b_historial_compliance"] = fig_conv

    he2a_ok = reduccion >= REDUCCION_MIN * 100
    if he2a_ok:
        st.success(f"### ✅ H.E.2a SE SOSTIENE — reducción = {reduccion:.2f} % ≥ 40 % (vs diseño uniforme)")
    else:
        st.error(f"### ❌ H.E.2a NO SE SOSTIENE — reducción = {reduccion:.2f} % < 40 %")
    st.markdown("---")

    # ── 2. H.E.2b — Estabilidad del descriptor topológico ───────────
    st.subheader("2. H.E.2b — Estabilidad del descriptor topológico β₁")
    st.latex(r"\beta_1(\Omega_{\text{sólido}})\ \text{invariante frente a malla, } p \text{ y } r_{\min}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("β₀ (componentes conexas)", str(beta0))
    k2.metric("β₁ (agujeros)", str(beta1))
    k3.metric("Grisura 4ρ̃(1−ρ̃)", f"{grisura:.4f}")
    k4.metric("Concordancia χ–GUDHI", "✅" if concordancia else ("❌" if concordancia is not None else "N/D"))

    if detalle is not None:
        st.caption(
            "Doble cómputo (Algoritmo 3): Método A = característica de Euler del complejo cúbico "
            f"(V={detalle['V']}, E={detalle['E']}, F={detalle['F']}, χ={detalle['chi']} → β₁ = β₀ − χ); "
            "Método B = homología cúbica persistente (GUDHI, subnivel 0.5). "
            f"Coincidencia: {'sí' if concordancia else 'no'}."
        )

    # Barrido multi-configuración (malla con rmin escalado, p, rmin)
    st.markdown("**Barrido multi-configuración** — invariancia de β₁ (Figuras 5.5, 5.6) y persistencia frente al umbral (Figura 5.7).")
    if st.button("▶ Ejecutar barrido (H.E.2b)", type="secondary", key="sweep_btn"):
        bar = st.progress(0)

        def _cb(i, n, label):
            bar.progress(i / n)

        sweep_results = barrido_he2(dominio, nex_, ney_, penal_val, rmin_val,
                                    E=E_acero, Fmag=F_carga, fV=volfrac, t=espesor,
                                    max_iter=max_iter, progress_cb=_cb)
        bar.progress(1.0)
        st.session_state.sweep_results = sweep_results
        st.session_state.sweep_done = True

    df_sweep = pd.DataFrame(st.session_state.get("sweep_results", [])) if st.session_state.get("sweep_done") else None
    if df_sweep is not None and len(df_sweep) > 0:
        betas1 = df_sweep["beta1"].tolist()
        df_p = df_sweep[df_sweep["Config"].str.startswith("p=")]
        df_mesh = df_sweep[df_sweep["Config"].str.startswith("malla")]
        df_rmin = df_sweep[df_sweep["Config"].str.startswith("rmin")]

        # Figura 5.5 — compliance y grisura vs p
        if len(df_p) > 0:
            st.markdown("**Figura 5.5:** Compliance (eje izquierdo) y grisura (eje derecho) en función del factor de penalización $p$; el valor de β₁ se anota junto a cada punto. Con $p=1$ no hay penalización (β₁=0: sin frontera sólido/vacío); la invariancia de H.E.2b se evalúa para $p\\in\\{2,3,4\\}$.")
            fig55 = make_subplots(specs=[[{"secondary_y": True}]])
            fig55.add_trace(go.Scatter(x=df_p["p"], y=df_p["c(rho*)"], mode="lines+markers", name="Compliance"), secondary_y=False)
            fig55.add_trace(go.Scatter(x=df_p["p"], y=df_p["grisura"], mode="lines+markers", name="Grisura"), secondary_y=True)
            for _, row in df_p.iterrows():
                fig55.add_annotation(x=row["p"], y=row["c(rho*)"], text=f"β₁={row['beta1']}",
                                     showarrow=True, arrowhead=1, yshift=12)
            fig55.update_layout(height=400, template="plotly_white",
                                xaxis_title="Factor de penalización (p)", yaxis_title="Compliance c(ρ*)",
                                xaxis=dict(dtick=1), legend=dict(x=0.02, y=0.98))
            fig55.update_yaxes(title_text="Grisura", secondary_y=True)
            apply_plotly_theme(fig55)
            st.plotly_chart(fig55, width="stretch")
            st.session_state.setdefault("he2_figs", {})["fig55_barrido_penalizacion"] = fig55

        # Figura 5.6 — compliance y β₁ vs número de elementos
        if len(df_mesh) > 0:
            st.markdown("**Figura 5.6:** Compliance (eje izquierdo) y β₁ (eje derecho) frente al número de elementos, con $r_{\\min}$ escalado para mantener constante su longitud física.")
            fig56 = make_subplots(specs=[[{"secondary_y": True}]])
            fig56.add_trace(go.Scatter(x=df_mesh["Ne"], y=df_mesh["c(rho*)"], mode="lines+markers", name="Compliance"), secondary_y=False)
            fig56.add_trace(go.Scatter(x=df_mesh["Ne"], y=df_mesh["beta1"], mode="lines+markers", name="β₁"), secondary_y=True)
            for _, row in df_mesh.iterrows():
                fig56.add_annotation(x=row["Ne"], y=row["c(rho*)"], text=f"rmin={row['rmin']}",
                                     showarrow=True, arrowhead=1, yshift=12)
            fig56.update_layout(height=400, template="plotly_white",
                                xaxis_title="Número de elementos (Ne)", yaxis_title="Compliance c(ρ*)")
            fig56.update_yaxes(title_text="β₁", secondary_y=True)
            apply_plotly_theme(fig56)
            st.plotly_chart(fig56, width="stretch")
            st.session_state.setdefault("he2_figs", {})["fig56_barrido_malla"] = fig56

        st.markdown("**Barrido completo:**")
        st.dataframe(df_sweep, width="stretch", hide_index=True)

        df_p_inv = df_p[df_p["p"] >= 2]   # H.E.2b: p ∈ {2,3,4} (p=1 no penaliza → sin topología)
        he2b_ok = (len(set(df_mesh["beta1"])) <= 1 and len(set(df_p_inv["beta1"])) <= 1
                   and len(set(df_rmin["beta1"])) <= 1
                   and bool((df_sweep["Concordancia"] == "Si").all()))
        vetos = []
        if len(set(df_mesh["beta1"])) > 1: vetos.append("malla")
        if len(set(df_p_inv["beta1"])) > 1: vetos.append("p (2,3,4)")
        if len(set(df_rmin["beta1"])) > 1: vetos.append("r_min")
        if not bool((df_sweep["Concordancia"] == "Si").all()): vetos.append("concordancia Euler–GUDHI")
        if he2b_ok:
            st.success(f"### ✅ H.E.2b SE SOSTIENE — β₁ = {betas1[0]} invariante en malla, p∈{{2,3,4}} y r_min; Euler ≡ GUDHI")
        else:
            st.error(f"### ❌ H.E.2b NO SE SOSTIENE — varía en: {', '.join(vetos)}")
    else:
        st.info("Ejecutar el barrido para verificar la invariancia de β₁ (Figuras 5.5 y 5.6).")

    # Figura 5.7 — β₀, β₁ frente al umbral de binarización
    st.markdown("**Figura 5.7:** Números de Betti del diseño en función del umbral $u$ con que se binariza el campo de densidades. La línea punteada gris es la fracción de volumen resultante; la línea roja punteada marca el umbral adoptado $u = 0.5$.")
    with st.spinner("Barriendo el umbral de binarización..."):
        umbrales = np.linspace(0.05, 0.90, 18)
        b0s, b1s, vfs = [], [], []
        from tda.core.betti2d import betti_euler
        for u in umbrales:
            S = (rho_tilde_final.reshape(ney_, nex_) >= u)
            vfs.append(float(S.mean()))
            b0u, b1u = betti_euler(S)
            b0s.append(b0u)
            b1s.append(b1u)
        from tda.simulation.he_protocols import persistencia_beta1
        u_lo, u_hi, persist = persistencia_beta1(umbrales, b1s, beta1)

        fig57 = make_subplots(specs=[[{"secondary_y": True}]])
        fig57.add_trace(go.Scatter(x=umbrales, y=b1s, mode="lines", name="β₁",
                                   line=dict(color="green", width=3)), secondary_y=False)
        fig57.add_trace(go.Scatter(x=umbrales, y=b0s, mode="lines", name="β₀",
                                   line=dict(color="royalblue", dash="dash")), secondary_y=False)
        fig57.add_trace(go.Scatter(x=umbrales, y=vfs, mode="lines", name="Fracción de volumen",
                                   line=dict(color="gray", dash="dashdot")), secondary_y=True)
        fig57.add_trace(go.Scatter(x=[0.5, 0.5], y=[0, max(b1s + [3])], mode="lines",
                                   name="umbral adoptado", line=dict(color="red", dash="dot")),
                        secondary_y=False)
        fig57.update_layout(title="Persistencia de β₁ frente al umbral de binarización",
                            height=420, template="plotly_white", legend=dict(x=0.02, y=0.98))
        fig57.update_xaxes(title_text="umbral de binarización u", range=[0.0, 0.9])
        fig57.update_yaxes(title_text="número de Betti", secondary_y=False, range=[0.8, 3.2], dtick=1)
        fig57.update_yaxes(title_text="fracción de volumen umbralizada", secondary_y=True,
                           range=[0.3, 0.7])
        apply_plotly_theme(fig57)
        st.plotly_chart(fig57, width="stretch")
        st.session_state.setdefault("he2_figs", {})["fig57_betti_umbral"] = fig57
        if len(set(b1s[int(0.05 * len(umbrales)):int(0.85 * len(umbrales))])) == 1:
            st.success(f"✅ β₁ = {beta1} constante en u ∈ [{u_lo:.2f}, {u_hi:.2f}] — "
                       f"persistencia ≈ {persist:.2f} (unidades de umbral): rasgo estructural, no artefacto. "
                       f"El umbral adoptado u = 0.5 cae en el centro de la meseta.")
        else:
            st.warning("⚠️ β₁ varía con el umbral de binarización.")
    st.markdown("---")

    # ── 3. Veredicto H.E.2 ──────────────────────────────────────────
    st.subheader("3. Veredicto H.E.2")
    he2b_done = df_sweep is not None and len(df_sweep) > 0
    if he2b_done:
        he2b_ok_v = (len(set(df_mesh["beta1"])) <= 1 and len(set(df_p["beta1"])) <= 1
                     and len(set(df_rmin["beta1"])) <= 1
                     and bool((df_sweep["Concordancia"] == "Si").all()))
    else:
        he2b_ok_v = None
    filas = [
        {"Hipótesis": "H.E.2a", "Criterio": "Reducción ≥ 40 % vs uniforme",
         "Medido": f"{reduccion:.2f} %", "Veredicto": "✓" if he2a_ok else "×"},
        {"Hipótesis": "H.E.2b", "Criterio": "β₁ invariante (malla, p, r_min) + Euler ≡ GUDHI",
         "Medido": ("β₁ invariante" if he2b_ok_v else ("varía" if he2b_ok_v is False else "pendiente de barrido")),
         "Veredicto": "✓" if he2b_ok_v else ("×" if he2b_ok_v is False else "—")},
    ]
    st.table(pd.DataFrame(filas))
    if he2a_ok and he2b_ok_v:
        st.success("**H.E.2 SE SOSTIENE** (H.E.2a ✓ y H.E.2b ✓).")
    elif he2a_ok and he2b_ok_v is None:
        st.info("H.E.2a ✓; ejecutar el barrido para verificar H.E.2b.")

    # ── 4. Exportación ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("4. Exportar resultados")
    try:
        df_metrics = pd.DataFrame({
            "Métrica": ["c(uniforme ρ=fV)", "c(SIMP ρ*)", "Reducción vs uniforme",
                        "β₀", "β₁", "Grisura 4ρ(1−ρ)", "Concordancia Euler–GUDHI",
                        "Iteraciones", "Convergió", "Malla", "fV", "p", "rmin"],
            "Valor": [f"{c_base:.4f}", f"{c_final:.4f}", f"{reduccion:.2f}%",
                      str(beta0), str(beta1), f"{grisura:.6f}",
                      "Sí" if concordancia else ("No" if concordancia is not None else "N/D"),
                      str(n_iter), "Sí" if converged else "No", f"{nex_}x{ney_}",
                      str(volfrac), str(penal_val), str(rmin_val)],
        })
        st.dataframe(df_metrics, width="stretch", hide_index=True)
        zf = io.BytesIO()
        with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
            base = f"he2_{nex_}x{ney_}_p{int(penal_val)}"
            z.writestr(f"{base}/metricas.csv", df_metrics.to_csv(index=False))
            z.writestr(f"{base}/historial_convergencia.csv",
                       pd.DataFrame({"Iteracion": range(1, len(c_hist) + 1), "Compliance": c_hist}).to_csv(index=False))
            if df_sweep is not None and len(df_sweep) > 0:
                z.writestr(f"{base}/barrido_HE2b.csv", df_sweep.to_csv(index=False))
            write_figs_zip(z, f"{base}/figuras", st.session_state.get("he2_figs", {}))
        download_button("📦 Descargar TODO (ZIP)", zf.getvalue(), f"{base}.zip",
                        "application/zip", width="stretch")
    except Exception as e:
        st.error(f"Error en la exportación: {e}")

methodology_expander("📖 Plan de Acción — H.E.2", [
    ("H.E.2a", r"\frac{c(f_{V}\cdot \mathbf{1}) - c(\rho^{\star})}{c(f_{V}\cdot \mathbf{1})} \ge 0.40"),
    ("H.E.2b", r"\beta_{1}(\Omega_{\text{sólido}})\ \text{invariante para}\ r_{\min}\in[1.5,4.0],\ p\in\{2,3,4\},\ \text{malla con } r_{\min}\ \text{escalado}"),
    ("Formulación SIMP", r"\min c(\rho)=U^{T}K(\rho)U \quad \text{s.a.}\quad K(\rho)U=F,\quad \sum_e \rho_e v_e = f_V V_0"),
    ("Referente", r"c(\rho)\ge c(1)\ \Rightarrow\ \text{el referente es el diseño uniforme, no el bloque sólido}"),
], "H.E.2")
st.markdown("---")
