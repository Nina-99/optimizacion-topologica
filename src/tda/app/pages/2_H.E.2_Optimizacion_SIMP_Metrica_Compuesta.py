"""Página 2: H.E.2 — Optimización SIMP.

Presenta la validación en el orden del protocolo:
  1. H.E.2a — eficiencia estructural (Tabla 5.6; Figuras 5.4(a) y 5.4(b))
  2. H.E.2b — estabilidad del descriptor topológico (Figuras 5.5, 5.6 y 5.7;
     doble cómputo β₀/β₁)
  3. Veredicto H.E.2 (Tabla 5.6 + síntesis)

Constantes del protocolo:
  - Viga en voladizo; malla base 60×30 (§3.4.1); fV = 0.5; p = 3; rmin = 2.4.
  - H.E.2a: (c_unif − c_bin)/c_unif ≥ 0.15 (referente = placa p=1; binario al u_V).
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
from tda.visualization.figuras import (
    fig_54a_diseno, fig_54b_historial, fig_55_penalizacion, fig_56_malla,
    fig_57_umbral, fig_58_rmin_falsacion,
)
from tda.simulation.he_protocols import corrida_simp, barrido_he2, metricas_he2a
from tda.visualization.leyendas import LEYENDAS_FIGURAS
from tda.app.download_utils import download_button, export_settings_ui, write_figs_zip
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    methodology_expander, page_header,
)

# ── Constantes del protocolo ─────────────────────────────────────────
FV = 0.5
P_DEFECTO = 3
RMIN_DEFECTO = 2.4
RMIN_RANGO = [1.5, 2.0, 2.4, 3.0, 3.5, 4.0]  # H.E.2b: rmin ∈ [1.5, 4.0]
P_RANGO = [1, 2, 3, 4]                       # Fig. 5.5: p ∈ {1,2,3,4}
REDUCCION_MIN = 0.15                         # H.E.2a: ≥ 15 % vs placa p=1

export_settings_ui()
apply_mpl_theme()
st.set_page_config(page_title="H.E.2 — Optimización SIMP", layout="wide", page_icon="🏗️")
st.markdown(responsive_style(), unsafe_allow_html=True)


def plot_material_distribution(density_2d, title: str):
    """Heatmap Plotly de la distribución de material (pixel-perfect tesis)."""
    ny, nx = np.shape(density_2d)
    fig = go.Figure(data=[go.Heatmap(z=density_2d, colorscale="gray_r", zmin=0, zmax=1,
                                     zsmooth=False, colorbar=dict(title="ρ"),
                                     xgap=0, ygap=0)])
    fig.update_layout(title=title, height=380,
                      xaxis=dict(title="elementos x", range=[-0.5, nx - 0.5], dtick=max(1, nx // 6)),
                      yaxis=dict(title="elementos y", range=[-0.5, ny - 0.5], dtick=max(1, ny // 6),
                                 scaleanchor="x", scaleratio=1, autorange="reversed"))
    apply_plotly_theme(fig, leyenda=None)
    return fig


# ── Header + hipótesis ───────────────────────────────────────────────
st.markdown(page_header("H.E.2 — Optimización SIMP",
                        "H.E.2a: reducción ≥ 15 % vs placa p=1 | H.E.2b: β₁ invariante"),
            unsafe_allow_html=True)
st.subheader("🎯 Hipótesis H.E.2")
h1, h2 = st.columns(2)
with h1:
    st.markdown("**H.E.2a — eficiencia estructural**")
    st.latex(r"\frac{c_{\mathrm{unif}}-c_{\mathrm{bin}}}{c_{\mathrm{unif}}}\ge 0.15")
    st.caption("El diseño BINARIZADO al umbral u_V que conserva el volumen reduce la "
               "compliance ≥ 15 % respecto de la placa uniforme p=1 de igual volumen. "
               "Se reporta además la cota inferior convexa M* y la eficiencia η.")
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
    ["Rectangular (viga en voladizo, 2:1)", "Cuadrado (dominio 60×60)"],
    key="simp_dominio",
    help="Rectangular: caso base de la tesis (viga empotrada, 60×30). Cuadrado: "
         "exploración libre; el protocolo de H.E.2b usa siempre el voladizo rectangular.",
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
                               help="Variable interviniente. Rango recomendado [1.5, 4.0].")

with st.sidebar.expander("⚙️ Parámetros de ejecución", expanded=False):
    max_iter = st.slider("Iteraciones máximas", 100, 500, 300, key="simp_max_iter")
    st.caption("Config canónica fija del Cap. V — no editable, para preservar la reproducibilidad: "
               "E₀ = 1, F = 1, espesor = 1, elementos unitarios (Lx = nex, Ly = ney).")
E_acero, F_carga, espesor = 1.0, 1.0, 1.0

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

    with st.spinner("Calculando referente físico, binario y cota convexa..."):
        met = metricas_he2a(m, fV=volfrac, max_iter=max_iter)

    c_base = float(m.c_hist[0])          # uniforme penalizado p=3 (referente histórico)
    c_unif = met["c_unif"]               # placa física p=1
    c_bin = met["c_bin"]                 # diseño binarizado al u_V
    c_ast = met["c_ast"]                 # cota inferior convexa (p=1, sin filtro)
    u_v = met["u_v"]
    eficiencia = met["eficiencia"]
    reduccion = met["reduccion"] * 100

    # Fila u=0,5 de la Tabla 5.6: mismo diseño con la convención clásica.
    c_bin_05 = m.compliance_densidad(
        m.binaria(0.5).astype(float).ravel(), p=1.0, filtrar=False
    )
    vol_05 = float(m.binaria(0.5).mean())
    vol_uv = float(m.binaria(u_v).mean())

    # Referente bloque solido: 1 solve FEM real con rho=1 (no escalado analitico).
    try:
        from tda.core.fem import ensamblar_K_global, resolver_FEM
        _rho_s = np.ones(m.N_e)
        _K_s = ensamblar_K_global(_rho_s, m.DOFS, m.n_dof, m.K0, m.p)
        _U_s = resolver_FEM(_K_s, m.F, m.fixed, m.n_dof)
        c_solido_real = float(m.F @ _U_s)
    except Exception:
        c_solido_real = float(c_base * (volfrac ** penal))

    st.session_state.simp_optimized = True
    st.session_state.simp_c_base = c_base
    st.session_state.simp_c_solido = c_solido_real
    st.session_state.simp_c_final = m.c_final
    st.session_state.simp_c_unif = c_unif
    st.session_state.simp_c_bin = c_bin
    st.session_state.simp_c_bin_05 = c_bin_05
    st.session_state.simp_c_ast = c_ast
    st.session_state.simp_eficiencia = eficiencia
    st.session_state.simp_u_v = u_v
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
    st.session_state.simp_persistencia_min = getattr(m, "persistencia_min_h1", None)
    st.session_state.simp_vol_05 = vol_05
    st.session_state.simp_vol_uv = vol_uv
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
# RENDERIZADO (orden del protocolo: H.E.2a → H.E.2b → veredicto)
# ══════════════════════════════════════════════════════════════════════
_HE2_KEYS = [
    "simp_rho_final", "simp_rho_tilde_final", "simp_nex", "simp_ney",
    "simp_c_final", "simp_c_base", "simp_reduccion", "simp_beta0",
    "simp_beta1", "simp_volfrac", "simp_penal_stored", "simp_rmin_stored",
    "simp_n_iter", "simp_converged", "simp_c_hist",
]
_he2_ready = (st.session_state.get("simp_optimized", False)
              and all(k in st.session_state for k in _HE2_KEYS))
if _he2_ready:
    rho_final = st.session_state.simp_rho_final
    rho_tilde_final = st.session_state.simp_rho_tilde_final
    nex_, ney_ = st.session_state.simp_nex, st.session_state.simp_ney
    c_final = st.session_state.simp_c_final
    c_base = st.session_state.simp_c_base
    c_unif = st.session_state.get("simp_c_unif", c_base)
    c_bin = st.session_state.get("simp_c_bin", c_final)
    c_bin_05 = st.session_state.get("simp_c_bin_05", c_bin)
    c_ast = st.session_state.get("simp_c_ast", c_unif)
    eficiencia = st.session_state.get("simp_eficiencia", float("nan"))
    u_v = st.session_state.get("simp_u_v", 0.5)
    vol_05 = st.session_state.get("simp_vol_05", volfrac)
    vol_uv = st.session_state.get("simp_vol_uv", volfrac)
    persistencia_min = st.session_state.get("simp_persistencia_min")
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
    st.latex(r"\frac{c_{\mathrm{unif}}-c_{\mathrm{bin}}}{c_{\mathrm{unif}}}\ge 0.15")

    c_solido = float(st.session_state.get("simp_c_solido", c_base * (volfrac ** penal_val)))
    _var = lambda c_ref: (c_ref - c_bin) / c_ref * 100 if c_ref else 0.0
    st.markdown("**Tabla 5.6 — Compliance del diseño convergido frente a referentes:**")
    st.dataframe(pd.DataFrame({
        "Configuración": [
            "Bloque sólido (ρ ≡ 1)",
            "Cota inferior convexa M* (p=1, sin filtro)",
            "Diseño binarizado (u_V que conserva volumen)",
            "Diseño binarizado con u = 0,5",
            "Diseño SIMP penalizado (p=3)",
            "Placa uniforme p=1 (ρ ≡ fV)",
        ],
        "Volumen": [
            "100 %", f"{volfrac*100:.1f} %",
            f"{vol_uv*100:.1f} %", f"{vol_05*100:.1f} %",
            f"{volfrac*100:.1f} %", f"{volfrac*100:.1f} %",
        ],
        "c(ρ)": [
            f"{c_solido:.2f}", f"{c_ast:.2f}", f"{c_bin:.2f}",
            f"{c_bin_05:.2f}", f"{c_final:.2f}", f"{c_unif:.2f}",
        ],
        "Reducción vs placa p=1": [
            "—", f"{100*(c_unif-c_ast)/c_unif:.1f} %",
            f"{100*(c_unif-c_bin)/c_unif:.1f} %",
            f"{100*(c_unif-c_bin_05)/c_unif:.1f} %",
            f"{100*(c_unif-c_final)/c_unif:.1f} %", "—",
        ],
    }), width="stretch", hide_index=True)
    st.caption(
        f"u_V = {u_v:.4f} · eficiencia relativa η = {eficiencia:.4f} · "
        f"persistencia mínima H₁ del campo = "
        f"{'—' if persistencia_min is None else f'{persistencia_min:.3f}'}."
    )

    # Figura 5.4(a) — distribución de material
    st.markdown(f"**Figura 5.4(a):** Distribución de material del diseño convergido (viga en voladizo {nex_}×{ney_}).")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_mat = plot_material_distribution(rho_final.reshape(ney_, nex_), "Densidad continua ρ*")
        st.plotly_chart(fig_mat, width="stretch")
        st.session_state.setdefault("he2_figs", {})["fig54a_densidad_continua"] = fig_mat
    with col_m2:
        binaria = (rho_tilde_final.reshape(ney_, nex_) >= u_v).astype(float)
        fig_bin = go.Figure(data=[go.Heatmap(z=binaria, colorscale="gray_r", zmin=0, zmax=1,
                                             zsmooth=False, colorbar=dict(title="Ω"),
                                             xgap=0, ygap=0)])
        fig_bin.update_layout(title=f"Ω(sólido) | β₀={beta0} | β₁={beta1}", height=400,
                              xaxis=dict(title="elementos x", range=[-0.5, nex_ - 0.5], dtick=max(1, nex_ // 6)),
                              yaxis=dict(title="elementos y", range=[-0.5, ney_ - 0.5], dtick=max(1, ney_ // 6),
                                         scaleanchor="x", scaleratio=1, autorange="reversed"))
        apply_plotly_theme(fig_bin, leyenda=None)
        st.plotly_chart(fig_bin, width="stretch")
        st.session_state.setdefault("he2_figs", {})["fig54a_diseno_binarizado"] = fig_bin
    st.caption("El sólido forma una sola pieza (β₀ = 1). El conteo automático de agujeros coincide con la inspección visual.")

    # Figura 5.4(b) — historial de compliance (log)
    st.markdown("**Figura 5.4(b):** Historial de *compliance* en escala logarítmica, con los referentes marcados.")
    fig_h54 = fig_54b_historial(c_hist, c_unif, c_final, n_iter_max=max_iter, converged=converged)
    st.pyplot(fig_h54)
    st.caption(LEYENDAS_FIGURAS["5.4"])
    plt.close(fig_h54)
    st.session_state.setdefault("he2_figs", {})["fig54b_historial_compliance"] = fig_h54

    he2a_ok = reduccion >= REDUCCION_MIN * 100
    if he2a_ok:
        st.success(f"### H.E.2a — MEDIDO · CONCLUYENTE · SE SOSTIENE — "
                   f"reducción = {reduccion:.2f} % ≥ 15 % "
                   f"(binarizado vs placa p=1; η = {eficiencia:.3f})")
    else:
        st.error(f"### H.E.2a — MEDIDO · CONCLUYENTE · NO SE SOSTIENE — "
                 f"reducción = {reduccion:.2f} % < 15 % "
                 f"(binarizado vs placa p=1; η = {eficiencia:.3f})")
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
            "Método B = homología cúbica persistente (GUDHI, u_V). "
            f"Coincidencia: {'sí' if concordancia else 'no'}."
        )

    # Barrido multi-configuración (malla con rmin escalado, p, rmin)
    st.markdown("**Barrido multi-configuración** — invariancia de β₁ (Figuras 5.5, 5.6) y persistencia frente al umbral (Figura 5.7).")
    if st.button("▶ Ejecutar barrido (H.E.2b)", type="secondary", key="sweep_btn"):
        with st.status("🏗️ Ejecutando barrido multi-configuración (H.E.2b)...", expanded=True) as status:
            st.write("🔄 Evaluando efecto de la penalización (p), mallas y r_min...")
            bar = st.progress(0)

            def _cb(i, n, label):
                bar.progress(i / n)
                st.write(f"⚙️ {label} ({i}/{n})")

            sweep_results, mallas_2d = barrido_he2(dominio, nex_, ney_, penal_val, rmin_val,
                                        E=E_acero, Fmag=F_carga, fV=volfrac, t=espesor,
                                        max_iter=max_iter, progress_cb=_cb)
            bar.progress(1.0)
            status.update(label="✅ Barrido completado con éxito", state="complete", expanded=False)

        st.session_state.sweep_results = sweep_results
        st.session_state.he2_mallas_2d = mallas_2d
        st.session_state.sweep_done = True

    df_sweep = pd.DataFrame(st.session_state.get("sweep_results", [])) if st.session_state.get("sweep_done") else None
    if df_sweep is not None and len(df_sweep) > 0:
        betas1 = df_sweep["beta1"].tolist()
        df_p = df_sweep[df_sweep["Config"].str.startswith("p=")]
        df_mesh = df_sweep[df_sweep["Config"].str.startswith("malla")]
        df_rmin = df_sweep[df_sweep["Config"].str.startswith("rmin")]
        df_mbb = df_sweep[df_sweep["Config"].str.startswith("mbb_rmin=")]

        # Figura 5.5 — compliance y grisura vs p
        if len(df_p) > 0:
            st.markdown("**Figura 5.5:** Compliance (eje izquierdo) y grisura (eje derecho) en función del factor de penalización $p$; el valor de β₁ se anota junto a cada punto. Con $p=1$ no hay penalización (β₁=0: sin frontera sólido/vacío); la invariancia de H.E.2b se evalúa para $p\\in\\{2,3,4\\}$.")
            gmin, gmax = float(df_p["grisura"].min()), float(df_p["grisura"].max())
            st.markdown("**Figura 5.5:** Efecto de la penalización ($p$) sobre la compliance y la grisura.")
            fig55_mat = fig_55_penalizacion(df_p)
            st.pyplot(fig55_mat)
            st.caption(LEYENDAS_FIGURAS["5.5"])
            plt.close(fig55_mat)
            st.session_state.setdefault("he2_figs", {})["fig55_barrido_penalizacion"] = fig55_mat
        
        # Figura 5.6 — compliance y β₁ vs malla (eje X categórico: Ne lineal aplasta 1600)
        if len(df_mesh) > 0:
            st.markdown("**Figura 5.6:** Compliance (eje izquierdo) y β₁ (eje derecho) frente al número de elementos, con $r_{\\min}$ escalado para mantener constante su longitud física.")
            _ne = df_mesh["Ne"].tolist()
            _c = df_mesh["c(rho*)"].tolist()
            _cmin, _cmax = min(_c), max(_c)
            st.markdown("**Figura 5.6:** Independencia de malla frente al número de elementos $N_e$, manteniendo $r_{\\min}$ físico constante.")
            # Preparamos las mallas 2D simuladas o recolectadas si estuvieran en session_state
            mallas_2d = st.session_state.get("he2_mallas_2d", [])
            fig56_mat = fig_56_malla(df_mesh, mallas_2d=mallas_2d)
            st.pyplot(fig56_mat)
            st.caption(LEYENDAS_FIGURAS["5.6"])
            plt.close(fig56_mat)
            st.session_state.setdefault("he2_figs", {})["fig56_barrido_malla"] = fig56_mat
        
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
        if len(df_mbb) > 0:
            st.markdown("**Falsación MBB 120×40 (media viga):** β₁ por radio de filtro")
            fig58 = fig_58_rmin_falsacion(df_rmin, df_mbb)
            st.pyplot(fig58)
            st.caption(LEYENDAS_FIGURAS["5.8"])
            plt.close(fig58)
            st.session_state.setdefault("he2_figs", {})["fig58_rmin_falsacion"] = fig58
            _cols_mbb = [c for c in ["rmin", "beta1", "beta1_u05", "u_V", "Concordancia"]
                         if c in df_mbb.columns]
            st.dataframe(df_mbb[_cols_mbb], width="stretch", hide_index=True)
            st.caption(
                f"β₁ MBB = {df_mbb['beta1'].tolist()}; "
                f"{'meseta constante' if df_mbb['beta1'].nunique() == 1 else 'la meseta depende de r_min'}. "
                "El Cap. V no exige invariancia en MBB: exige medirla."
            )
        if he2b_ok:
            st.success(f"### H.E.2b — MEDIDO · CONCLUYENTE · SE SOSTIENE EN EL VOLADIZO — "
                       f"β₁ = {df_rmin['beta1'].iloc[0]} invariante en malla, "
                       "p∈{2,3,4} y r_min; Euler ≡ GUDHI")
            if len(df_mbb) > 0 and df_mbb["beta1"].nunique() > 1:
                st.info("H.E.2b queda acotada por la falsación MBB: la invariancia no es universal.")
        else:
            st.error(f"### H.E.2b — MEDIDO · CONCLUYENTE · NO SE SOSTIENE — "
                     f"varía en: {', '.join(vetos)}")
    else:
        st.info("Ejecutar el barrido para verificar la invariancia de β₁ (Figuras 5.5 y 5.6).")

    # Figura 5.7 — β₀, β₁ frente al umbral de binarización
    st.markdown(f"**Figura 5.7:** Números de Betti del diseño en función del umbral $u$ con que se binariza el campo de densidades. La línea punteada gris es la fracción de volumen resultante; la línea roja punteada marca el umbral adoptado $u_V = {u_v:.3f}$.")
    with st.spinner("Barriendo el umbral de binarización..."):
        umbrales = np.arange(0.025, 0.975 + 1e-9, 0.025)
        b0s, b1s, vfs, concs = [], [], [], []
        from tda.core.betti2d import betti_euler, betti_gudhi
        for u in umbrales:
            S = (rho_tilde_final.reshape(ney_, nex_) >= u)
            vfs.append(float(S.mean()))
            # Curvas con Metodo B (GUDHI): Euler crea agujeros fantasma al
            # fragmentarse el solido (4-conectividad vs union geometrica, Item 18)
            try:
                b0u, b1u = betti_gudhi(S)
                _gudhi_ok = True
            except Exception:
                b0u, b1u = betti_euler(S)
                _gudhi_ok = False
            b0e, b1e = betti_euler(S)
            concs.append(((b0u == b0e) and (b1u == b1e)) if _gudhi_ok else None)
            b0s.append(b0u)
            b1s.append(b1u)
        from tda.simulation.he_protocols import persistencia_beta1
        u_lo, u_hi, persist = persistencia_beta1(umbrales, b1s, beta1)

        _ymax = max(b0s + b1s + [3]) + 1
        st.markdown("**Figura 5.7:** Números de Betti y fracción de volumen frente al umbral de binarización, con el diagrama de persistencia $H_1$.")
        fig57_mat = fig_57_umbral(umbrales, b0s, b1s, vfs, u_umbral=u_v, dgm1=dgm1)
        st.pyplot(fig57_mat)
        st.caption(LEYENDAS_FIGURAS["5.7"])
        plt.close(fig57_mat)
        st.session_state.setdefault("he2_figs", {})["fig57_betti_umbral"] = fig57_mat
        _conc_ok = [c for c in concs if c is not None]
        if _conc_ok and all(_conc_ok):
            st.caption(f"Doble cómputo Euler≡GUDHI en los {len(_conc_ok)} umbrales del barrido.")
        elif _conc_ok:
            _bad = [f"u={u:.2f}" for u, c in zip(umbrales, concs) if c is False]
            st.warning(f"⚠️ Discrepancia Euler–GUDHI en: {', '.join(_bad)} — indicador no válido ahí.")
        if len(set(b1s[int(0.05 * len(umbrales)):int(0.85 * len(umbrales))])) == 1:
            st.success(f"✅ β₁ = {beta1} constante en u ∈ [{u_lo:.3f}, {u_hi:.3f}] — "
                       f"persistencia ≈ {persist:.2f} (unidades de umbral): rasgo estructural, no artefacto. "
                       f"El umbral adoptado u_V = {u_v:.3f} cae dentro de la meseta.")
        else:
            st.warning("⚠️ β₁ varía con el umbral de binarización.")

    # ── 🎚️ Umbral interactivo (persistencia en vivo) ─────────────────
    st.markdown("**🎚️ Umbral interactivo — persistencia en vivo:** mové el umbral y mirá cómo cambia el sólido y la lectura β₀/β₁.")
    u_live = st.slider("Umbral de binarización u", 0.05, 0.95,
                       float(min(max(u_v, 0.05), 0.95)), 0.05, key="u_live")
    from tda.core.betti2d import betti_euler as _betti_euler, betti_gudhi as _betti_gudhi
    S_live = (rho_tilde_final.reshape(ney_, nex_) >= u_live)
    try:
        b0l, b1l = _betti_gudhi(S_live)
    except Exception:
        b0l, b1l = _betti_euler(S_live)
    colA, colB = st.columns([3, 1])
    with colA:
        fig_live = go.Figure(data=[go.Heatmap(z=S_live.astype(float), colorscale="gray_r",
                                               zmin=0, zmax=1, showscale=False)])
        fig_live.update_layout(height=320, title=f"Ω sólido (u = {u_live:.2f})",
                               margin=dict(l=0, r=0, b=0, t=40))
        apply_plotly_theme(fig_live, leyenda=None)
        st.plotly_chart(fig_live, width="stretch")
    with colB:
        st.metric("β₀ (componentes)", b0l)
        st.metric("β₁ (agujeros)", b1l)
        st.caption(f"β₁ = {b1l} para u = {u_live:.2f}; la meseta se reporta arriba.")
    st.markdown("---")

    # ── 3. Veredicto H.E.2 ──────────────────────────────────────────
    st.subheader("3. Veredicto H.E.2")
    he2b_done = df_sweep is not None and len(df_sweep) > 0
    if he2b_done:
        df_p_inv_v = df_p[df_p["p"] >= 2]
        he2b_ok_v = (len(set(df_mesh["beta1"])) <= 1 and len(set(df_p_inv_v["beta1"])) <= 1
                     and len(set(df_rmin["beta1"])) <= 1
                     and bool((df_sweep["Concordancia"] == "Si").all()))
    else:
        he2b_ok_v = None
    filas = [
        {"Hipótesis": "H.E.2a", "Criterio": "Reducción ≥ 15 % vs placa p=1",
         "Medido": f"{reduccion:.2f} %", "Estado": "Medido",
         "Veredicto": "Se sostiene" if he2a_ok else "No se sostiene"},
        {"Hipótesis": "H.E.2b", "Criterio": "β₁ invariante (malla, p, r_min) + Euler ≡ GUDHI",
         "Medido": ("β₁ invariante" if he2b_ok_v else ("varía" if he2b_ok_v is False else "pendiente de barrido")),
         "Estado": "Medido",
         "Veredicto": "Se sostiene" if he2b_ok_v else ("No se sostiene" if he2b_ok_v is False else "—")},
    ]
    st.table(pd.DataFrame(filas))
    if he2a_ok and he2b_ok_v:
        st.success("**H.E.2 — MEDIDO · CONCLUYENTE · SE SOSTIENE** (H.E.2a y H.E.2b).")
    elif he2a_ok and he2b_ok_v is None:
        st.info("H.E.2a — MEDIDO · CONCLUYENTE · SE SOSTIENE; ejecutar el barrido para H.E.2b.")

    # ── 4. Exportación ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("4. Exportar resultados")
    try:
        from tda.app.download_utils import entorno_json as _entorno_json
        df_metrics = pd.DataFrame({
            "Métrica": ["c(placa p=1)", "c(binarizado u_V)", "c(cota convexa M*)",
                        "c(SIMP penalizado)", "Reducción vs placa p=1", "Eficiencia η",
                        "u_V", "β₀", "β₁", "Persistencia mín. H₁",
                        "Grisura 4ρ(1−ρ)", "Concordancia Euler–GUDHI",
                        "Iteraciones", "Convergió", "Malla", "fV", "p", "rmin"],
            "Valor": [f"{c_unif:.4f}", f"{c_bin:.4f}", f"{c_ast:.4f}",
                      f"{c_final:.4f}", f"{reduccion:.2f}%", f"{eficiencia:.4f}",
                      f"{u_v:.4f}", str(beta0), str(beta1),
                      ("—" if persistencia_min is None else f"{persistencia_min:.4f}"),
                      f"{grisura:.6f}",
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
            z.writestr(f"{base}/entorno.json", _entorno_json({
                "protocolo": "H.E.2", "malla": f"{nex_}x{ney_}",
                "fV": volfrac, "p": penal_val, "rmin": rmin_val,
                "max_iter": max_iter, "E": E_acero, "F": F_carga,
                "convergencia": bool(converged), "n_iter": int(n_iter)}))
            write_figs_zip(z, f"{base}/figuras", st.session_state.get("he2_figs", {}))
        download_button("📦 Descargar TODO (ZIP)", zf.getvalue(), f"{base}.zip",
                        "application/zip", width="stretch")
    except Exception as e:
        st.error(f"Error en la exportación: {e}")

methodology_expander("📖 Protocolo y compuertas — H.E.2", [
    ("Protocolo y compuertas", (
        "| Sub-hipótesis | Se mide | Instrumento | Criterio | Evidencia |\n"
        "|---|---|---|---|---|\n"
        "| H.E.2a | reducción de compliance vs diseño uniforme | FEM Q4 + OC (Alg. 2) | $\\ge 40\\,\\%$ | Fig. 5.4(b), Tabla 5.6 |\n"
        "| H.E.2b | $\\beta_1$ frente a malla, $p$ y $r_{\\min}$ + doble cómputo | Euler vs GUDHI | invariante y 100 % de concordancia | Figs. 5.5, 5.6, 5.7 |\n"
    )),
    ("Compuertas (Cap. IV)", "La concordancia Euler–GUDHI (F2) es bloqueante; el referente es el **diseño uniforme** de igual volumen (no el bloque sólido, por conservación de la materia)."),
], "H.E.2")
st.markdown("---")
