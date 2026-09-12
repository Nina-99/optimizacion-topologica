"""Página 2: H.E.2 — Optimización SIMP + Métrica Compuesta.

Valida las hipótesis corregidas tras el estudio piloto (4.4):

H.E.2a' (CORREGIDA — 4.4 Hallazgo 2):
    (c(fV·1) − c(ρ⋆)) / c(fV·1) ≥ 0.40
    Reducción ≥ 40% frente al diseño uniforme de igual volumen.
    El referente original (bloque sólido) era lógicamente imposible
    por la Proposición 8.3: c(ρ) ≥ c(1) para todo ρ admisible.

H.E.2b' (CORREGIDA — 4.4 Hallazgo 3):
    β1(Ω_sólido) es independiente de la resolución de malla y del
    radio del filtro rmin ∈ [1.5, 4.0].
    Descriptor invariante (descriptivo, no prescriptivo).
    La cota original β1 ≤ 2 era prescriptiva sobre un invariante
    que el método no controla.

H.E.2a original (FALSIFICADA — reducción = −91.07%):
    Reducción ≥ 40% vs bloque sólido. Imposible por Prop. 8.3.

H.E.2b original (FALSIFICADA — β1 = 2 ≤ 2):
    β1 ≤ 2. Cota fijada a priori sobre invariante no controlado.

Referencias:
- Perfil 4.4: Estudio piloto y reformulación de hipótesis
- Perfil 8.3: Formulación SIMP, Proposición 8.3
- metodologia_implementacion.txt 2.3: El referente hace la hipótesis imposible
- metodologia_implementacion.txt 8: Resultados del Caso 2
"""

import streamlit as st
from tda.app.download_utils import download_button, export_settings_ui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import zipfile
from datetime import datetime


# Helper to create material distribution heatmap Plotly figure
def plot_material_distribution(density_2d, title: str):
    """Return a Plotly heatmap figure for material distribution.
    Applies the global Plotly theme before returning.
    """
    fig = go.Figure(data=[go.Heatmap(
        z=density_2d,
        colorscale='gray_r',
        zmin=0, zmax=1,
        colorbar=dict(title='rho')
    )])
    fig.update_layout(title=title)
    apply_plotly_theme(fig)
    return fig

from tda.optimization.metric_simp import MetricaTDA_SIMP
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme,
    responsive_style,
    methodology_expander, page_header
)

# ── Configuración de exportación (.exe) ──
export_settings_ui()

# Aplicar tema matplotlib global
apply_mpl_theme()

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="H.E.2 — Optimización SIMP + Métrica Compuesta", layout="wide", page_icon="🏗️")

st.markdown(responsive_style(), unsafe_allow_html=True)

# ── Page Header ──
st.markdown(page_header(
    "H.E.2 — Optimización SIMP + Métrica Compuesta μ_α",
    "H.E.2a': reduccion >=40% vs uniforme | H.E.2b': beta1 invariante"
), unsafe_allow_html=True)
# Intro subheader
st.subheader("Objetivo de Validacion -- Hipotesis H.E.2")
st.markdown("""
**H.E.2a':** La optimizacion SIMP con p=3 y f_V=0.5 reduce la compliance al menos un 40% respecto al diseno uniforme de igual volumen.

**H.E.2b':** El Betti 1 (Omega solido) es independiente de la resolucion de malla y del radio del filtro r_min en [1.5, 4.0].
""")

# ── Sidebar ──
st.sidebar.header("🏗️ Optimización SIMP")

# Dominio (determina geometría y mallas disponibles)
dominio = st.sidebar.radio(
    "Dominio",
    ["Rectangular (viga en voladizo, 2:1)", "Cuadrado (independencia de malla)"],
    key="simp_dominio",
    help="Rectangular: caso tesis (viga empotrada). "
         "Cuadrado: para verificar invariancia de malla (H.E.2b)."
)

# Mallas según dominio (Perfil4.txt 4.3, Cuadro 1)
if dominio.startswith("Rectangular"):
    opciones_malla = [
        "40x20 (800 elem)",
        "60x30 (1800 elem, Caso Tesis)",
        "80x40 (3200 elem)",
        "120x60 (7200 elem)",
    ]
    idx_default = 1
    Lx_val, Ly_val = 120.0, 40.0
else:
    # Cuadrado: Ne ∈ {1600, 6400, 14400} → 40², 80², 120²
    opciones_malla = [
        "40x40 (1600 elem)",
        "80x80 (6400 elem)",
        "120x120 (14400 elem)",
    ]
    idx_default = 0
    Lx_val, Ly_val = 120.0, 120.0

malla_opcion = st.sidebar.selectbox(
    "Resolución de Malla",
    opciones_malla,
    index=idx_default,
    key="simp_malla",
)

# fV fijo en 0.5 (Hipótesis H.E.2a': diseño uniforme de igual volumen)
st.sidebar.markdown("**f_V = 0.5** (fijo, diseño uniforme de igual volumen)")
volfrac = 0.5

# Penalización p: variable independiente (Cuadro 1: p ∈ {1, 2, 3, 4})
penal = st.sidebar.number_input(
    "Factor Penalización (p)",
    value=3.0, min_value=1.0, max_value=5.0, step=1.0,
    key="simp_penal",
    help="Variable independiente. p=3 es el estándar SIMP. "
         "El barrido H.E.2b prueba p ∈ {2, 3, 4}."
)

# Radio del filtro r_min: variable interveniente (Perfil4.txt: r_min ∈ [1.5, 4.0])
rmin = st.sidebar.number_input(
    "Radio Filtro (r_min, elementos)",
    value=2.4, min_value=1.0, max_value=5.0, step=0.1,
    key="simp_rmin",
    help="Variable interveniente. Rango válido: [1.5, 4.0] elementos. "
         "Default tesis: 2.4."
)

# Peso α (observer, no parte de la hipótesis)
with st.sidebar.expander("⚙️ Parámetros del Observer", expanded=False):
    alpha = st.number_input(
        "Peso α (métrica μ_α)",
        value=0.012, step=0.001, format="%.3f",
        key="simp_alpha",
        help="Peso del término topológico β1 en la métrica compuesta. "
             "No es parte de la hipótesis, es un parámetro del observer. "
             "Default tesis: 0.012."
    )
    max_iter = st.slider(
        "Iteraciones Máximas",
        min_value=100, max_value=500, value=200,
        key="simp_max_iter",
        help="Máximo de iteraciones SIMP. Default: 200."
    )

# Parámetros físicos fijos (no son variables de la hipótesis)
with st.sidebar.expander("📐 Parámetros Físicos (fijos)", expanded=False):
    E_acero = st.number_input(
        "Módulo Young E₀ (MPa)", value=200000.0, step=1000.0,
        help="Acero = 200 GPa = 200,000 MPa"
    )
    F_carga = st.number_input(
        "Carga F (N)", value=1000.0, step=100.0,
        help="Carga aplicada = 1 kN = 1000 N"
    )
    espesor = st.number_input(
        "Espesor (mm)", value=1.0, step=0.1,
        help="Espesor de la viga en 2D plano"
    )

# Track parameters for stale-data warning (HE.2 specific)
current_params_he2 = (dominio, malla_opcion, penal, rmin, alpha, max_iter)
if 'last_params_he2' not in st.session_state:
    st.session_state.last_params_he2 = current_params_he2
stale_data_he2 = st.session_state.last_params_he2 != current_params_he2

# Aviso de datos obsoletos para H.E.2
if stale_data_he2 and st.session_state.get('simp_optimized', False):
    st.warning("⚠️ Parámetros modificados – vuelva a ejecutar la optimización para actualizar los resultados.")


# ============================================================
# CUERPO PRINCIPAL
# ============================================================

ejecutar_simp = st.button("▶ Iniciar Optimización SIMP", type="primary")
st.markdown("---")

col_res1, col_res2 = st.columns([2, 1])
with col_res1:
    st.subheader("Evolución del Diseño")
    plot_ph = st.empty()
with col_res2:
    st.subheader("Métricas de Control")
    metric_ph = st.empty()

if ejecutar_simp:
    # Parsear malla
    dims = malla_opcion.split(" ")[0].split("x")
    nelx, nely = int(dims[0]), int(dims[1])

    history = []

    # Condiciones de contorno: viga en voladizo
    nnx = nelx + 1
    nny = nely + 1
    n_dof = 2 * nnx * nny

    # Empotramiento en borde izquierdo (x=0): todos los nodos columna 0
    dofs_fijos = np.arange(0, 2 * (nely + 1))

    # Carga puntual en centro del borde derecho: nodo (nelx, nely//2)
    node_load = (nely // 2) * nnx + nelx
    F = np.zeros(n_dof)
    F[2 * node_load + 1] = -F_carga

    # Callback for Streamlit – using top-level plot_material_distribution helper (defined earlier)
    # Instancia del optimizador MetricaTDA_SIMP con dimensiones físicas reales
    m = MetricaTDA_SIMP(
        nex=nelx, ney=nely, E=E_acero, nu=0.3,
        Lx=Lx_val, Ly=Ly_val, t=espesor,
        f_V=volfrac, p=penal, r_min=rmin, alpha=alpha, max_iter=max_iter
    )

    m.definir_problema(F, dofs_fijos)

    # Callback para Streamlit
    def ui_callback_simp(k, c, delta_c, delta_rho, rho):
        history.append({
            "Iteration": int(k),
            "Compliance": float(c),
            "delta_c": float(delta_c),
            "delta_rho": float(delta_rho)
        })
        if k % 2 == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(-rho.reshape(nely, nelx), cmap='gray', aspect='equal')
            ax.axis('off')
            plot_ph.pyplot(fig)
            plt.close(fig)
            metric_ph.markdown(f"**Iteracion:** {k} | **Compliance:** {c:.2f} N-mm")

    with st.spinner('Optimizando SIMP...'):
        m.optimizar(callback=ui_callback_simp, verbose=False)

    with st.spinner('Analizando topología (TDA)...'):
        mu = m.fase_tda(verbose=False)

    st.success("¡Optimización Finalizada!")

    # Baseline: diseño base uniforme ρ=f_V (hipótesis: "respecto al diseño base uniforme")
    c_base = float(m.c_hist[0])

    # Reducción vs diseño base uniforme (criterio de la hipótesis H.E.2: ≥40%)
    reduccion = ((c_base - m.c_final) / c_base) * 100

    # Guardar resultados en session_state
    st.session_state.simp_c_base = c_base
    st.session_state.simp_reduccion = reduccion
    st.session_state.simp_c_final = m.c_final
    st.session_state.simp_beta1 = m.beta1
    st.session_state.simp_beta0 = m.beta0
    st.session_state.simp_mu = m.mu
    st.session_state.simp_n_iter = m.n_iter
    st.session_state.simp_converged = m.converged
    st.session_state.simp_dgm0 = m.dgm0
    st.session_state.simp_dgm1 = m.dgm1
    st.session_state.simp_c_hist = np.array(m.c_hist)
    st.session_state.simp_rho_hist = m.rho_hist
    st.session_state.simp_rho_tilde_hist = m.rho_tilde_hist
    st.session_state.simp_volfrac = volfrac
    st.session_state.simp_penal_stored = penal
    st.session_state.simp_alpha_stored = alpha
    st.session_state.simp_rmin_stored = rmin
    st.session_state.simp_malla_stored = malla_opcion
    st.session_state.simp_nex = nelx
    st.session_state.simp_ney = nely
    st.session_state.simp_history = history
    st.session_state.simp_t_simp = m.t_simp
    st.session_state.simp_t_tda = m.t_tda
    st.session_state.simp_rho_final = m.rho_final
    st.session_state.simp_concordancia = getattr(m, '_betti_concordancia', None)
    st.session_state.simp_optimized = True
    st.session_state.last_params_he2 = current_params_he2  # reset stale warning

# ═══════════════════════════════════════════════════════════════════
# RENDERIZADO DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════
if st.session_state.get('simp_optimized', False):

    # ── Extraer datos de session_state ──
    rho_final = st.session_state.simp_rho_final
    nex_ = st.session_state.simp_nex
    ney_ = st.session_state.simp_ney
    c_final = st.session_state.simp_c_final
    c_base = st.session_state.simp_c_base
    reduccion = st.session_state.simp_reduccion
    beta0 = st.session_state.simp_beta0
    beta1 = st.session_state.simp_beta1
    mu = st.session_state.simp_mu
    volfrac = st.session_state.simp_volfrac
    alpha_val = st.session_state.simp_alpha_stored
    rmin_val = st.session_state.simp_rmin_stored
    penal_val = st.session_state.simp_penal_stored
    n_iter = st.session_state.simp_n_iter
    converged = st.session_state.simp_converged
    c_hist = st.session_state.simp_c_hist
    rho_tilde_hist = st.session_state.get('simp_rho_tilde_hist', None)
    dgm0 = st.session_state.get('simp_dgm0', None)
    dgm1 = st.session_state.simp_dgm1
    t_simp = st.session_state.simp_t_simp
    t_tda = st.session_state.simp_t_tda
    concordancia = st.session_state.get('simp_concordancia', None)

    # Grisura: 4 * rho * (1 - rho) — promedio sobre diseño final
    grisura = float(np.mean(4.0 * rho_final * (1.0 - rho_final)))

    # ═══════════════════════════════════════════════════════════════
    # VEREDICTO GLOBAL — visible y prominente
    # ═══════════════════════════════════════════════════════════════
    # H.E.2a' es prescriptiva (≥40%), H.E.2b' es descriptiva (invariancia)
    he2a_ok = reduccion >= 40
    if he2a_ok:
        veredicto = "H.E.2a' CUMPLIDA"
        st.success("### ✅ H.E.2a' CUMPLIDA — Reducción ≥ 40% vs uniforme")
    else:
        veredicto = "H.E.2a' NO CUMPLIDA"
        st.error("### ❌ H.E.2a' NO CUMPLIDA — Reducción < 40%")

    # Badges resumen
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Reduccion", f"{reduccion:.1f}%", delta="≥40% OK" if he2a_ok else "<40% FAIL",
              delta_color="normal" if he2a_ok else "inverse")
    b2.metric("β1 (esta corrida)", str(beta1),
              delta="Correr barrido para verificar invariancia")
    b3.metric("μ_α", f"{mu:.4f}")
    b4.metric("Convergio", "✅ Si" if converged else "❌ No")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # 1. H.E.2a' — Reduccion de Compliance (Cuadro 1)
    # ═══════════════════════════════════════════════════════════════
    st.subheader("1. H.E.2a' — Reduccion de Compliance ≥ 40%")

    st.markdown(
        f"**Formula:** (c_base - c_final) / c_base × 100 = "
        f"({c_base:.2f} - {c_final:.2f}) / {c_base:.2f} × 100 = **{reduccion:.2f}%**"
    )

    if he2a_ok:
        st.success(f"✅ Reduccion = {reduccion:.2f}% ≥ 40% → **H.E.2a' CUMPLIDA**")
    else:
        st.warning(f"⚠️ Reduccion = {reduccion:.2f}% < 40% → **H.E.2a' NO CUMPLIDA**")

    # Cuadro 1: comparacion de referentes
    st.markdown("**Cuadro 1 — Compliance vs referentes:**")
    df_cuadro1 = pd.DataFrame({
        "Configuracion": ["Bloque solido (ρ=1)", "Diseno uniforme (ρ=fV)", "Diseno SIMP (ρ*)"],
        "Volumen": ["100%", f"{volfrac*100:.0f}%", f"{volfrac*100:.0f}%"],
        "c(ρ)": ["—", f"{c_base:.2f}", f"{c_final:.2f}"],
        "Reduccion vs uniforme": ["—", "—", f"{reduccion:.2f}%"],
    })
    st.dataframe(df_cuadro1, width="stretch", hide_index=True)

    # Grafico de convergencia
    if c_hist is not None and len(c_hist) > 0:
        st.markdown("**Historial de compliance (convergencia):**")
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(c_hist) + 1)),
            y=c_hist,
            mode='lines+markers',
            line=dict(color='royalblue', width=2),
            name='c(k)'
        ))
        fig_conv.add_hline(y=c_base, line=dict(color='orange', dash='dash', width=1),
                           annotation_text=f'c_base = {c_base:.1f}')
        fig_conv.add_hline(y=c_final, line=dict(color='green', dash='dash', width=1),
                           annotation_text=f'c* = {c_final:.1f}')
        fig_conv.update_layout(
            title="Convergencia SIMP — Historial de Compliance",
            xaxis_title="Iteracion k",
            yaxis_title="Compliance c (N-mm)",
            height=350,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        apply_plotly_theme(fig_conv)
        st.plotly_chart(fig_conv, width="stretch")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # 2. H.E.2b' — Invariancia β1
    # ═══════════════════════════════════════════════════════════════
    st.subheader("2. H.E.2b' — Invariancia β1 (Descriptor topologico)")

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("β0 (Componentes conexas)", str(beta0))
    col_t2.metric("β1 (Agujeros)", str(beta1))
    col_t3.metric("Grisura", f"{grisura:.4f}", help="4ρ(1-ρ) promedio. 0 = binario perfecto.")
    col_t4.metric("Concordancia", "✅ Euler=GUDHI" if concordancia else ("❌ Discordante" if concordancia is not None else "ℹ️ Sin verificar"))

    st.info(
        f"β1 = {beta1} en esta configuración. "
        "H.E.2b' es descriptiva: la invariancia se verifica con el barrido multi-configuración."
    )

    st.caption(
        "H.E.2b' es descriptiva: β1(Ω_sólido) debe ser invariante bajo variacion "
        "de malla, r_min y p. Para una sola corrida se reporta el valor; la invariancia "
        "se verifica con el barrido multi-configuracion."
    )

    # Diagrama de persistencia H1
    if dgm1 is not None and len(dgm1) > 0:
        st.markdown("**Diagrama de persistencia H₁:**")
        finite_mask = np.isfinite(dgm1).all(axis=1)
        dgm1_finite = dgm1[finite_mask]

        fig_pers = go.Figure()
        if len(dgm1_finite) > 0:
            pers = dgm1_finite[:, 1] - dgm1_finite[:, 0]
            fig_pers.add_trace(go.Scatter(
                x=dgm1_finite[:, 0], y=dgm1_finite[:, 1],
                mode='markers',
                marker=dict(size=10, color=pers, colorscale='RdYlGn',
                            colorbar=dict(title='Persistencia'), showscale=True,
                            line=dict(color='black', width=0.5)),
                text=[f'Persistencia: {p:.3f}<br>Nacimiento: {b:.3f}<br>Muerte: {d:.3f}'
                      for (b, d), p in zip(dgm1_finite, pers)],
                hovertemplate='%{text}<extra></extra>',
                name='Ciclos H1'
            ))
        max_val = np.max(dgm1_finite) * 1.1 if len(dgm1_finite) > 0 else 2.0
        fig_pers.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(color='gray', dash='dash'),
            name='Diagonal (ruido)'
        ))
        fig_pers.update_layout(
            xaxis_title="Nacimiento (Birth)", yaxis_title="Muerte (Death)",
            height=400, margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        apply_plotly_theme(fig_pers)
        st.plotly_chart(fig_pers, width="stretch")
    else:
        st.info("No hay datos de persistencia H1 disponibles.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # 3. Distribucion de Material — vista dual
    # ═══════════════════════════════════════════════════════════════
    st.subheader("3. Distribucion de Material")

    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.markdown("**Densidad continua ρ*:**")
        st.plotly_chart(plot_material_distribution(
            rho_final.reshape(ney_, nex_), "Densidad continua ρ*"
        ), width="stretch")
    with col_mat2:
        st.markdown("**Diseño binarizado (ρ̃ ≥ ½):**")
        if rho_tilde_hist is not None and len(rho_tilde_hist) > 0:
            rho_tilde_final = rho_tilde_hist[-1]
            binarized = (rho_tilde_final >= 0.5).astype(float).reshape(ney_, nex_)
            fig_bin = go.Figure(data=[go.Heatmap(
                z=binarized, colorscale='gray_r', zmin=0, zmax=1,
                colorbar=dict(title='Ω')
            )])
            fig_bin.update_layout(title=f"Ω_solido | β0={beta0} | β1={beta1}", height=400)
            apply_plotly_theme(fig_bin)
            st.plotly_chart(fig_bin, width="stretch")
        else:
            st.info("Sin datos de diseño filtrado ρ̃.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # 4. Tabla de metricas + Exportar
    # ═══════════════════════════════════════════════════════════════
    st.subheader("4. Metricas y Exportacion")

    df_metrics = pd.DataFrame({
        "Metrica": [
            "Compliance base (ρ=fV)", "Compliance SIMP (ρ*)", "Reduccion vs uniforme",
            "β0 (Componentes conexas)", "β1 (Agujeros topologicos)",
            "Grisura (4ρ(1-ρ))", "μ_α (Metrica compuesta)",
            "Concordancia Euler-GUDHI",
            "Iteraciones SIMP", "Convergio", "Tiempo SIMP (s)", "Tiempo TDA (s)",
            "Malla", "fV", "p", "α", "r_min"
        ],
        "Valor": [
            f"{c_base:.4f}", f"{c_final:.4f}", f"{reduccion:.2f}%",
            str(beta0), str(beta1),
            f"{grisura:.6f}", f"{mu:.6f}",
            "Si" if concordancia else ("No" if concordancia is not None else "N/A"),
            str(n_iter), "Si" if converged else "No",
            f"{t_simp:.3f}", f"{t_tda:.3f}",
            f"{nex_}x{ney_}", str(volfrac), str(penal_val),
            str(alpha_val), str(rmin_val)
        ]
    })
    st.dataframe(df_metrics, width="stretch", hide_index=True)

    # Exportar ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        # params.txt
        params_txt = (
            f"H.E.2 — Optimizacion SIMP + Metrica Compuesta\n"
            f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Malla: {nex_}x{ney_}\n"
            f"volfrac={volfrac}\npenal={penal_val}\n"
            f"rmin={rmin_val}\nalpha={alpha_val}\nmax_iter={n_iter}\n"
            f"converged={converged}\n"
            f"\nResultados:\n"
            f"  c_base={c_base:.8f}\n  c_final={c_final:.8f}\n"
            f"  reduccion={reduccion:.4f}%\n"
            f"  beta0={beta0}\n  beta1={beta1}\n"
            f"  mu={mu:.8f}\n  grisura={grisura:.8f}\n"
        )
        zipf.writestr("params.txt", params_txt)

        # CSV de metricas
        csv_bytes = df_metrics.to_csv(index=False).encode('utf-8')
        zipf.writestr("metricas.csv", csv_bytes)

        # CSV del historial de convergencia
        df_hist = pd.DataFrame({
            "Iteracion": range(1, len(c_hist) + 1),
            "Compliance": c_hist
        })
        zipf.writestr("historial_convergencia.csv", df_hist.to_csv(index=False))

        # PNG del diseno final (matplotlib — no kaleido)
        fig_mpl, ax_mpl = plt.subplots(figsize=(8, 6))
        ax_mpl.imshow(rho_final.reshape(ney_, nex_), cmap='gray_r', aspect='equal', vmin=0, vmax=1)
        ax_mpl.set_title(f"SIMP | {nex_}x{ney_} | fV={volfrac} | c*={c_final:.4f} | β1={beta1}")
        ax_mpl.axis('off')
        plt.colorbar(ax_mpl.images[0], ax=ax_mpl, label='rho', fraction=0.046, pad=0.04)
        img_buf = io.BytesIO()
        fig_mpl.savefig(img_buf, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig_mpl)
        zipf.writestr("distribucion_material.png", img_buf.getvalue())

    zip_bytes = zip_buffer.getvalue()
    download_button(
        "📥 Descargar TODO (ZIP)", zip_bytes,
        f"he2_fV{volfrac:.2f}_a{alpha_val:.3f}.zip", mime="application/zip",
        width="stretch"
    )

    # ═══════════════════════════════════════════════════════════════
    # 5. BARRIDO MULTI-CONFIGURACIÓN H.E.2b
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("5. Barrido Multi-Configuracion H.E.2b (Invariancia)")

    st.markdown(
        "Verificacion de que β1 es invariante bajo variacion de **r_min** (5 valores), "
        "**p** (4 valores, Cuadro 7) y **resolucion de malla** (Cuadro 8). "
        "El barrido corre 12 configuraciones (rectangular) o 11 (cuadrado)."
    )

    if st.button("Ejecutar Barrido (11 configuraciones)", type="secondary",
                 key="sweep_btn"):

        # Configuraciones del barrido (matching Perfil4.txt 4.3 + Cuadro 7/8)
        configs = []
        # rmin sweep: 5 configs (Perfil4.txt: r_min ∈ [1.5, 4.0])
        for r in [1.5, 2.0, 2.4, 3.0, 4.0]:
            configs.append({"label": f"rmin={r}", "nelx": nex_, "nely": ney_,
                            "p": int(penal), "rmin": r})
        # p sweep: 4 configs (Cuadro 7: p ∈ {1, 2, 3, 4})
        for p_val in [1, 2, 3, 4]:
            configs.append({"label": f"p={p_val}", "nelx": nex_, "nely": ney_,
                            "p": p_val, "rmin": rmin})
        # mesh sweep: según dominio seleccionado
        if dominio.startswith("Rectangular"):
            # Rectangular: Ne ∈ {800, 1800, 3200, 7200}
            meshes_sweep = [(40, 20), (60, 30), (80, 40), (120, 60)]
            ref_nelx = 60  # referencia para escalar rmin
        else:
            # Cuadrado: Ne ∈ {1600, 6400, 14400} (Perfil4.txt Cuadro 1)
            meshes_sweep = [(40, 40), (80, 80), (120, 120)]
            ref_nelx = 40  # referencia para escalar rmin
        for nelx_s, nely_s in meshes_sweep:
            rmin_scaled = rmin * (nelx_s / ref_nelx)
            configs.append({"label": f"malla={nelx_s}x{nely_s}", "nelx": nelx_s,
                            "nely": nely_s, "p": int(penal), "rmin": rmin_scaled})

        sweep_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, cfg in enumerate(configs):
            status_text.text(
                f"[{i+1}/{len(configs)}] {cfg['label']} "
                f"(malla {cfg['nelx']}x{cfg['nely']}, p={cfg['p']}, "
                f"rmin={cfg['rmin']:.1f})..."
            )
            progress_bar.progress(i / len(configs))

            # Setup cantilever beam for this mesh
            nnx_s = cfg["nelx"] + 1
            nny_s = cfg["nely"] + 1
            n_dof_s = 2 * nnx_s * nny_s
            dofs_s = np.arange(0, 2 * (cfg["nely"] + 1))
            node_s = (cfg["nely"] // 2) * nnx_s + cfg["nelx"]
            F_s = np.zeros(n_dof_s)
            F_s[2 * node_s + 1] = -F_carga

            # Run SIMP + TDA
            m_s = MetricaTDA_SIMP(
                nex=cfg["nelx"], ney=cfg["nely"], E=E_acero, nu=0.3,
                Lx=Lx_val, Ly=Ly_val, t=espesor,
                f_V=volfrac, p=cfg["p"], r_min=cfg["rmin"],
                alpha=alpha, max_iter=max_iter
            )
            m_s.definir_problema(F_s, dofs_s)
            m_s.optimizar(verbose=False)
            m_s.fase_tda(verbose=False)

            conc_s = getattr(m_s, '_betti_concordancia', None)
            sweep_results.append({
                "Config": cfg["label"],
                "Malla": f"{cfg['nelx']}x{cfg['nely']}",
                "Ne": cfg["nelx"] * cfg["nely"],
                "rmin": cfg["rmin"],
                "p": cfg["p"],
                "c(rho*)": round(m_s.c_final, 4),
                "beta0": m_s.beta0,
                "beta1": m_s.beta1,
                "Concordancia": "Si" if conc_s else (
                    "No" if conc_s is not None else "N/A")
            })

        progress_bar.progress(1.0)
        status_text.text(f"Barrido completado: {len(configs)} configuraciones.")

        # Guardar en session_state
        st.session_state.sweep_results = sweep_results
        st.session_state.sweep_done = True

    # Mostrar resultados del barrido si existen
    if st.session_state.get('sweep_done', False):
        sweep_results = st.session_state.sweep_results
        df_sweep = pd.DataFrame(sweep_results)

        betas1 = [r["beta1"] for r in sweep_results]
        beta1_invariante = len(set(betas1)) == 1

        # Veredicto invariancia
        if beta1_invariante:
            st.success(
                f"### ✅ β1 INVARIANTE: β1 = {betas1[0]} en las "
                f"{len(sweep_results)} configuraciones"
            )
        else:
            unique_betas = sorted(set(betas1))
            st.warning(
                f"### ⚠️ β1 NO invariante: valores {unique_betas} "
                f"en {len(sweep_results)} configuraciones"
            )

        # Separar tablas por tipo de barrido
        st.markdown("**Barrido completo:**")
        st.dataframe(df_sweep, width="stretch", hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        # Cuadro 7: Barrido de penalización (metodologia_implementacion.txt L629)
        # ═══════════════════════════════════════════════════════════════
        df_p = df_sweep[df_sweep["Config"].str.startswith("p=")]
        if len(df_p) > 0:
            st.markdown("**Cuadro 7 — Barrido del factor de penalización:**")
            st.caption(
                "Grisura = 4ρ(1-ρ), nula si el diseño es binario. "
                "Concordancia: Euler vs GUDHI."
            )
            df_cuadro7 = df_p[["p", "c(rho*)", "beta0", "beta1", "Concordancia"]].copy()
            df_cuadro7.columns = ["p", "c(ρ*)", "β0", "β1", "Concordancia"]
            st.dataframe(df_cuadro7, width="stretch", hide_index=True)

            p_betas = df_p["beta1"].tolist()
            if len(set(p_betas)) == 1:
                st.success(
                    f"✅ β1 = {p_betas[0]} constante en p ∈ {{1,2,3,4}} → "
                    "topología independiente de penalización"
                )
            else:
                st.warning(f"⚠️ β1 varia con p: {set(p_betas)}")

        # ═══════════════════════════════════════════════════════════════
        # Cuadro 8: Independencia de malla (metodologia_implementacion.txt L693)
        # ═══════════════════════════════════════════════════════════════
        df_mesh = df_sweep[df_sweep["Config"].str.startswith("malla")]
        if len(df_mesh) > 0:
            st.markdown("**Cuadro 8 — Independencia de malla:**")
            st.caption(
                "rmin escalado proporcionalmente a nx para mantener "
                "longitud física constante."
            )
            df_cuadro8 = df_mesh[["Malla", "Ne", "rmin", "c(rho*)", "beta0", "beta1"]].copy()
            df_cuadro8.columns = ["Malla", "Ne", "rmin", "c(ρ*)", "β0", "β1"]
            st.dataframe(df_cuadro8, width="stretch", hide_index=True)

            mesh_betas = df_mesh["beta1"].tolist()
            if len(set(mesh_betas)) == 1:
                st.success(
                    f"✅ β1 = {mesh_betas[0]} constante en {len(df_mesh)} mallas → "
                    "independencia de resolución confirmada"
                )
            else:
                st.warning(f"⚠️ β1 varia entre mallas: {set(mesh_betas)}")

        # ═══════════════════════════════════════════════════════════════
        # Barrido de r_min
        # ═══════════════════════════════════════════════════════════════
        df_rmin = df_sweep[df_sweep["Config"].str.startswith("rmin")]
        if len(df_rmin) > 0:
            st.markdown("**Barrido de r_min (5 valores):**")
            df_rmin_show = df_rmin[["rmin", "c(rho*)", "beta0", "beta1", "Concordancia"]].copy()
            df_rmin_show.columns = ["rmin", "c(ρ*)", "β0", "β1", "Concordancia"]
            st.dataframe(df_rmin_show, width="stretch", hide_index=True)

            rmin_betas = df_rmin["beta1"].tolist()
            if len(set(rmin_betas)) == 1:
                st.success(
                    f"✅ β1 = {rmin_betas[0]} constante en "
                    f"rmin ∈ {{1.5, 2.0, 2.4, 3.0, 4.0}}"
                )
            else:
                st.warning(f"⚠️ β1 varia con r_min: {set(rmin_betas)}")

        # Veredicto H.E.2b consolidado
        st.markdown("---")
        he2a_ok = reduccion >= 40
        he2b_ok_mesh = len(set(df_mesh["beta1"].tolist())) == 1 if len(df_mesh) > 0 else False
        he2b_ok_rmin = len(set(df_rmin["beta1"].tolist())) == 1 if len(df_rmin) > 0 else False
        he2b_ok_p = len(set(df_p["beta1"].tolist())) == 1 if len(df_p) > 0 else False

        if he2a_ok and he2b_ok_mesh and he2b_ok_rmin and he2b_ok_p:
            st.success(
                "### ✅ H.E.2 CUMPLIDA — Reduccion ≥40% y β1 invariante "
                "en mallas, r_min y p"
            )
        elif he2a_ok and he2b_ok_mesh:
            st.warning(
                "### ⚠️ H.E.2 PARCIAL — Reduccion OK, β1 invariante en mallas, "
                "pero varia en r_min o p"
            )
        else:
            st.error(
                "### ❌ H.E.2 NO CUMPLIDA — Verificar condiciones"
            )

# ════════════════════════════════════════════════════════════════
# METODOLOGÍA
# ════════════════════════════════════════════════════════════════
methodology_expander(
    "📖 Metodología — H.E.2",
    [
        (
            "H.E.2a' (corregida — la que SÍ se valida)",
            r"\frac{c(f_{V}\cdot \mathbf{1}) - c(\rho^{\star})}{c(f_{V}\cdot \mathbf{1})} \ge 0.40"
        ),
        (
            "H.E.2b' (corregida)",
            r"\beta_{1}(\Omega_{\text{sólido}}) \text{ es invariante para } r_{\min} \in [1.5,\ 4.0], p \in \{2,3,4\}, N_{e} \in \{1600,\ 6400,\ 14400\}"
        ),
        (
            "Métrica compuesta",
            r"\mu_{\alpha} = c + \alpha \cdot \beta_{1} \quad \text{sujeto a:}\quad K(\rho)\cdot U = F,\quad \sum_{e} \rho_{e} v_{e} = f_{V}\cdot V_{0}"
        ),
        (
            "Proposición 8.3 (cota inferior)",
            r"c(\rho) \ge c(1) \text{ para todo } \rho \text{ admisible } \rightarrow \text{comparar vs bloque sólido es imposible}"
        )
    ],
    "H.E.2"
)

st.markdown("---")


