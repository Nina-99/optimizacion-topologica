"""Página 2: H.E.2 — Optimización SIMP + Métrica Compuesta.

Valida la Hipótesis Específica 2 (H.E.2):
"El método SIMP con p = 3 y fracción de volumen fV = 0.5 converge
a una distribución de material que reduce la compliance global en
al menos un 40% respecto al diseño base uniforme (ρ = fV), generando
una topología con β₁(Ω) ≤ 2 verificable computacionalmente."

Referencias del Documento:
- Definición 1.6: Malla FEM y variable de diseño ρ
- Definición 1.7: Problema SIMP
- Definición 1.8: Nube de Puntos del Diseño SIMP
- Definición 1.9: Métrica Compuesta TDA-SIMP (μ_α = c + α·β₁)
- Teorema 1.1: Bien-definición y acotación de μ_α
- Algoritmo 1: Métrica Compuesta TDA-SIMP
- Cuadro 1: Parámetros de la simulación
"""

import streamlit as st
from tda.app.download_utils import download_button, export_settings_ui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import io
import zipfile
from datetime import datetime

from tda.optimization.metric_simp import MetricaTDA_SIMP
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, is_dark,
    metric_card, report_header, responsive_style, diagnosticar_he2,
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

# ── Modo Defensa Toggle ──
_modo_original = st.sidebar.checkbox("🔄 Usar parámetros de la tesis (α=0.012, r_min=2.4)", value=False,
    help="Restaura los valores del Cuadro 1 de la tesis. Default: α=0.036, r_min=3.0 (validados experimentalmente).")

# ── Page Header ──
st.markdown(page_header(
    "H.E.2 — Optimización SIMP + Métrica Compuesta μ_α",
    "Algoritmo 1: μ_α = c + α·β₁ | p=3, f_V=0.5, reducción ≥40%"
), unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.header("🏗️ Optimización SIMP")
malla_opcion = st.sidebar.selectbox(
    "Resolución de Malla",
    ["60x30 (Caso Tesis)", "80x50 (4000 elem)", "100x60 (6000 elem)", "80x80 (6400 elem)"],
    key="simp_malla",
    help="Resolución de elementos finitos Q4. 60×30 es el caso de la tesis. Mayor resolución captura detalles más finos pero aumenta el tiempo de cómputo. 80×50 y 100×60 permiten mejores reducciones de compliance (hacia el 40% objetivo)."
)
volfrac = st.sidebar.slider("Fracción de Volumen", 0.0, 1.0, 0.5, 0.01, key="simp_volfrac_input",
    help="Fracción de volumen permitida respecto al dominio completo. f_V = 0.5 significa que solo el 50% del espacio puede tener material. Valores típicos: 0.3-0.7. "
    "Cuidado: f_V muy cercano a 0 o 1 puede causar problemas de convergencia en el optimizer SIMP.")
penal = st.sidebar.number_input("Factor Penalización (p)", value=3.0, step=1.0, key="simp_penal",
    help="Penaliza densidades intermedias (material gris) forzando una solución 0/1. p=3 es el estándar SIMP. p>3 converge más rápido pero puede ser inestable.")
rmin = st.sidebar.number_input("Radio Filtro", value=(2.4 if _modo_original else 3.0), step=0.1, key="simp_rmin",
    help="Radio del filtro de sensibilidad. Default: 3.0 (validado). Con valor original: 2.4.")
alpha = st.sidebar.number_input("Peso α (métrica μ_α)", value=(0.012 if _modo_original else 0.036), step=0.001, format="%.3f", key="simp_alpha",
    help="Peso del término topológico β₁. Default: 0.036 (validado, β₁≤2). Con valor original: 0.012.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Parámetros Físicos")
E_acero = st.sidebar.number_input("Módulo Young E₀ (MPa)", value=200000.0, step=1000.0, help="Acero = 200 GPa = 200,000 MPa")
F_carga = st.sidebar.number_input("Carga F (N)", value=1000.0, step=100.0, help="Carga aplicada = 1 kN = 1000 N")
espesor = st.sidebar.number_input("Espesor (mm)", value=1.0, step=0.1, help="Espesor de la viga en 2D plano")

# Nuevo slider para iteraciones máximas
max_iter = st.sidebar.slider("Iteraciones Máximas", min_value=100, max_value=500, value=200,
    help="Número máximo de iteraciones del bucle SIMP. Valor por defecto (200) sugiere una convergencia más completa. "
    "La barra de progreso se ajustará automáticamente a este valor.")


# ==========================================
# FUNCIÓN DE ANIMACIÓN SIMP CON PLOTLY
# ==========================================
def crear_animacion_simp(rho_hist, c_hist, nex, ney, volfrac, penal):
    """
    Crea una figura Plotly animada con la evolución del proceso SIMP.
    Incluye slider de iteración, botones Play/Pause/Reset,
    y gráfico de convergencia sincronizado.
    """
    n_frames = len(rho_hist)
    dark = is_dark()

    # Colores según tema
    bg = 'rgba(0,0,0,0)' if dark else '#fafafa'
    paper_bg = 'rgba(0,0,0,0)' if dark else 'white'
    grid_c = '#444444' if dark else '#ecf0f1'
    font_c = '#ecf0f1' if dark else '#2c3e50'
    annot_bg = 'rgba(30,30,30,0.9)' if dark else 'rgba(255,255,255,0.9)'
    annot_border = '#555555' if dark else '#bdc3c7'

    # Downsample si hay demasiados frames (>50)
    if n_frames > 50:
        step = max(1, n_frames // 50)
        idx = list(range(0, n_frames, step))
        if idx[-1] != n_frames - 1:
            idx.append(n_frames - 1)
        rho_hist_ds = [rho_hist[i] for i in idx]
        c_hist_ds = [c_hist[i] for i in idx]
        n_frames_ds = len(rho_hist_ds)
    else:
        rho_hist_ds = rho_hist
        c_hist_ds = c_hist
        idx = list(range(n_frames))
        n_frames_ds = n_frames

    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.12,
        subplot_titles=(
            "Distribución de Material (ρ)",
            "Convergencia de Compliance c(ρ)"
        )
    )

    # Frame inicial (iteración 0)
    rho_2d_0 = rho_hist_ds[0].reshape(ney, nex)

    # Heatmap de densidades
    fig.add_trace(
        go.Heatmap(
            z=rho_2d_0,
            colorscale='gray_r',
            zmin=0, zmax=1,
            colorbar=dict(title='ρ', x=1.02, len=0.7, thickness=15),
            hovertemplate='x: %{x}<br>y: %{y}<br>ρ: %{z:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Línea de convergencia (vacía inicialmente, se llena en frames)
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(color='royalblue', width=2),
            name='Compliance',
            hovertemplate='Iter: %{x}<br>c: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Línea vertical de iteración actual
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Iteración actual',
            showlegend=True
        ),
        row=2, col=1
    )

    # Crear frames para la animación
    frames = []
    for k, (i_frame, (rho_k, c_k)) in enumerate(zip(idx, zip(rho_hist_ds, c_hist_ds))):
        rho_2d_k = rho_k.reshape(ney, nex)
        iter_real = i_frame + 1  # 1-indexed

        # Compliance history UP TO this iteration
        c_up_to = c_hist[:i_frame + 1]

        # Meter toda la info en una sola string para el título
        n_solid = int(np.sum(rho_k > 0.5))
        pct_solid = 100 * n_solid / len(rho_k)

        frame = go.Frame(
            data=[
                # Heatmap
                go.Heatmap(
                    z=rho_2d_k,
                    colorscale='gray_r',
                    zmin=0, zmax=1,
                    colorbar=dict(title='ρ', x=1.02, len=0.7, thickness=15)
                ),
                # Convergence line
                go.Scatter(
                    x=np.arange(1, len(c_up_to) + 1),
                    y=c_up_to,
                    mode='lines',
                    line=dict(color='royalblue', width=2)
                ),
                # Current iteration marker
                go.Scatter(
                    x=[iter_real, iter_real],
                    y=[min(c_hist), max(c_hist)],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash')
                )
            ],
            name=str(k),
            layout=go.Layout(
                annotations=[dict(
                    x=0.5, y=1.0, xref='paper', yref='paper',
                    text=f'<b>Iteración {iter_real}/{n_frames}</b> | '
                         f'c = {c_k:.4f} | Sólidos: {n_solid}/{len(rho_k)} ({pct_solid:.1f}%)',
                    showarrow=False, font=dict(size=13, color=font_c),
                    align='center', bgcolor=annot_bg,
                    bordercolor=annot_border, borderwidth=1,
                    xanchor='center', yanchor='bottom'
                )]
            )
        )
        frames.append(frame)

    fig.frames = frames

    # Determinar rango y del convergence plot
    c_min = min(c_hist)
    c_max = max(c_hist)
    c_range = c_max - c_min if c_max != c_min else 1.0

    # Layout general
    fig.update_layout(
        title=dict(
            text=f'<b>Evolución de la Optimización SIMP</b><br>'
                 f'<sup>Malla {nex}×{ney} | f_V={volfrac} | p={penal}</sup>',
            font=dict(size=16),
            x=0.5, xanchor='center'
        ),
        height=650,
        hovermode='x unified',
        # Ejes del heatmap
        xaxis1=dict(visible=False, range=[-0.5, nex - 0.5]),
        yaxis1=dict(visible=False, range=[-0.5, ney - 0.5], scaleanchor='x', autorange='reversed'),
        # Ejes de la convergencia
        xaxis2=dict(
            title='Iteración',
            range=[0, n_frames + 1],
            gridcolor=grid_c,
            zeroline=False
        ),
        yaxis2=dict(
            title='Compliance c(ρ)',
            type='log',
            range=[
                np.log10(max(c_min - 0.1 * c_range, 1e-10)),
                np.log10(c_max + 0.1 * c_range)
            ],
            gridcolor=grid_c,
            zeroline=False
        ),
        # Slider
        sliders=[{
            'currentvalue': {
                'prefix': 'Iteración: ',
                'font': {'size': 14, 'color': font_c},
                'xanchor': 'center'
            },
            'len': 0.92,
            'x': 0.04,
            'y': 0.0,
            'pad': {'t': 40, 'b': 10},
            'font': {'size': 11},
            'steps': [
                {
                    'args': [[str(k)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(idx[k] + 1),
                    'method': 'animate'
                }
                for k in range(n_frames_ds)
            ]
        }],
        # Botones de control
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x': 0.0,
            'y': -0.05,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {'size': 12},
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 150, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[], {
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏪ Reset',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'fromcurrent': False,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        plot_bgcolor=bg,
        paper_bgcolor=paper_bg,
    )

    return fig


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
            metric_ph.markdown(f"**Iteración:** {k} | **Compliance:** {c:.2f} N·mm")

    # Instancia del optimizador MetricaTDA_SIMP con dimensiones físicas reales
    m = MetricaTDA_SIMP(
        nex=nelx, ney=nely, E=E_acero, nu=0.3,
        Lx=120.0, Ly=40.0, t=espesor,
        f_V=volfrac, p=penal, r_min=rmin, alpha=alpha, max_iter=max_iter
    )

    m.definir_problema(F, dofs_fijos)

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
    st.session_state.simp_c_final = m.c_final
    st.session_state.simp_beta1 = m.beta1
    st.session_state.simp_beta0 = m.beta0
    st.session_state.simp_mu = m.mu
    st.session_state.simp_n_iter = m.n_iter
    st.session_state.simp_converged = m.converged
    st.session_state.simp_dgm1 = m.dgm1
    st.session_state.simp_nube = m.nube
    st.session_state.simp_eps_star = m.eps_star
    st.session_state.simp_c_hist = np.array(m.c_hist)
    st.session_state.simp_rho_hist = m.rho_hist
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
    st.session_state.simp_optimized = True

# Renderizar resultados si existen
if st.session_state.get('simp_optimized', False):

    # Extraer datos de session_state
    rho_final = st.session_state.simp_rho_final
    nex_ = st.session_state.simp_nex
    ney_ = st.session_state.simp_ney
    c_final = st.session_state.simp_c_final
    c_hist = st.session_state.simp_c_hist
    rho_hist = st.session_state.get('simp_rho_hist', None)
    dgm1 = st.session_state.simp_dgm1
    nube = st.session_state.simp_nube
    eps_star = st.session_state.simp_eps_star

    tab_evol, tab_res, tab_tda, tab_report = st.tabs([
        "📈 Evolución", "🏗️ Resultados", "🔬 Análisis TDA", "📊 Reporte"
    ])

    # ── Banner de parámetros usados (persistencia entre páginas) ──
    malla_str = st.session_state.get("simp_malla_stored", f"{nex_}x{ney_}")
    volfrac_str = st.session_state.get("simp_volfrac", "—")
    penal_str = st.session_state.get("simp_penal_stored", "—")
    alpha_str = st.session_state.get("simp_alpha_stored", "—")
    rmin_str = st.session_state.get("simp_rmin_stored", "—")
    st.info(
        f"📋 Mostrando resultados previos — "
        f"Malla: {malla_str}, "
        f"f_V={volfrac_str}, p={penal_str}, "
        f"α={alpha_str}, r_min={rmin_str}. "
        f"Cambiá los parámetros en el sidebar y ejecutá una nueva optimización."
    )

    # ═══════════════════════════════════════════════════
    # TAB 1: EVOLUCIÓN INTERACTIVA
    # ═══════════════════════════════════════════════════
    with tab_evol:
        st.subheader("Animación del Proceso de Optimización")

        volfrac = st.session_state.simp_volfrac
        penal = st.session_state.simp_penal_stored

        if rho_hist is not None and len(rho_hist) > 1:
            fig_anim = crear_animacion_simp(rho_hist, c_hist, nex_, ney_, volfrac, penal)
            st.plotly_chart(fig_anim, width='stretch')
        else:
            st.info("No hay historial de iteraciones para animar.")
            fig_final, ax_final = plt.subplots(figsize=(8, 4))
            ax_final.imshow(rho_final.reshape(ney_, nex_), cmap='gray_r', aspect='equal')
            ax_final.axis('off')
            ax_final.set_title("Distribución Final de Material")
            st.pyplot(fig_final)
            plt.close(fig_final)

        # Métricas rápidas siempre visibles
        st.markdown("---")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        c_final = st.session_state.simp_c_final
        c_base = st.session_state.simp_c_base
        reduccion = st.session_state.simp_reduccion
        
        col_q1.metric("Compliance Base (ρ=0.5)", f"{c_base:.1f} N·mm")
        col_q2.metric("Compliance SIMP (50%)", f"{c_final:.1f} N·mm",
                      delta=f"{reduccion:.1f}% vs Base", delta_color="normal")
        col_q3.metric("β₁ (Agujeros)", st.session_state.simp_beta1)
        col_q4.metric("μ_α Compuesta", f"{st.session_state.simp_mu:.2f}")

        # α* calibrado (Proposición 1.1)
        from tda.core.metric import calibrar_alpha_optimo
        alpha_star = calibrar_alpha_optimo(
            [st.session_state.simp_c_base, st.session_state.simp_c_final],
            [0, st.session_state.simp_beta1]
        )
        st.info(
            f"**α* calibrado (Prop. 1.1):** {alpha_star:.4f} — "
            f"α ingresado: {st.session_state.simp_alpha_stored:.4f} — "
            f"{'✅ Consistente' if abs(alpha_star - st.session_state.simp_alpha_stored) < 0.01 else '⚠️ Diferencia significativa'}"
        )

        # ── Convergencia Dual: Δc/c y Δρ ──
        history = st.session_state.get("simp_history", [])
        if history and len(history) > 1:
            st.markdown("---")
            st.subheader("Convergencia Dual (Criterio del Documento)")
            st.caption(
                "Criterio de convergencia: Δc/c < 10⁻⁴ AND Δρ < 10⁻⁴ "
                "(Documento, Algoritmo 1, paso 8)"
            )

            iterations = [h["Iteration"] for h in history]
            delta_c_vals = [h.get("delta_c", 0) for h in history]
            delta_rho_vals = [h.get("delta_rho", 0) for h in history]

            fig_dual = go.Figure()

            # Δc/c — el callback ya entrega el cambio relativo |c_k - c_{k-1}|/c_{k-1},
            # así que NO se normaliza de nuevo (antes se dividía otra vez por c_ref).
            delta_c_norm = [abs(dc) for dc in delta_c_vals]
            fig_dual.add_trace(go.Scatter(
                x=iterations, y=delta_c_norm,
                mode='lines', name='Δc/c (Compliance)',
                line=dict(color='royalblue', width=2)
            ))

            # Δρ
            fig_dual.add_trace(go.Scatter(
                x=iterations, y=delta_rho_vals,
                mode='lines', name='Δρ (Densidades)',
                line=dict(color='orange', width=2)
            ))

            # Umbrales de convergencia (10⁻⁴)
            fig_dual.add_hline(
                y=1e-4, line=dict(color='red', dash='dash', width=1),
                annotation_text='Umbral = 10⁻⁴',
                annotation_position='right'
            )

            fig_dual.update_layout(
                xaxis_title="Iteración k",
                yaxis_title="Valor",
                yaxis_type="log",
                height=350,
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
            )
            apply_plotly_theme(fig_dual)
            st.plotly_chart(fig_dual, width='stretch')

            # Verificación de convergencia
            converged_c = delta_c_norm[-1] < 1e-4 if delta_c_norm else False
            converged_rho = delta_rho_vals[-1] < 1e-4 if delta_rho_vals else False

            c1, c2, c3 = st.columns(3)
            c1.metric("Δc/c final", f"{delta_c_norm[-1]:.2e}" if delta_c_norm else "—",
                      delta="✅ < 10⁻⁴" if converged_c else "❌ ≥ 10⁻⁴",
                      delta_color="normal" if converged_c else "inverse")
            c2.metric("Δρ final", f"{delta_rho_vals[-1]:.2e}" if delta_rho_vals else "—",
                      delta="✅ < 10⁻⁴" if converged_rho else "❌ ≥ 10⁻⁴",
                      delta_color="normal" if converged_rho else "inverse")
            c3.metric("Convergencia dual",
                      "✅ Lograda" if (converged_c and converged_rho) else "❌ No lograda",
                      delta_color="normal" if (converged_c and converged_rho) else "inverse")

    # ═══════════════════════════════════════════════════
    # TAB 2: RESULTADOS FINALES (figura 2×2)
    # ═══════════════════════════════════════════════════
    with tab_res:
        st.subheader("Resultados Finales de la Optimización")

        fig_res, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig_res.suptitle(
            f"Resultados SIMP | {nex_}×{ney_} | f_V={st.session_state.simp_volfrac} | μ_α={st.session_state.simp_mu:.4f}",
            fontsize=14
        )

        # (0,0): Mapa de densidades
        im = axes[0, 0].imshow(rho_final.reshape(ney_, nex_), cmap='gray_r', aspect='equal', vmin=0, vmax=1)
        axes[0, 0].set_title(f"Distribución de Material ρ* | c = {c_final:.4f}")
        axes[0, 0].axis('off')
        plt.colorbar(im, ax=axes[0, 0], label='Densidad ρ', fraction=0.046, pad=0.04)

        # (0,1): Diagrama de persistencia H1
        if dgm1 is not None and len(dgm1) > 0:
            finite = np.isfinite(dgm1[:, 1])
            if np.any(finite):
                pers = dgm1[finite, 1] - dgm1[finite, 0]
                sc = axes[0, 1].scatter(dgm1[finite, 0], dgm1[finite, 1], c=pers, cmap='RdYlGn', s=60,
                                        zorder=5, label='Ciclos H₁')
                plt.colorbar(sc, ax=axes[0, 1], label='Persistencia', fraction=0.046, pad=0.04)
                max_val = np.max(dgm1[finite]) * 1.1 if len(dgm1[finite]) > 0 else 2.0
                axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Diagonal (ruido)')
                if eps_star is not None:
                    axes[0, 1].axhline(y=eps_star / 2, color='red', linestyle=':', alpha=0.7,
                                       label=f'Umbral = ε*/2')
                axes[0, 1].legend(fontsize=8)
        axes[0, 1].set_xlabel("Nacimiento b")
        axes[0, 1].set_ylabel("Muerte d")
        axes[0, 1].set_title(f"Diagrama Persistencia H₁ | β₁ significativo = {st.session_state.simp_beta1}")
        axes[0, 1].grid(True, linestyle='--', alpha=0.3)

        # (1,0): Convergencia
        if c_hist is not None and len(c_hist) > 0:
            axes[1, 0].semilogy(range(1, len(c_hist) + 1), c_hist, 'b-', linewidth=1.5, label='c(k)')
            axes[1, 0].axhline(y=c_final, color='r', linestyle='--', alpha=0.7, label=f'c* = {c_final:.4f}')
            axes[1, 0].legend()
        axes[1, 0].set_xlabel("Iteración k")
        axes[1, 0].set_ylabel("Compliance c(ρ)")
        axes[1, 0].set_title(f"Convergencia SIMP | {len(c_hist)} iteraciones")
        axes[1, 0].grid(True, linestyle='--', alpha=0.3)

        # (1,1): Nube de puntos
        if nube is not None and len(nube) > 0:
            axes[1, 1].scatter(nube[:, 0], nube[:, 1], s=8, alpha=0.5, c='navy')
        axes[1, 1].set_xlim(0, nex_)
        axes[1, 1].set_ylim(0, ney_)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].set_xlabel("x (elementos)")
        axes[1, 1].set_ylabel("y (elementos)")
        axes[1, 1].set_title(
            f"Nube X(ρ*) | β₀={st.session_state.simp_beta0} | β₁={st.session_state.simp_beta1} | μ_α={st.session_state.simp_mu:.4f}")
        axes[1, 1].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_res)
        plt.close(fig_res)

    # ═══════════════════════════════════════════════════
    # TAB 3: ANÁLISIS TDA
    # ═══════════════════════════════════════════════════
    with tab_tda:
        st.subheader("Análisis Topológico (TDA)")

        # Métricas TDA en tarjetas visuales adaptables al tema
        b1 = st.session_state.simp_beta1
        color_b1 = "#27ae60" if b1 == 0 else "#e74c3c"
        manufacturable = b1 == 0
        estado = "✅ APTA" if manufacturable else "⚠️ REVISAR"
        color_est = "#27ae60" if manufacturable else "#e74c3c"

        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            st.markdown(metric_card(
                value=str(st.session_state.simp_beta0),
                title="β₀",
                subtitle="Componentes conexas",
                variant="beta0"
            ), unsafe_allow_html=True)

        with col_t2:
            st.markdown(metric_card(
                value=str(b1),
                title="β₁",
                subtitle="Agujeros topológicos",
                variant="beta1",
                value_color=color_b1
            ), unsafe_allow_html=True)

        with col_t3:
            st.markdown(metric_card(
                value=f"{st.session_state.simp_mu:.4f}",
                title="μ_α",
                subtitle="Métrica compuesta",
                variant="mu"
            ), unsafe_allow_html=True)

        with col_t4:
            st.markdown(metric_card(
                value=estado,
                title="Manufactura",
                subtitle="Verificación topológica",
                variant="manufacturing",
                value_color=color_est
            ), unsafe_allow_html=True)

        st.markdown("---")

        # Diagrama de persistencia interactivo (Plotly)
        st.subheader("Diagrama de Persistencia Interactivo")

        # Filtrar puntos no finitos
        dgm1_local = st.session_state.simp_dgm1
        if dgm1_local is not None and len(dgm1_local) > 0:
            finite_mask = np.isfinite(dgm1_local).all(axis=1)
            dgm1_finite = dgm1_local[finite_mask]
        else:
            dgm1_finite = np.empty((0, 2))

        fig_tda = go.Figure()

        if len(dgm1_finite) > 0:
            pers = dgm1_finite[:, 1] - dgm1_finite[:, 0]
            fig_tda.add_trace(go.Scatter(
                x=dgm1_finite[:, 0],
                y=dgm1_finite[:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color=pers,
                    colorscale='RdYlGn',
                    colorbar=dict(title='Persistencia'),
                    showscale=True,
                    line=dict(color='black', width=0.5)
                ),
                text=[f'Persistencia: {p:.3f}<br>Nacimiento: {b:.3f}<br>Muerte: {d:.3f}'
                      for (b, d), p in zip(dgm1_finite, pers)],
                hovertemplate='%{text}<extra></extra>',
                name='Ciclos H₁'
            ))

        # Diagonal
        if len(dgm1_finite) > 0:
            max_val = np.max(dgm1_finite) * 1.1
        else:
            max_val = 2.0
        fig_tda.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Nacimiento = Muerte'
        ))

        # Umbral
        eps_star_local = st.session_state.simp_eps_star
        if eps_star_local is not None and eps_star_local > 0:
            fig_tda.add_hline(
                y=eps_star_local / 2,
                line=dict(color='red', dash='dot'),
                annotation_text=f'Umbral = ε*/2 = {eps_star_local / 2:.3f}',
                annotation_position='right'
            )

        fig_tda.update_layout(
            xaxis_title="Tiempo de Nacimiento (Birth)",
            yaxis_title="Tiempo de Muerte (Death)",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
            hovermode='closest',
        )
        apply_plotly_theme(fig_tda)
        st.plotly_chart(fig_tda, width='stretch')

        st.markdown("---")

        # Información adicional
        with st.expander("📖 Interpretación de los invariantes topológicos"):
            st.markdown("""
            **β₀ (Número de Betti de dimensión 0):** Número de componentes conexas.
            - β₀ = 1 → El diseño es una sola pieza continua (ideal para manufactura).
            - β₀ > 1 → Hay partes desconectadas. Puede indicar necesidad de soportes.

            **β₁ (Número de Betti de dimensión 1):** Número de agujeros 1-dimensionales.
            - β₁ = 0 → Estructura sin agujeros internos (manufacturabilidad garantizada).
            - β₁ > 0 → Hay cavidades o túneles. Pueden ser intencionales o espurios.

            **μ_α = c + α·β₁:** Métrica compuesta TDA-SIMP.
            - Penaliza diseños con agujeros topológicos innecesarios.
            - Menor μ_α → mejor balance entre eficiencia mecánica y simplicidad topológica.
            """)

        # ── Validación H.E.2 ──
        with st.expander("✅ Validación H.E.2", expanded=True):
            st.markdown("""
            **Hipótesis H.E.2:** SIMP con p=3 y fV=0.5 reduce la compliance
            en al menos un 40% respecto al diseño base uniforme (ρ=fV),
            generando una topología con β₁(Ω) ≤ 2.
            """)
            reduccion = st.session_state.simp_reduccion

            diag = diagnosticar_he2(
                reduccion=reduccion,
                beta1=st.session_state.simp_beta1,
                mu=st.session_state.simp_mu,
                alpha=st.session_state.simp_alpha_stored,
                rmin=st.session_state.simp_rmin_stored,
                penal=st.session_state.simp_penal_stored,
                volfrac=st.session_state.simp_volfrac,
                n_iter=st.session_state.simp_n_iter,
                converged=st.session_state.simp_converged,
                max_iter=max_iter,
            )

            # Badges resumen
            st.markdown(diag["badges_html"], unsafe_allow_html=True)
            st.markdown("---")

            # Veredicto
            if diag["veredicto"] == "CUMPLIDA":
                st.success(f"✅ **H.E.2 {diag['veredicto']}:** Reducción ≥40% y β₁ ≤ 2")
            elif diag["veredicto"] == "PARCIAL":
                st.warning(f"⚠️ **H.E.2 {diag['veredicto']}:** Se cumple solo uno de los dos criterios")
            else:
                st.error(f"❌ **H.E.2 {diag['veredicto']}:** No se cumple ningún criterio")

            # Razones
            with st.expander("📖 ¿Por qué?", expanded=True):
                for r in diag["razones"]:
                    st.markdown(f"- {r}")

            # Sugerencias
            if diag["sugerencias"]:
                with st.expander("💡 Sugerencias de ajuste", expanded=diag["veredicto"] != "CUMPLIDA"):
                    for s in diag["sugerencias"]:
                        st.markdown(f"- {s}")

    # ═══════════════════════════════════════════════════
    # TAB 4: REPORTE Y EXPORTACIÓN
    # ═══════════════════════════════════════════════════
    with tab_report:
        st.subheader("Exportar Resultados")

        # ── Datos compartidos ─────────────────────────────────────────────
        beta0 = st.session_state.simp_beta0
        beta1 = st.session_state.simp_beta1
        c_final = st.session_state.simp_c_final
        mu = st.session_state.simp_mu
        reduccion = st.session_state.simp_reduccion
        volfrac = st.session_state.simp_volfrac
        penal = st.session_state.simp_penal_stored
        alpha_val = st.session_state.simp_alpha_stored
        n_iter = st.session_state.simp_n_iter
        converged = st.session_state.simp_converged
        t_simp = st.session_state.simp_t_simp
        t_tda = st.session_state.simp_t_tda
        nex_ = st.session_state.simp_nex
        ney_ = st.session_state.simp_ney
        rho_final = st.session_state.simp_rho_final
        c_hist = st.session_state.simp_c_hist
        dgm1 = st.session_state.simp_dgm1
        nube = st.session_state.simp_nube
        eps_star = st.session_state.simp_eps_star
        manufacturable = beta1 == 0

        # ── Header del reporte ──
        st.markdown(report_header(
            "📦 Paquete de Resultados TDA-SIMP",
            "Todos los datos necesarios para tu tesis y presentación"
        ), unsafe_allow_html=True)

        # ── Fila 1: PDF + PNG + LaTeX ─────────────────────────────────────
        st.markdown("#### 📄 Documentos")
        col_r1, col_r2, col_r3 = st.columns(3)

        # ── PDF Profesional (multipágina) ──────────────────────────────────
        with col_r1:
            pdf_buf = io.BytesIO()
            with PdfPages(pdf_buf) as pdf:

                # ── Página 1: Portada ──────────────────────────────────────
                fig_cover = plt.figure(figsize=(8.27, 11.69))  # A4 vertical
                fig_cover.patch.set_facecolor('#f8f9fa')
                ax_cover = fig_cover.add_axes([0.1, 0.1, 0.8, 0.8])
                ax_cover.axis('off')

                # Título
                ax_cover.text(0.5, 0.85, 'Reporte de Optimización TDA-SIMP',
                              ha='center', va='center', fontsize=24, fontweight='bold',
                              color='#2c3e50', transform=ax_cover.transAxes)
                ax_cover.text(0.5, 0.78, 'Algoritmo 1 — Métrica Compuesta',
                              ha='center', va='center', fontsize=14,
                              color='#7f8c8d', transform=ax_cover.transAxes)

                # Línea decorativa
                ax_cover.plot([0.2, 0.8], [0.74, 0.74], color='#3498db',
                              linewidth=2, transform=ax_cover.transAxes)

                # Parámetros del problema
                params_text = (
                    f"Malla: {nex_}×{ney_} elementos Q4\n"
                    f"Fracción de volumen: f_V = {volfrac}\n"
                    f"Penalización SIMP: p = {penal}\n"
                    f"Peso topológico: α = {alpha_val}\n"
                    f"Iteraciones: {n_iter} {'✓ Convergió' if converged else '✗ No convergió'}"
                )
                ax_cover.text(0.5, 0.60, params_text, ha='center', va='center',
                              fontsize=11, color='#34495e', transform=ax_cover.transAxes,
                              linespacing=1.8, family='monospace')

                # Resultados principales
                ax_cover.plot([0.15, 0.85], [0.48, 0.48], color='#bdc3c7',
                              linewidth=0.5, transform=ax_cover.transAxes)

                res_text = (
                    f"Compliance final:      c = {c_final:.5f}\n"
                    f"Reducción vs base:     {reduccion:.2f}%\n"
                    f"Componentes conexas:   β₀ = {beta0}\n"
                    f"Agujeros topológicos:  β₁ = {beta1}\n"
                    f"Métrica compuesta:     μ_α = {mu:.5f}\n"
                    f"Manufacturabilidad:    {'✓ APTA' if manufacturable else '⚠ REVISAR'}"
                )
                ax_cover.text(0.5, 0.38, res_text, ha='center', va='center',
                              fontsize=11, color='#2c3e50', transform=ax_cover.transAxes,
                              linespacing=1.8, family='monospace')

                # Tiempos de cómputo
                ax_cover.text(0.5, 0.18,
                              f"Tiempo SIMP: {t_simp:.2f}s  |  Tiempo TDA: {t_tda:.3f}s  |  Total: {t_simp + t_tda:.2f}s",
                              ha='center', va='center', fontsize=9,
                              color='#95a5a6', transform=ax_cover.transAxes)

                # Footer
                ax_cover.text(0.5, 0.05,
                              'Jorge Larry Copa Cruz · Maestría en Matemática · UAGRM 2026',
                              ha='center', va='center', fontsize=8,
                              color='#bdc3c7', transform=ax_cover.transAxes,
                              style='italic')

                pdf.savefig(fig_cover, dpi=200)
                plt.close(fig_cover)

                # ── Página 2: Figura 2×2 completa ──────────────────────────
                # Usar estilo default para exportación (fondo blanco)
                with plt.style.context('default'):
                    fig_res_pdf, axes_pdf = plt.subplots(2, 2, figsize=(10, 8))
                    fig_res_pdf.patch.set_facecolor('white')
                    for ax_row in axes_pdf:
                        for ax in ax_row:
                            ax.set_facecolor('white')
                    fig_res_pdf.suptitle(
                        f'Resultados de Optimización | {nex_}×{ney_} | f_V={volfrac} | μ_α={mu:.4f}',
                        fontsize=14, fontweight='bold')

                    # (0,0) Density
                    rho_2d = rho_final.reshape(ney_, nex_)
                    im = axes_pdf[0, 0].imshow(rho_2d, cmap='gray_r', aspect='equal', vmin=0, vmax=1)
                    axes_pdf[0, 0].set_title(f'Distribución ρ* | c={c_final:.4f}')
                    axes_pdf[0, 0].axis('off')
                    plt.colorbar(im, ax=axes_pdf[0, 0], label='ρ', fraction=0.046, pad=0.04)

                    # (0,1) Persistence diagram
                    if dgm1 is not None and len(dgm1) > 0:
                        finite = np.isfinite(dgm1[:, 1])
                        if np.any(finite):
                            pers = dgm1[finite, 1] - dgm1[finite, 0]
                            sc = axes_pdf[0, 1].scatter(dgm1[finite, 0], dgm1[finite, 1],
                                                        c=pers, cmap='RdYlGn', s=40, zorder=5)
                            plt.colorbar(sc, ax=axes_pdf[0, 1], label='Persistencia', fraction=0.046, pad=0.04)
                            mv = np.max(dgm1[finite]) * 1.1
                            axes_pdf[0, 1].plot([0, mv], [0, mv], 'k--', alpha=0.4)
                            if eps_star:
                                axes_pdf[0, 1].axhline(eps_star / 2, color='r', ls=':', alpha=0.7, label=f'ε*/2')
                            axes_pdf[0, 1].legend(fontsize=7)
                    axes_pdf[0, 1].set_title(f'Diagrama H₁ | β₁={beta1}')
                    axes_pdf[0, 1].set_xlabel('Birth')
                    axes_pdf[0, 1].set_ylabel('Death')
                    axes_pdf[0, 1].grid(True, alpha=0.3)

                    # (1,0) Convergence
                    if c_hist is not None and len(c_hist) > 0:
                        axes_pdf[1, 0].semilogy(range(1, len(c_hist) + 1), c_hist, 'b-', lw=1.5)
                        axes_pdf[1, 0].axhline(c_final, color='r', ls='--', alpha=0.7, label=f'c*={c_final:.4f}')
                        axes_pdf[1, 0].legend(fontsize=8)
                    axes_pdf[1, 0].set_title(f'Convergencia | {len(c_hist)} iters')
                    axes_pdf[1, 0].set_xlabel('Iteración k')
                    axes_pdf[1, 0].set_ylabel('Compliance c')
                    axes_pdf[1, 0].grid(True, alpha=0.3)

                    # (1,1) Point cloud
                    if nube is not None and len(nube) > 0:
                        axes_pdf[1, 1].scatter(nube[:, 0], nube[:, 1], s=5, alpha=0.5, c='navy')
                    axes_pdf[1, 1].set_xlim(0, nex_)
                    axes_pdf[1, 1].set_ylim(0, ney_)
                    axes_pdf[1, 1].set_aspect('equal')
                    axes_pdf[1, 1].set_title(f'Nube X(ρ*) | β₀={beta0} | β₁={beta1} | μ_α={mu:.4f}')
                    axes_pdf[1, 1].set_xlabel('x (elem.)')
                    axes_pdf[1, 1].set_ylabel('y (elem.)')
                    axes_pdf[1, 1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    pdf.savefig(fig_res_pdf, dpi=200)
                    plt.close(fig_res_pdf)

                    # ── Página 3: Tabla de métricas ────────────────────────────
                    fig_tab = plt.figure(figsize=(8.27, 11.69))
                    fig_tab.patch.set_facecolor('white')
                    ax_tab = fig_tab.add_axes([0.1, 0.1, 0.8, 0.8])
                    ax_tab.axis('off')

                    ax_tab.text(0.5, 0.95, 'Métricas de la Optimización',
                                ha='center', fontsize=18, fontweight='bold', color='#2c3e50',
                                transform=ax_tab.transAxes)

                    # Crear tabla
                    col_labels = ['Métrica', 'Valor', 'Unidad']
                    rows = [
                        ['Compliance final', f'{c_final:.5f}', 'N·mm'],
                        ['Reducción vs base', f'{reduccion:.2f}', '%'],
                        ['Fracción de volumen', f'{volfrac}', '—'],
                        ['Penalización p', f'{penal}', '—'],
                        ['α (peso topológico)', f'{alpha_val}', '—'],
                        ['β₀ (componentes)', str(beta0), '—'],
                        ['β₁ (agujeros)', str(beta1), '—'],
                        ['μ_α (métrica compuesta)', f'{mu:.5f}', '—'],
                        ['Manufacturable', 'Sí' if manufacturable else 'No', '—'],
                        ['Iteraciones', str(n_iter), '—'],
                        ['Convergió', 'Sí' if converged else 'No', '—'],
                        ['Tiempo SIMP', f'{t_simp:.2f}', 's'],
                        ['Tiempo TDA', f'{t_tda:.3f}', 's'],
                        ['Tiempo total', f'{t_simp + t_tda:.2f}', 's'],
                        [f'Malla', f'{nex_}×{ney_}', 'elementos'],
                    ]

                    table = ax_tab.table(cellText=rows, colLabels=col_labels,
                                         loc='center', cellLoc='center',
                                         colWidths=[0.35, 0.25, 0.15])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.6)

                    for (row, col), cell in table.get_celld().items():
                        if row == 0:
                            cell.set_facecolor('#3498db')
                            cell.set_text_props(color='white', fontweight='bold')
                        elif row % 2 == 0:
                            cell.set_facecolor('#f0f3f5')
                        cell.set_edgecolor('#dfe6e9')

                    ax_tab.text(0.5, 0.02,
                                'Generado por Plataforma TDA-SIMP · Jorge Larry Copa Cruz · UAGRM 2026',
                                ha='center', fontsize=7, color='#bdc3c7',
                                transform=ax_tab.transAxes, style='italic')

                    pdf.savefig(fig_tab, dpi=200)
                    plt.close(fig_tab)

            pdf_bytes = pdf_buf.getvalue()

            download_button(
                label="📄 PDF",
                data=pdf_bytes,
                file_name=f"reporte_TDA-SIMP_{nex_}x{ney_}.pdf",
                mime="application/pdf",
                width='stretch',
                help="PDF de 3 páginas: portada + figura + tabla de métricas"
            )

        # ── Exportar PNG 2×2 ───────────────────────────────────────────────
        with col_r2:
            with plt.style.context('default'):
                fig_png, axes_png = plt.subplots(2, 2, figsize=(12, 9))
                fig_png.patch.set_facecolor('white')
                for ax_row in axes_png:
                    for ax in ax_row:
                        ax.set_facecolor('white')
                fig_png.suptitle(f'Resultados SIMP-TDA | {nex_}×{ney_} | f_V={volfrac} | μ_α={mu:.4f}',
                                 fontsize=13, fontweight='bold')

                # Density
                rho_2d = rho_final.reshape(ney_, nex_)
                axes_png[0, 0].imshow(rho_2d, cmap='gray_r', aspect='equal', vmin=0, vmax=1)
                axes_png[0, 0].set_title(f'Distribución ρ*')
                axes_png[0, 0].axis('off')

                # Persistence
                if dgm1 is not None and len(dgm1) > 0:
                    finite = np.isfinite(dgm1[:, 1])
                    if np.any(finite):
                        axes_png[0, 1].scatter(dgm1[finite, 0], dgm1[finite, 1], c='orange', marker='^', alpha=0.7)
                        mv = np.max(dgm1[finite]) * 1.1
                        axes_png[0, 1].plot([0, mv], [0, mv], 'k--', alpha=0.4)
                        if eps_star:
                            axes_png[0, 1].axhline(eps_star / 2, color='r', ls=':', alpha=0.7)
                axes_png[0, 1].set_title('Diagrama H₁')
                axes_png[0, 1].set_xlabel('Birth')
                axes_png[0, 1].set_ylabel('Death')
                axes_png[0, 1].grid(True, alpha=0.3)

                # Convergence
                if c_hist is not None and len(c_hist) > 0:
                    axes_png[1, 0].semilogy(range(1, len(c_hist) + 1), c_hist, 'b-', lw=1.5)
                    axes_png[1, 0].axhline(c_final, color='r', ls='--', alpha=0.7)
                axes_png[1, 0].set_title('Convergencia')
                axes_png[1, 0].set_xlabel('Iteración')
                axes_png[1, 0].set_ylabel('c')
                axes_png[1, 0].grid(True, alpha=0.3)

                # Point cloud
                if nube is not None and len(nube) > 0:
                    axes_png[1, 1].scatter(nube[:, 0], nube[:, 1], s=5, alpha=0.5, c='navy')
                axes_png[1, 1].set_title(f'Nube X(ρ*)')
                axes_png[1, 1].set_xlabel('x')
                axes_png[1, 1].set_ylabel('y')
                axes_png[1, 1].set_aspect('equal')
                axes_png[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                png_buf = io.BytesIO()
                fig_png.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig_png)
                png_bytes = png_buf.getvalue()

            download_button(
                label="🖼️ PNG",
                data=png_bytes,
                file_name=f"figura_resultados_{nex_}x{ney_}.png",
                mime="image/png",
                width='stretch',
                help="Figura 2×2 a 300 DPI lista para la tesis"
            )

        # ── Fila 2: ZIP con todos los datos ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Datos numéricos")

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"optimizacion_{nex_}x{ney_}_{ts}"

        # CSV Métricas
        df_metrics = pd.DataFrame({
            "Metrica": [
                "Compliance_Final", "Beta_0", "Beta_1", "Mu_alpha",
                "Reduccion_vs_homogeneo_%", "Volumen", "Penalizacion_p",
                "Alpha", "Iteraciones", "Convergio", "Tiempo_SIMP_s",
                "Tiempo_TDA_s", "Malla_Nx", "Malla_Ny"
            ],
            "Valor": [
                float(c_final), int(beta0), int(beta1), float(mu),
                float(reduccion), float(volfrac), float(penal),
                float(alpha_val), int(n_iter), 1 if converged else 0,
                float(t_simp), float(t_tda), int(nex_), int(ney_)
            ]
        })

        # CSV Historia
        df_history = pd.DataFrame({
            "Iteracion": range(1, len(c_hist) + 1),
            "Compliance": c_hist
        })

        # CSV Densidades
        rho_flat = rho_final.flatten()
        y_idx, x_idx = np.meshgrid(range(ney_), range(nex_), indexing='ij')
        df_dens = pd.DataFrame({
            "Elemento_X": x_idx.flatten(),
            "Elemento_Y": y_idx.flatten(),
            "Densidad_rho": rho_flat
        })

        # Resumen TXT
        resumen_lines = [
            f"H.E.2 — Optimización SIMP + Métrica Compuesta",
            f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Malla: {nex_}x{ney_}",
            f"Parámetros: f_V={volfrac}, p={penal}, α={alpha_val}, r_min={rmin}",
            f"",
            f"Resultados:",
            f"  Compliance final: {c_final:.8f}",
            f"  Reducción vs base: {reduccion:.2f}%",
            f"  β₀={beta0}, β₁={beta1}",
            f"  μ_α={mu:.8f}",
            f"  Convergió: {'Sí' if converged else 'No'}",
            f"  Iteraciones: {n_iter}",
        ]
        resumen_txt = "\n".join(resumen_lines)

        # Empaquetar ZIP
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{folder_name}/metricas.csv", df_metrics.to_csv(index=False))
            zf.writestr(f"{folder_name}/historial_convergencia.csv", df_history.to_csv(index=False))
            zf.writestr(f"{folder_name}/densidades.csv", df_dens.to_csv(index=False))
            zf.writestr(f"{folder_name}/resumen.txt", resumen_txt)
        zip_bytes = zip_buf.getvalue()

        download_button(
            label="📥 Descargar ZIP (todos los datos)",
            data=zip_bytes,
            file_name=f"{folder_name}.zip",
            mime="application/zip",
            width='stretch',
            help="Metricas + historial + densidades + resumen en una sola carpeta"
        )

        # ── Fila 3: Tabla de métricas expandible ───────────────────────────
        st.markdown("---")
        df_display = pd.DataFrame({
            "Métrica": [
                "Compliance Final c(ρ*)", "β₀ (Componentes conexas)", "β₁ (Agujeros)",
                "μ_α (Métrica Compuesta)", "Reducción vs Sólido",
                "Fracción de Volumen f_V", "Penalización p", "Peso α",
                "Iteraciones", "Convergencia", "Tiempo SIMP", "Tiempo TDA", "Manufacturable"
            ],
            "Valor": [
                f"{c_final:.5f}", str(beta0), str(beta1),
                f"{mu:.5f}", f"{reduccion:.2f}%",
                f"{volfrac}", f"{penal}", f"{alpha_val}",
                str(n_iter), "Sí" if converged else "No",
                f"{t_simp:.3f} s", f"{t_tda:.3f} s",
                "✅ APTA" if manufacturable else "⚠️ REVISAR"
            ]
        })

        with st.expander("📋 Ver tabla completa de métricas", expanded=True):
            st.dataframe(df_display, width='stretch', hide_index=True)

# ════════════════════════════════════════════════════════════════
# METODOLOGÍA
# ════════════════════════════════════════════════════════════════
methodology_expander(
    "📖 Metodología — H.E.2",
    [
        (
            "fórmulas",
            r"""\mu_\alpha = c + \alpha \cdot \beta_1
\quad \text{sujeto a:} \quad K(\rho) \cdot U = F, \quad \int_\Omega \rho \, d\Omega \leq f_V \cdot |\Omega|
\quad \text{Algoritmo 1: inicializar } \rho_e = f_V, \text{ resolver FEM, calcular } c \text{ y } \beta_1,
\quad \text{actualizar } \rho_e \text{ con OC hasta convergencia dual}"""
        )
    ],
    "H.E.2"
)
st.markdown("---")


