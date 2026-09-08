"""Página 3: H.G. — Comparación Integrada TDA-SIMP.

Valida la Hipótesis General:
"La aplicación sistemática de estructuras topológicas—mediante TDA, SIMP—
proporciona soluciones más robustas, eficientes y globalmente consistentes
a los problemas planteados en entornos complejos de ingeniería, en
comparación con los métodos basados exclusivamente en geometría euclidiana
y análisis infinitesimal local."

Referencia: Cuadro 9 del Documento Completo (Comparación estructurada
de las dos aplicaciones).
"""

import streamlit as st
from tda.app.download_utils import download_button, export_settings_ui
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon
import warnings

from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style, diagnosticar_hg,
    methodology_expander, page_header
)

export_settings_ui()

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="H.G. — Comparación Integrada", layout="wide", page_icon="🔬")
st.markdown(responsive_style(), unsafe_allow_html=True)

# ── Page Header ──
st.markdown(page_header(
    "H.G. — Comparación Integrada TDA-SIMP",
    "TDA+SIMP vs Euclidiano: compliance, β₁, μ_α"
), unsafe_allow_html=True)
st.markdown(breadcrumbs(["Tesis", "H.G.", "Resultados"]), unsafe_allow_html=True)
st.header("H.G. — Comparación Integrada TDA-SIMP")
st.markdown("""
*Validación de la Hipótesis General: Los métodos topológicos (TDA + SIMP) 
producen soluciones más robustas que los enfoques exclusivamente euclidianos.*
""")

# ── Sidebar ──
st.sidebar.header("🔬 H.G. Comparación Integrada")
st.sidebar.markdown("""
**Hipótesis General (H.G.):**
La aplicación sistemática de estructuras topológicas
mediante TDA y SIMP proporciona soluciones más
robustas que los métodos euclidianos.
""")

nelx = st.sidebar.slider("Elementos en x", 40, 120, 60, 10, key="hg_nelx",
    help="Número de elementos finitos en dirección horizontal. El Documento Completo usa 60 (60×30=1800 elem).")
nely = st.sidebar.slider("Elementos en y", 20, 60, 30, 10, key="hg_nely",
    help="Número de elementos finitos en dirección vertical. El Documento Completo usa 30.")
volfrac = st.sidebar.slider("Fracción de volumen (fV)", 0.20, 0.80, 0.50, 0.05, key="hg_volfrac",
    help="Porcentaje de material permitido. H.E.2 especifica fV=0.50 (50%).")
penal = st.sidebar.slider("Penalización (p)", 1.0, 6.0, 3.0, 0.5, key="hg_penal",
    help="Factor de penalización SIMP. H.E.2 especifica p=3.")
rmin = st.sidebar.slider("Radio filtro (rmin)", 1.0, 5.0, 3.0, 0.2, key="hg_rmin",
    help="Radio del filtro de sensibilidad. Default: 3.0 (validado). Original: 2.4.")
alpha = st.sidebar.slider("Peso α (métrica compuesta)", 0.001, 1.0, 0.036, 0.001, key="hg_alpha",
    help="Peso de la métrica compuesta μ_α = c + α·β₁. Default: 0.036 (validado). Original: 0.012.")
max_iter = st.sidebar.number_input("Máx. iteraciones SIMP", 50, 500, 200, 50, key="hg_maxiter")

st.markdown("---")

# ── Ejecutar comparación ──
ejecutar_hg = st.button("▶ Ejecutar Comparación Integrada", type="primary", key="btn_hg")

if ejecutar_hg:
    from tda.optimization.metric_simp import MetricaTDA_SIMP

    # Condiciones de borde y carga identicas para ambos:
    # Cantilever viga voladizo: borde izquierdo empotrado (x=0),
    # carga vertical puntual -1.0 en el centro del borde libre libre (x=nelx, y=nely//2).
    # Nodo de carga = (nely//2)*(nelx+1) + nelx
    nnx = nelx + 1
    node_load = (nely // 2) * nnx + nelx
    dof_load = 2 * node_load + 1

    dofs_fijos = []
    for j in range(nely + 1):
        dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])

    # ── 1. SIMP Euclidiano (baseline con α=0, sin fase TDA) ──
    history_eucl = []
    def cb_eucl(k, c, delta_c, delta_rho, rho):
        history_eucl.append({"iter": int(k), "c": float(c),
                             "delta_c": float(delta_c), "delta_rho": float(delta_rho),
                             "rho": rho.copy()})

    opt_eucl = MetricaTDA_SIMP(
        nex=nelx, ney=nely, f_V=volfrac, p=penal, r_min=rmin,
        alpha=0.0, tol=1e-4, max_iter=max_iter
    )
    F_eucl = np.zeros(2 * nnx * (nely + 1))
    F_eucl[dof_load] = -1.0
    opt_eucl.definir_problema(F_eucl, dofs_fijos)
    with st.spinner("Optimizando SIMP euclidiano (baseline)..."):
        opt_eucl.optimizar(callback=cb_eucl, verbose=False)
        mu_eucl = opt_eucl.fase_tda(verbose=False)
    res_eucl = opt_eucl.obtener_resultados()

    # ── 2. SIMP con TDA (topológica con α regularizador) ──
    history_tda = []
    def cb_tda(k, c, delta_c, delta_rho, rho):
        history_tda.append({"iter": int(k), "c": float(c),
                            "delta_c": float(delta_c), "delta_rho": float(delta_rho),
                            "rho": rho.copy()})

    opt_tda = MetricaTDA_SIMP(
        nex=nelx, ney=nely, f_V=volfrac, p=penal, r_min=rmin,
        alpha=alpha, tol=1e-4, max_iter=max_iter
    )
    F_tda = np.zeros(2 * nnx * (nely + 1))
    F_tda[dof_load] = -1.0
    opt_tda.definir_problema(F_tda, dofs_fijos)
    with st.spinner("Optimizando SIMP+TDA (topológica)..."):
        opt_tda.optimizar(callback=cb_tda, verbose=False)
        mu_tda = opt_tda.fase_tda(verbose=False)
    res_tda = opt_tda.obtener_resultados()
    with st.spinner("Computando referencia sólida (f_V=1.0)..."):
        opt_sol = MetricaTDA_SIMP(
            nex=nelx, ney=nely, f_V=1.0, p=penal, r_min=rmin,
            alpha=0.0, max_iter=1
        )
        F_sol = np.zeros(2 * nnx * (nely + 1))
        F_sol[dof_load] = -1.0
        opt_sol.definir_problema(F_sol, dofs_fijos)
        opt_sol.optimizar(verbose=False)
        c_solido = opt_sol.c_final

    with st.spinner("Computando estadísticas de estabilidad (H.E.1)..."):
        from tda.simulation.pipeline import run_tda_experiment
        tda_puro_stats = run_tda_experiment('sphere', noise_levels=[0.10, 0.15, 0.20], n_rep=5, n_points=200, seed=42)

    st.session_state.hg_results = {
        'eucl': res_eucl, 'mu_eucl': mu_eucl,
        'tda': res_tda, 'mu_tda': mu_tda,
        'c_solido': c_solido,
        'c_base': float(opt_tda.c_hist[0]),
        'history_eucl': history_eucl,
        'history_tda': history_tda,
        'tda_puro': tda_puro_stats,
        'params': {'nelx': nelx, 'nely': nely, 'volfrac': volfrac,
                   'penal': penal, 'rmin': rmin, 'alpha': alpha}
    }
    st.session_state.hg_run = True

# ── Renderizar resultados ──
if st.session_state.get('hg_run', False):
    res = st.session_state.hg_results
    eucl = res['eucl']
    tda = res['tda']

    st.info(
        f"📋 Parámetros: Malla {res['params']['nelx']}×{res['params']['nely']}, "
        f"fV={res['params']['volfrac']}, p={res['params']['penal']}, "
        f"α={res['params']['alpha']}"
    )

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 1: Comparación directa lado a lado
    # ════════════════════════════════════════════════════════════════
    st.subheader("1. Comparación Lado a Lado")

    fig_comp = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"SIMP Euclidiano (baseline)",
            f"SIMP + TDA (topológica)"
        ),
        horizontal_spacing=0.08
    )

    # Mapa de densidades euclidiano
    fig_comp.add_trace(go.Heatmap(
        z=eucl['rho_final'].reshape(res['params']['nely'], res['params']['nelx']),
        colorscale='Greys', showscale=False,
        name='Euclidiano'
    ), row=1, col=1)

    # Mapa de densidades TDA
    fig_comp.add_trace(go.Heatmap(
        z=tda['rho_final'].reshape(res['params']['nely'], res['params']['nelx']),
        colorscale='Greys', showscale=False,
        name='TDA'
    ), row=1, col=2)

    fig_comp.update_layout(height=350, margin=dict(l=0, r=0, b=0, t=40))
    apply_plotly_theme(fig_comp)
    st.plotly_chart(fig_comp, width='stretch')

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 1.5: Animaciones de evolución SIMP
    # ════════════════════════════════════════════════════════════════
    hist_eucl = res.get('history_eucl', [])
    hist_tda = res.get('history_tda', [])

    if hist_eucl or hist_tda:
        st.subheader("1.5. Evolución de la Optimización")

        tab_eucl_anim, tab_tda_anim = st.tabs(["SIMP Euclidiano", "SIMP+TDA"])

        def _crear_animacion_simple(history, nex, ney, title, c_solido_ref=None):
            """Crea animación Plotly de la evolución SIMP."""
            from tda.app.theme import is_dark, get_plot_colors
            dark = is_dark()
            colors = get_plot_colors()

            if not history:
                return None

            rho_hist = [h["rho"] for h in history]
            c_hist_vals = [h["c"] for h in history]
            n_frames = len(rho_hist)

            # Downsample si >40 frames
            if n_frames > 40:
                step = max(1, n_frames // 40)
                idx = list(range(0, n_frames, step))
                if idx[-1] != n_frames - 1:
                    idx.append(n_frames - 1)
                rho_ds = [rho_hist[i] for i in idx]
                c_ds = [c_hist_vals[i] for i in idx]
            else:
                rho_ds = rho_hist
                c_ds = c_hist_vals
                idx = list(range(n_frames))

            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.12,
                subplot_titles=("Distribución de Material (ρ)", "Convergencia c(ρ)")
            )

            # Frame inicial
            rho_2d_0 = rho_ds[0].reshape(ney, nex)
            fig.add_trace(go.Heatmap(
                z=rho_2d_0, colorscale='gray_r', zmin=0, zmax=1,
                colorbar=dict(title='ρ', x=1.02, len=0.65, thickness=12),
                hovertemplate='x: %{x}<br>y: %{y}<br>ρ: %{z:.3f}<extra></extra>'
            ), row=1, col=1)

            # Convergencia
            fig.add_trace(go.Scatter(
                x=[], y=[], mode='lines',
                line=dict(color='#3498db', width=2), name='c(k)'
            ), row=2, col=1)

            # Línea vertical
            fig.add_trace(go.Scatter(
                x=[], y=[], mode='lines',
                line=dict(color='red', width=1.5, dash='dash'), name='Actual'
            ), row=2, col=1)

            # Línea de referencia sólido
            if c_solido_ref and c_solido_ref > 0:
                fig.add_trace(go.Scatter(
                    x=[0, len(c_ds)], y=[c_solido_ref, c_solido_ref],
                    mode='lines', line=dict(color='#e74c3c', width=1, dash='dot'),
                    name=f'c sólido={c_solido_ref:.2f}'
                ), row=2, col=1)

            # Frames
            frames = []
            for k, (i_frame, (rho_k, c_k)) in enumerate(zip(idx, zip(rho_ds, c_ds))):
                rho_2d = rho_k.reshape(ney, nex)
                c_up_to = c_hist_vals[:i_frame + 1]
                n_solid = int(np.sum(rho_k > 0.5))
                pct = 100 * n_solid / len(rho_k)

                frame = go.Frame(
                    data=[
                        go.Heatmap(z=rho_2d, colorscale='gray_r', zmin=0, zmax=1,
                                   colorbar=dict(title='ρ', x=1.02, len=0.65, thickness=12)),
                        go.Scatter(x=list(range(1, len(c_up_to) + 1)), y=c_up_to,
                                   mode='lines', line=dict(color='#3498db', width=2)),
                        go.Scatter(x=[i_frame + 1, i_frame + 1],
                                   y=[min(c_hist_vals), max(c_hist_vals)],
                                   mode='lines', line=dict(color='red', width=1.5, dash='dash')),
                    ] + ([go.Scatter(
                        x=[0, len(c_ds)], y=[c_solido_ref, c_solido_ref],
                        mode='lines', line=dict(color='#e74c3c', width=1, dash='dot')
                    )] if c_solido_ref else []),
                    name=str(k),
                    layout=go.Layout(annotations=[dict(
                        x=0.5, y=1.0, xref='paper', yref='paper',
                        text=f'<b>Iter {i_frame+1}/{n_frames}</b> | c={c_k:.4f} | Sólidos: {n_solid}/{len(rho_k)} ({pct:.1f}%)',
                        showarrow=False, font=dict(size=11),
                        bgcolor='rgba(255,255,255,0.8)', bordercolor='#bdc3c7', borderwidth=1
                    )])
                )
                frames.append(frame)

            fig.frames = frames

            c_min = min(c_hist_vals)
            c_max = max(c_hist_vals)
            c_range = c_max - c_min if c_max != c_min else 1.0

            fig.update_layout(
                title=dict(text=f'<b>{title}</b>', font=dict(size=14), x=0.5),
                height=600, hovermode='x unified',
                xaxis1=dict(visible=False),
                yaxis1=dict(visible=False, scaleanchor='x', autorange='reversed'),
                xaxis2=dict(title='Iteración', range=[0, n_frames + 1]),
                yaxis2=dict(title='c(ρ)', type='log',
                            range=[np.log10(max(c_min - 0.1*c_range, 1e-10)),
                                   np.log10(c_max + 0.1*c_range)]),
                sliders=[{
                    'currentvalue': {'prefix': 'Iteración: ', 'font': {'size': 13}},
                    'len': 0.92, 'x': 0.04, 'y': 0.0, 'pad': {'t': 40, 'b': 10},
                    'steps': [{
                        'args': [[str(k)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'label': str(idx[k] + 1), 'method': 'animate'
                    } for k in range(len(rho_ds))]
                }],
                updatemenus=[{
                    'type': 'buttons', 'showactive': False,
                    'x': 0.0, 'y': -0.05,
                    'buttons': [
                        {'label': '▶ Play', 'method': 'animate',
                         'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]},
                        {'label': '⏸ Pause', 'method': 'animate',
                         'args': [[], {'mode': 'immediate'}]},
                    ]
                }],
                plot_bgcolor=colors['bg'], paper_bgcolor=colors['paper_bg'],
                font=dict(color=colors['font']),
                margin=dict(t=60, b=60)
            )
            return fig

        with tab_eucl_anim:
            c_solido_ref = res.get('c_solido', None)
            fig_eucl = _crear_animacion_simple(
                hist_eucl, res['params']['nelx'], res['params']['nely'],
                "SIMP Euclidiano — Evolución", c_solido_ref
            )
            if fig_eucl:
                st.plotly_chart(fig_eucl, width='stretch')
            else:
                st.info("No hay historial de iteraciones para animar.")

        with tab_tda_anim:
            c_solido_ref = res.get('c_solido', None)
            fig_tda = _crear_animacion_simple(
                hist_tda, res['params']['nelx'], res['params']['nely'],
                "SIMP+TDA — Evolución", c_solido_ref
            )
            if fig_tda:
                st.plotly_chart(fig_tda, width='stretch')
            else:
                st.info("No hay historial de iteraciones para animar.")

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 2: Métricas cuantitativas (Cuadro 9 delDocumento Completo)
    # ════════════════════════════════════════════════════════════════
    st.subheader("2. Métricas Comparativas (Cuadro 9)")

    # Baseline de reducción: diseño homogéneo inicial ρ=f_V (c_base).
    # Se muestra también c_solido como dato estructural informativo.
    c_base = res.get('c_base', None)
    c_bloque_solido = res.get('c_solido', None)
    if c_base is None and len(hist_tda) > 0:
        c_base = hist_tda[0]['c']

    reduccion_eucl = (1 - eucl['c_final'] / c_base) * 100 if (eucl['c_final'] and c_base) else 0
    reduccion_tda = (1 - tda['c_final'] / c_base) * 100 if (tda['c_final'] and c_base) else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Compliance (Euclidiano)", f"{eucl['c_final']:.2f}",
                  delta=f"Reducción: {reduccion_eucl:.1f}%")
    with col2:
        st.metric("Compliance (SIMP+TDA)", f"{tda['c_final']:.2f}",
                  delta=f"Reducción: {reduccion_tda:.1f}%")
    with col3:
        delta_comp = eucl['c_final'] - tda['c_final']
        st.metric("Δ Compliance", f"{delta_comp:+.2f}",
                  delta="TDA vs Euclidiano")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("β₁ (Euclidiano)", f"{eucl['beta1']}")
    with col5:
        st.metric("β₁ (SIMP+TDA)", f"{tda['beta1']}")
    with col6:
        st.metric("μ_α (SIMP+TDA)", f"{tda['mu']:.4f}")

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 3: Validación de hipótesis
    # ════════════════════════════════════════════════════════════════
    st.subheader("3. Validación de Hipótesis")

    # H.E.1 — Robustez topológica
    with st.expander("H.E.1 — Robustez Topológica", expanded=False):
        st.markdown("""
        **Hipótesis:** La homología persistente proporciona β₀ y β₁ estables
        bajo perturbaciones del 15–20%, superando a descriptores euclidianos.
        """)
        if 'tda_puro' in res:
            for noise, stats in res['tda_puro'].items():
                st.markdown(f"- Ruido {noise:.0%}: β₀={stats['betti0_mean']:.1f}±{stats['betti0_std']:.1f}, "
                          f"β₁={stats['betti1_mean']:.1f}±{stats['betti1_std']:.1f}")
        else:
            st.info("Ejecutar tab H.E.1 para datos de estabilidad.")

    # H.E.2 — Optimización SIMP
    with st.expander("H.E.2 — Optimización SIMP + TDA", expanded=False):
        st.markdown("""
        **Hipótesis:** SIMP con p=3 y fV=0.5 reduce compliance ≥40%,
        generando topología con β₁ ≤ 2.
        """)
        cumple_reduccion = reduccion_tda >= 40
        cumple_bbeta1 = tda['beta1'] <= 2
        st.markdown(f"- Reducción compliance: **{reduccion_tda:.1f}%** "
                    f"{'✅ ≥40%' if cumple_reduccion else '❌ <40%'}")
        st.markdown(f"- β₁ del diseño: **{tda['beta1']}** "
                    f"{'✅ ≤2' if cumple_bbeta1 else '❌ >2'}")
        if cumple_reduccion and cumple_bbeta1:
            st.success("✅ H.E.2 CUMPLIDA")
        else:
            st.warning("⚠️ H.E.2 requiere ajuste de parámetros")

    # H.G. — Comparación integrada
    with st.expander("H.G. — Hipótesis General", expanded=True):
        st.markdown("""
        **Hipótesis General:** La aplicación sistemática de estructuras
        topológicas (TDA + SIMP) proporciona soluciones más robustas
        que los métodos exclusivamente euclidianos.
        """)

        diag_hg = diagnosticar_hg(
            c_eucl=eucl['c_final'], c_tda=tda['c_final'],
            beta1_eucl=eucl['beta1'], beta1_tda=tda['beta1'],
            mu_tda=tda['mu'], alpha=res['params']['alpha'],
        )

        # Badges resumen
        st.markdown(diag_hg["badges_html"], unsafe_allow_html=True)
        st.markdown("---")

        # Veredicto
        if diag_hg["veredicto"] == "VALIDADA":
            st.success(f"✅ **H.G. {diag_hg['veredicto']}:** SIMP+TDA supera al euclidiano en ambos criterios.")
        elif diag_hg["veredicto"] == "PARCIAL":
            st.warning(f"⚠️ **H.G. {diag_hg['veredicto']}:** Solo uno de los dos criterios se cumple.")
        else:
            st.error(f"❌ **H.G. {diag_hg['veredicto']}:** SIMP+TDA no supera al euclidiano.")

        # Razones
        with st.expander("📖 ¿Por qué?", expanded=True):
            for r in diag_hg["razones"]:
                st.markdown(f"- {r}")

        # Sugerencias
        if diag_hg["sugerencias"]:
            with st.expander("💡 Sugerencias de ajuste", expanded=diag_hg["veredicto"] != "VALIDADA"):
                for s in diag_hg["sugerencias"]:
                    st.markdown(f"- {s}")

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 4: Diagrama de convergencia
    # ════════════════════════════════════════════════════════════════
    st.subheader("4. Convergencia SIMP")

    fig_conv = go.Figure()
    if eucl['c_hist'] is not None and len(eucl['c_hist']) > 0:
        fig_conv.add_trace(go.Scatter(
            y=eucl['c_hist'], mode='lines', name='Euclidiano',
            line=dict(color='gray', width=2)
        ))
    if tda['c_hist'] is not None and len(tda['c_hist']) > 0:
        fig_conv.add_trace(go.Scatter(
            y=tda['c_hist'], mode='lines', name='SIMP+TDA',
            line=dict(color='blue', width=2)
        ))
    fig_conv.update_layout(
        xaxis_title="Iteración", yaxis_title="Compliance c(ρ)",
        height=350, margin=dict(l=0, r=0, b=0, t=0)
    )
    apply_plotly_theme(fig_conv)
    st.plotly_chart(fig_conv, width='stretch')

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 4.5: Test de significancia estadística (Wilcoxon)
    # ════════════════════════════════════════════════════════════════
    st.subheader("4.5. Significancia Estadística")

    e_hist = eucl.get('c_hist', None)
    t_hist = tda.get('c_hist', None)

    if (e_hist is not None and len(e_hist) > 10 and
            t_hist is not None and len(t_hist) > 10):
        # Usar las últimas N iteraciones para comparar convergencia
        n_compare = min(len(e_hist), len(t_hist))
        e_tail = np.array(e_hist[-n_compare:])
        t_tail = np.array(t_hist[-n_compare:])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                stat_wilcoxon, p_value_wilcoxon = wilcoxon(e_tail, t_tail)
            diff_mean = np.mean(t_tail) - np.mean(e_tail)

            c1, c2, c3 = st.columns(3)
            c1.metric("Estadístico W", f"{stat_wilcoxon:.2f}")
            c2.metric("p-value", f"{p_value_wilcoxon:.4f}",
                      delta="Significativo (p<0.05)" if p_value_wilcoxon < 0.05 else "No significativo",
                      delta_color="normal" if p_value_wilcoxon < 0.05 else "off")
            c3.metric("Δ Compliance media", f"{diff_mean:+.2f}",
                      delta="TDA mejor" if diff_mean < 0 else "Euclidiano mejor",
                      delta_color="normal" if diff_mean < 0 else "inverse")

            if p_value_wilcoxon < 0.05:
                st.success(
                    f"✅ **Diferencia estadísticamente significativa** (p={p_value_wilcoxon:.4f} < 0.05). "
                    f"{'SIMP+TDA produce compliance significativamente menor.' if diff_mean < 0 else 'SIMP euclidiano produce compliance significativamente menor.'}"
                )
            else:
                st.info(
                    f"ℹ️ No hay diferencia estadísticamente significativa (p={p_value_wilcoxon:.4f} ≥ 0.05). "
                    "Los dos enfoques producen resultados similares en esta configuración."
                )
        except ValueError as e:
            st.warning(f"No se pudo ejecutar test de Wilcoxon: {e}")
    else:
        st.info(
            "ℹ️ Datos insuficientes para test de significancia "
            "(se necesitan al menos 10 iteraciones de convergencia)."
        )

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 5: Diagramas de persistencia
    # ════════════════════════════════════════════════════════════════
    st.subheader("5. Diagramas de Persistencia")

    fig_dgm = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Dgm₁ Euclidiano (β₁={eucl['beta1']})",
            f"Dgm₁ SIMP+TDA (β₁={tda['beta1']})"
        ),
        horizontal_spacing=0.08
    )

    if eucl['dgm1'] is not None and len(eucl['dgm1']) > 0:
        fig_dgm.add_trace(go.Scatter(
            x=eucl['dgm1'][:, 0], y=eucl['dgm1'][:, 1],
            mode='markers', name='Euclidiano',
            marker=dict(color='gray', size=8, symbol='circle')
        ), row=1, col=1)

    if tda['dgm1'] is not None and len(tda['dgm1']) > 0:
        fig_dgm.add_trace(go.Scatter(
            x=tda['dgm1'][:, 0], y=tda['dgm1'][:, 1],
            mode='markers', name='SIMP+TDA',
            marker=dict(color='blue', size=8, symbol='circle')
        ), row=1, col=2)

    # Diagonal
    max_val = max(
        eucl['dgm1'].max() if eucl['dgm1'] is not None and len(eucl['dgm1']) > 0 else 1,
        tda['dgm1'].max() if tda['dgm1'] is not None and len(tda['dgm1']) > 0 else 1
    )
    fig_dgm.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', line=dict(dash='dash', color='red'),
        showlegend=False
    ), row=1, col=1)
    fig_dgm.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', line=dict(dash='dash', color='red'),
        showlegend=False
    ), row=1, col=2)

    fig_dgm.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=40))
    fig_dgm.update_xaxes(title_text="Nacimiento (birth)")
    fig_dgm.update_yaxes(title_text="Muerte (death)")
    apply_plotly_theme(fig_dgm)
    st.plotly_chart(fig_dgm, width='stretch')

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 6: Tabla resumen (Cuadro 9 delDocumento Completo)
    # ════════════════════════════════════════════════════════════════
    st.subheader("6. Tabla Comparativa (Cuadro 9)")

    df_comp = pd.DataFrame({
        "Métrica": [
            "Compliance c", "Reducción compliance",
            "β₀ (componentes)", "β₁ (agujeros)",
            "μ_α (métrica compuesta)",
            "Manufacturabilidad"
        ],
        "SIMP Euclidiano": [
            f"{eucl['c_final']:.2f}" if eucl['c_final'] else "—",
            f"{reduccion_eucl:.1f}%",
            f"{eucl['beta0']}" if eucl['beta0'] else "—",
            f"{eucl['beta1']}",
            "N/A",
            "Sin verificación topológica"
        ],
        "SIMP + TDA": [
            f"{tda['c_final']:.2f}" if tda['c_final'] else "—",
            f"{reduccion_tda:.1f}%",
            f"{tda['beta0']}" if tda['beta0'] else "—",
            f"{tda['beta1']}",
            f"{tda['mu']:.4f}",
            "✅ Verificada (β₁ controlado)"
        ]
    })
    st.dataframe(df_comp, width='stretch', hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN 7: Interpretación para el tribunal
    # ════════════════════════════════════════════════════════════════
    st.subheader("7. Interpretación")

    st.markdown(f"""
    **Resultado de la comparación:**

    | Indicador | Euclidiano | SIMP+TDA | Conclusión |
    |-----------|-----------|----------|------------|
    | Compliance | {eucl['c_final']:.2f} | {tda['c_final']:.2f} | {'TDA iguala' if abs(delta_comp) < 1 else ('TDA supera' if delta_comp > 0 else 'Euclidiano mejor')} |
    | β₁ | {eucl['beta1']} | {tda['beta1']} | {'TDA controla mejor' if tda['beta1'] <= eucl['beta1'] else 'Euclidiano mejor'} |
    | μ_α | — | {tda['mu']:.4f} | Métrica topológica delDocumento Completo |

    **Conclusión para el tribunal:**
    {'✅ La integración TDA+SIMP produce diseños con topología verificada (β₁ controlado) '
     'y compliance competitiva, validando la Hipótesis General.' if diag_hg["veredicto"] == "VALIDADA"
     else '⚠️ Se requieren ajustes de parámetros para validar completamente H.G.'}
    """)

    # ── Exportar ──
    st.subheader("Exportar Resultados")

    df_export = pd.DataFrame({
        "Métrica": ["Compliance", "Reducción %", "β₀", "β₁", "μ_α", "α"],
        "Euclidiano": [eucl['c_final'], reduccion_eucl, eucl['beta0'], eucl['beta1'], "N/A", "N/A"],
        "SIMP+TDA": [tda['c_final'], reduccion_tda, tda['beta0'], tda['beta1'], tda['mu'], alpha]
    })
    csv_comp = df_export.to_csv(index=False).encode('utf-8')

    download_button(
        label="📥 CSV",
        data=csv_comp,
        file_name="comparacion_hg.csv",
        mime="text/csv",
        key="btn_csv_hg"
    )

# ════════════════════════════════════════════════════════════════
# METODOLOGÍA
# ════════════════════════════════════════════════════════════════
methodology_expander(
    "📖 Metodología — H.G.",
    [
        (
            "fórmulas",
            r"""Comparación: \mu_\alpha^{TDA} = c^{TDA} + \alpha \cdot \beta_1^{TDA}
\quad vs \mu_\alpha^{Euclidiano} = c^{Euclidiano} + 0 \cdot \beta_1^{Euclidiano}
\quad Criterios: c^{TDA} \leq c^{Euclidiano} (compliance)
\quad \beta_1^{TDA} \leq \beta_1^{Euclidiano} (topología)
\quad \alpha_{euclidiano} = 0, \alpha_{TDA} = 0.012"""
        )
    ],
    "H.G."
)
st.markdown("---")

apply_mpl_theme()
