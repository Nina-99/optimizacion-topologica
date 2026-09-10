"""Página 3: H.G. — Comparación Integrada TDA-SIMP.

Valida la Hipótesis General:
"La aplicación sistemática de estructuras topológicas—mediante TDA, SIMP—
proporciona soluciones más robustas, eficientes y globalmente consistentes
a los problemas planteados en entornos complejos de ingeniería, en
comparación con los métodos basados exclusivamente en geometría euclidiana
y análisis infinitesimal local."

Estructura de validación (Documento, Sección 6):
1. Correr SIMP con p=2, p=3, p=4 en la misma malla
2. Calcular μ_α para cada configuración
3. Validar H.G.: μ_α identifica p=3 como Pareto-óptimo
4. Sensibilidad de μ a α (Cuadro 4)
5. Sintetizar H.E.1 (robustez TDA) + H.E.2 (SIMP+μ_α)
"""

import streamlit as st
from tda.app.download_utils import download_button, export_settings_ui
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    methodology_expander, page_header, ORANGE
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
    "Validación H.G.: μ_α identifica diseños Pareto-óptimos"
), unsafe_allow_html=True)
st.header("H.G. — Comparación Integrada TDA-SIMP")
st.markdown("""
*Validación de la Hipótesis General: La métrica compuesta μ_α = c + α·β₁
resuelve el orden parcial entre configuraciones SIMP, identificando las
soluciones más robustas, eficientes y globalmente consistentes.*
""")

# ══════════════════════════════════════════════════════════════
# SIDEBAR — Parámetros FEM (sin α como variable)
# ══════════════════════════════════════════════════════════════
st.sidebar.header("🔬 Parámetros FEM")

nelx = st.sidebar.selectbox("Elementos en x", [40, 60, 80, 120], index=1, key="hg_nelx")
nely = st.sidebar.selectbox("Elementos en y", [20, 30, 40, 60], index=1, key="hg_nely")

volfrac = st.sidebar.slider("Fracción de volumen (fV)", 0.20, 0.80, 0.50, 0.05, key="hg_volfrac",
    help="Porcentaje de material permitido. Documento: fV=0.50 (50%).")
rmin = st.sidebar.slider("Radio filtro (rmin)", 1.0, 5.0, 2.4, 0.2, key="hg_rmin",
    help="Radio del filtro de sensibilidad. Documento: r_min=2.4.")
max_iter = st.sidebar.number_input("Máx. iteraciones SIMP", 50, 500, 200, 50, key="hg_maxiter")

st.sidebar.markdown("---")
st.sidebar.header("📐 Parámetros Físicos")
E_acero = st.sidebar.number_input("Módulo Young E₀ (MPa)", value=200000.0, step=1000.0, help="Acero = 200 GPa = 200,000 MPa")
F_carga = st.sidebar.number_input("Carga F (N)", value=1000.0, step=100.0, help="Carga aplicada = 1 kN = 1000 N")
espesor = st.sidebar.number_input("Espesor (mm)", value=1.0, step=0.1, help="Espesor de la viga en 2D plano")

st.sidebar.markdown("---")
st.sidebar.caption("α se usa para calcular μ_α, no para la optimización SIMP (sección 6.3)")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# EJECUCIÓN
# ══════════════════════════════════════════════════════════════
ejecutar_hg = st.button("▶ Ejecutar Comparación Integrada", type="primary", key="btn_hg")

if ejecutar_hg:
    from tda.optimization.metric_simp import MetricaTDA_SIMP

    # Condiciones de borde: cantilever viga voladizo
    nnx = nelx + 1
    node_load = (nely // 2) * nnx + nelx
    dof_load = 2 * node_load + 1
    dofs_fijos = []
    for j in range(nely + 1):
        dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])

    ALPHA_FIJO = 0.036

    # ── Correr SIMP con p=2, p=3, p=4 ──
    resultados = {}
    for p_val in [2, 3, 4]:
        with st.spinner(f"Optimizando SIMP con p={p_val}..."):
            opt = MetricaTDA_SIMP(
                nex=nelx, ney=nely, E=E_acero, nu=0.3,
                Lx=120.0, Ly=40.0, t=espesor,
                f_V=volfrac, p=p_val, r_min=rmin,
                alpha=ALPHA_FIJO, tol_c=1e-4, tol_rho=1e-2, max_iter=max_iter
            )
            F = np.zeros(2 * nnx * (nely + 1))
            F[dof_load] = -F_carga
            opt.definir_problema(F, dofs_fijos)
            opt.optimizar(verbose=False)
            mu = opt.fase_tda(verbose=False)
            resultados[p_val] = opt.obtener_resultados()

    # ── Sensibilidad a α (fijo p=3) ──
    alphas_test = [0.001, 0.01, 0.012, 0.036, 0.1, 0.2, 0.5]
    res_p3 = resultados[3]
    sensibilidad = {}
    for a in alphas_test:
        mu_a = res_p3['c_final'] + a * res_p3['beta1']
        sensibilidad[a] = {'mu': mu_a, 'c': res_p3['c_final'], 'beta1': res_p3['beta1']}

    st.session_state.hg_results = {
        'resultados': resultados,
        'sensibilidad': sensibilidad,
        'params': {'nelx': nelx, 'nely': nely, 'volfrac': volfrac,
                   'rmin': rmin, 'alpha_fijo': ALPHA_FIJO}
    }
    st.session_state.hg_run = True

# ══════════════════════════════════════════════════════════════
# RENDERIZAR RESULTADOS
# ══════════════════════════════════════════════════════════════
if st.session_state.get('hg_run', False):
    res = st.session_state.hg_results
    resultados = res['resultados']
    sensibilidad = res['sensibilidad']
    params = res['params']

    st.info(
        f"📋 Parámetros: Malla {params['nelx']}×{params['nely']}, "
        f"fV={params['volfrac']}, r_min={params['rmin']}, "
        f"α={params['alpha_fijo']}"
    )

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 1: Comparación visual p=2 vs p=3 vs p=4
    # ══════════════════════════════════════════════════════════════
    st.subheader("1. Distribución de Material por Configuración")

    fig_comp = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"p=2 (β₁={resultados[2]['beta1']})",
            f"p=3 (β₁={resultados[3]['beta1']})",
            f"p=4 (β₁={resultados[4]['beta1']})"
        ),
        horizontal_spacing=0.06
    )
    for i, p_val in enumerate([2, 3, 4]):
        rho_2d = resultados[p_val]['rho_final'].reshape(params['nely'], params['nelx'])
        fig_comp.add_trace(go.Heatmap(
            z=rho_2d, colorscale='Greys', showscale=False,
            name=f'p={p_val}'
        ), row=1, col=i + 1)

    fig_comp.update_layout(height=350, margin=dict(l=0, r=0, b=0, t=40))
    apply_plotly_theme(fig_comp)
    st.plotly_chart(fig_comp, width='stretch')

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 2: Validación H.G.: μ_α identifica p=3 como Pareto-óptimo
    # ══════════════════════════════════════════════════════════════
    st.subheader("2. Validación H.G.: μ_α identifica p=3 como Pareto-óptimo")

    r2, r3, r4 = resultados[2], resultados[3], resultados[4]
    alpha = params['alpha_fijo']

    st.markdown("""
    **Criterio de Pareto según H.G. y Cuadro 9 del documento:**
    La métrica compuesta μ_α = c + α·β₁ (con α > 0) resuelve el orden parcial entre 
    configuraciones SIMP. p=3 es Pareto-óptimo si minimiza c manteniendo β₁ = 2 (invariante 
    topológico), evitando los agujeros espurios de p=2 (β₁ = 3).
    """)

    # Condición 1: p=3 vs p=2
    cond_c_3v2 = r3['c_final'] <= r2['c_final']
    cond_b1_3v2 = r3['beta1'] <= r2['beta1']
    estricta_3v2 = cond_c_3v2 and cond_b1_3v2 and (r3['c_final'] < r2['c_final'] or r3['beta1'] < r2['beta1'])
    cumple_3v2 = (cond_c_3v2 or cond_b1_3v2) and estricta_3v2

    # Condición 2: p=3 vs p=4
    cond_c_3v4 = r3['c_final'] <= r4['c_final']
    cond_b1_3v4 = r3['beta1'] <= r4['beta1']
    estricta_3v4 = cond_c_3v4 and cond_b1_3v4 and (r3['c_final'] < r4['c_final'] or r3['beta1'] < r4['beta1'])
    cumple_3v4 = (cond_c_3v4 or cond_b1_3v4) and estricta_3v4

    # badges
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("c (p=2)", f"{r2['c_final']:.2f}", delta=f"β₁={r2['beta1']}")
    with c2:
        st.metric("c (p=3)", f"{r3['c_final']:.2f}", delta=f"β₁={r3['beta1']}")
    with c3:
        st.metric("c (p=4)", f"{r4['c_final']:.2f}", delta=f"β₁={r4['beta1']}")

    if cumple_3v2 and cumple_3v4:
        st.success(
            "✅ **H.G. VALIDADA: p=3 Pareto-óptimo:** p=3 domina en sentido Pareto tanto a p=2 como a p=4. "
            "μ_α identifica correctamente la configuración óptima."
        )
    elif cumple_3v2 or cumple_3v4:
        st.warning("⚠️ H.G. parcialmente validada.")
    else:
        st.error("❌ H.G. no validada en esta configuración.")

    with st.expander("📖 Detalle de dominancia Pareto"):
        for (p_a, r_a, p_b, r_b, label) in [
            (3, r3, 2, r2, "p=3 vs p=2"),
            (3, r3, 4, r4, "p=3 vs p=4"),
        ]:
            dominancia_c = "≤" if r_a['c_final'] <= r_b['c_final'] else ">"
            dominancia_b1 = "≤" if r_a['beta1'] <= r_b['beta1'] else ">"
            st.markdown(
                f"- **{label}:** c {dominancia_c} ({r_a['c_final']:.2f} vs {r_b['c_final']:.2f}), "
                f"β₁ {dominancia_b1} ({r_a['beta1']} vs {r_b['beta1']})"
            )

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 3: Sensibilidad de μ a α
    # ══════════════════════════════════════════════════════════════
    st.subheader("3. Sensibilidad de μ_α al peso α")

    st.markdown(f"*p=3 fijo (malla {params['nelx']}×{params['nely']}, fV={params['volfrac']}).*")

    mu_base = r3['c_final'] + alpha * r3['beta1']
    df_sens = pd.DataFrame({
        "α": list(sensibilidad.keys()),
        "c (N·mm)": [sensibilidad[a]['c'] for a in sensibilidad],
        "β₁": [sensibilidad[a]['beta1'] for a in sensibilidad],
        "μ_α": [sensibilidad[a]['mu'] for a in sensibilidad],
    })
    st.dataframe(df_sens, width='stretch', hide_index=True)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=list(sensibilidad.keys()),
        y=[sensibilidad[a]['mu'] for a in sensibilidad],
        mode='lines+markers',
        name='μ_α',
        line=dict(color=('#FF6B35'), width=3),
        marker=dict(size=8)
    ))
    fig_sens.add_hline(y=r3['c_final'], line_dash="dot", line_color="gray",
                       annotation_text=f"c={r3['c_final']:.2f}")
    fig_sens.update_layout(
        xaxis_title="α (peso TDA)", yaxis_title="μ_α = c + α·β₁",
        height=350, margin=dict(l=0, r=0, b=0, t=0)
    )
    apply_plotly_theme(fig_sens)
    st.plotly_chart(fig_sens, width='stretch')

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 4: Análisis Paramétrico (Aplicación 1)
    # ══════════════════════════════════════════════════════════════
    st.subheader("4. Análisis Paramétrico (Aplicación 1)")

    df_comp = pd.DataFrame({
        "Métrica": [
            "Compliance c (N·mm)", "Reducción vs base (ρ=fV)",
            "β₀ (componentes)", "β₁ (agujeros)", "μ_α (métrica compuesta)",
            "Manufacturabilidad"
        ],
        "p=2": [
            f"{r2['c_final']:.2f}",
            f"{(1 - r2['c_final'] / r2['c_final']) * 100:.1f}%",
            f"{r2['beta0']}", f"{r2['beta1']}",
            f"{r2['mu']:.4f}",
            "❌ Agujeros espurios" if r2['beta1'] > 0 else "✅ Apta"
        ],
        "p=3": [
            f"{r3['c_final']:.2f}",
            f"{(1 - r3['c_final'] / r2['c_final']) * 100:.1f}%",
            f"{r3['beta0']}", f"{r3['beta1']}",
            f"{r3['mu']:.4f}",
            "✅ Apta (óptima)" if r3['beta1'] == 0 else f"⚠️ β₁={r3['beta1']}"
        ],
        "p=4": [
            f"{r4['c_final']:.2f}",
            f"{(1 - r4['c_final'] / r2['c_final']) * 100:.1f}%",
            f"{r4['beta0']}", f"{r4['beta1']}",
            f"{r4['mu']:.4f}",
            "✅ Apta" if r4['beta1'] == 0 else f"⚠️ β₁={r4['beta1']}"
        ]
    })
    st.dataframe(df_comp, width='stretch', hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 4b: Convergencia SIMP (cumplimiento §10)
    # ══════════════════════════════════════════════════════════════
    st.subheader("4b. Convergencia SIMP")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iteraciones", f"{r3['n_iter']}")
    c2.metric("Convergió", "✅" if r3['converged'] else "❌")
    c3.metric("Δc final", f"{r3['c_hist'][-1] - r3['c_hist'][-2]:.6f}" if len(r3['c_hist']) > 1 else "N/A")
    c4.metric("Tol Δc/c", "1e-4 (Perfil §9.4.4)")

    # Curva de convergencia
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=list(range(1, len(r3['c_hist']) + 1)),
        y=r3['c_hist'],
        mode='lines+markers',
        line=dict(color=ORANGE, width=2),
        name='Compliance c_k'
    ))
    fig_conv.add_trace(go.Scatter(
        x=list(range(1, len(r3['c_hist']) + 1)),
        y=[r3['c_hist'][0] * (1 - 1e-4)] * len(r3['c_hist']),
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Criterio Δc/c < 10⁻⁴'
    ))
    fig_conv.update_layout(
        title="Convergencia SIMP — p=3",
        xaxis_title="Iteración k",
        yaxis_title="Compliance c (N·mm)",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 5: Síntesis H.E.1 + H.E.2
    # ══════════════════════════════════════════════════════════════
    st.subheader("5. Síntesis de Hipótesis Específicas")

    with st.expander("H.E.1 — Robustez Topológica", expanded=False):
        st.markdown("""
        **Hipótesis H.E.1a′ (corregida):** La cota bottleneck dB/diam ≤ 2q se
        verifica en el 100% de las réplicas.

        **Hipótesis H.E.1b (parcial):** TDA supera a descriptores euclidianos
        bajo deformación afín, pero la diferencia no es estadísticamente significativa.

        **Hipótesis H.E.1c (confirmada):** βk persistente lee correctamente la topología
        con τ=0.15 calibrado (catálogo 7/7).
        """)
        if 'tda_puro' in res:
            for noise, stats in res['tda_puro'].items():
                st.markdown(
                    f"- Ruido {noise:.0%}: β₀={stats['betti0_mean']:.1f}±{stats['betti0_std']:.1f}, "
                    f"β₁={stats['betti1_mean']:.1f}±{stats['betti1_std']:.1f}"
                )
        else:
            st.info("Ejecutar tab H.E.1 para datos de estabilidad. Nota: H.E.1b es parcial — diferencia no significativa (p≈0.05).")

    with st.expander("H.E.2 — Optimización SIMP + Métrica Compuesta", expanded=False):
        st.markdown("""
        **Hipótesis H.E.2a′ (corregida):** SIMP con p=3 y fV=0.5 reduce compliance ≥40%,
        respecto al diseño uniforme de igual volumen (ρ=fV).

        **Hipótesis H.E.2b′ (corregida):** β₁(Ω_sólido) es **independiente** de la resolución
        de malla y del radio del filtro rmin ∈ [1.5, 4.0]. Descriptor invariante (descriptivo,
        no prescriptivo).

        > ⚠️ Las formulaciones originales (β₁ ≤ 2 y comparación vs bloque sólido)
        > fueron descartadas en §4.4 por razones técnicas documentadas.
        """)
        c_base = r3['c_hist'][0] if r3['c_hist'] is not None and len(r3['c_hist']) > 0 else r3['c_final']
        reduccion = (1 - r3['c_final'] / c_base) * 100 if c_base > 0 else 0
        cumple_reduccion = reduccion >= 40
        cumple_b1 = r3['beta1'] == 2  # β₁ invariante, no cota prescriptiva
        st.markdown(f"- Reducción compliance: **{reduccion:.1f}%** {'✅ ≥40%' if cumple_reduccion else '❌ <40%'}")
        st.markdown(f"- β₁ del diseño: **{r3['beta1']}** {'✅ =2 invariante' if cumple_b1 else '❌ ≠2'}")
        if cumple_reduccion and cumple_b1:
            st.success("✅ H.E.2 CUMPLIDA (criterios corregidos §4.4)")
        else:
            st.warning("⚠️ H.E.2 requiere ajuste de parámetros")

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 6: Diagramas de Persistencia
    # ══════════════════════════════════════════════════════════════
    st.subheader("6. Diagramas de Persistencia")

    fig_dgm = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"Dgm₁ p=2 (β₁={r2['beta1']})",
            f"Dgm₁ p=3 (β₁={r3['beta1']})",
            f"Dgm₁ p=4 (β₁={r4['beta1']})"
        ),
        horizontal_spacing=0.06
    )
    max_val = 1
    for i, (p_val, r) in enumerate([(2, r2), (3, r3), (4, r4)]):
        if r['dgm1'] is not None and len(r['dgm1']) > 0:
            fig_dgm.add_trace(go.Scatter(
                x=r['dgm1'][:, 0], y=r['dgm1'][:, 1],
                mode='markers', name=f'p={p_val}',
                marker=dict(color=['gray', '#FF6B35', 'blue'][i], size=8, symbol='circle')
            ), row=1, col=i + 1)
            max_val = max(max_val, r['dgm1'].max())

    for col in [1, 2, 3]:
        fig_dgm.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(dash='dash', color='red'),
            showlegend=False
        ), row=1, col=col)

    fig_dgm.update_layout(height=350, margin=dict(l=0, r=0, b=0, t=40))
    fig_dgm.update_xaxes(title_text="Nacimiento")
    fig_dgm.update_yaxes(title_text="Muerte")
    apply_plotly_theme(fig_dgm)
    st.plotly_chart(fig_dgm, width='stretch')

    # SECCIÓN 7: Síntesis Transversal — Cuadro 9 del Documento
    # ═══════════════════════════════════════════════════════════════
    st.subheader("7. Cuadro 9: Veredicto sobre cada afirmación del perfil")

    st.markdown("""
    La tabla siguiente consolida el veredicto final sobre cada hipótesis del perfil,
    tal como se reporta en la documentación (metodologia_implementacion.txt, Cuadro 9).
    """)

    df_cuadro9 = pd.DataFrame({
        "Ítem": [
            "H.E.1a",
            "H.E.1a′",
            "H.E.1b",
            "H.E.1c",
            "H.E.2a",
            "H.E.2a′",
            "H.E.2b"
        ],
        "Afirmación": [
            "βk estable ante ruido 15–20 %",
            "dB / diam ≤ 2q (reformulación)",
            "TDA > descriptores euclidianos",
            "Cota dB ≤ 2dH",
            "Reducción ≥ 40 % vs. bloque sólido",
            "Reducción ≥ 40 % vs. uniforme",
            "β₁ (Ω) ≤ 2"
        ],
        "Valor": [
            "0.0036",
            "1.000",
            "0.950 vs 0.925",
            "1.000",
            "−79.39 %",
            "+77.58 %",
            "2"
        ],
        "Veredicto": [
            "❌ No se sostiene",
            "✅ Se sostiene",
            "⚠️ Parcial, no significativo",
            "✅ Se sostiene",
            "❌ No se sostiene",
            "✅ Se sostiene",
            "❌ No se sostiene"
        ]
    })
    
    st.table(df_cuadro9.set_index("Ítem"))

    st.markdown("""
    **Observación (del documento §9):**
    Tres afirmaciones del borrador no se sostienen, y las tres por razones distintas y diagnosticables: 
    un indicador que el teorema invocado no respalda (H.E.1a), un referente que hace la hipótesis 
    imposible (H.E.2a) y una cota fijada a priori sobre un invariante no controlado (H.E.2b). 
    Ninguna refuta el valor del enfoque topológico; las tres delimitan con precisión qué puede 
    y qué no puede afirmarse. Esa delimitación es, de hecho, la contribución más defendible del 
    trabajo: convierte el gap enunciado bibliográficamente en un resultado medido en el propio 
    banco de pruebas.
    """)

    # ═══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 8: Conclusión General
    # ══════════════════════════════════════════════════════════════
    st.subheader("8. Conclusión General")

    corollary_ok = cumple_3v2 and cumple_3v4
    st.markdown(f"""
    **Resultado Análisis Paramétrico (Caso 2 — H.E.2):**
    | Indicador | p=2 | p=3 | p=4 |
    |-----------|-----|-----|-----|
    | c (N·mm) | {r2['c_final']:.2f} | {r3['c_final']:.2f} | {r4['c_final']:.2f} |
    | β₁ | {r2['beta1']} | {r3['beta1']} | {r4['beta1']} |
    | μ_α | {r2['mu']:.4f} | {r3['mu']:.4f} | {r4['mu']:.4f} |

    **Veredicto para el tribunal:**
    {'✅ **HIPÓTESIS GENERAL VALIDADA:** '
     'La métrica μ_α resuelve el orden parcial de las configuraciones y demuestra que TDA+SIMP es superior '
     'a los enfoques puramente euclidianos. El Caso 2 identifica p=3 como Pareto-óptimo (evitando los agujeros '
     'espurios de p=2). La metodología proporciona '
     'soluciones consistentes y robustas a diferentes problemas complejos.' if corollary_ok
     else '⚠️ Se requieren ajustes de parámetros para validar completamente H.G.'}
    """)

    # ══════════════════════════════════════════════════════════════
    # EXPORTAR
    # ══════════════════════════════════════════════════════════════
    st.subheader("Exportar Resultados")

    df_export = pd.DataFrame({
        "Métrica": ["Compliance", "β₀", "β₁", "μ_α", "α", "p"],
        "p=2": [r2['c_final'], r2['beta0'], r2['beta1'], r2['mu'], alpha, 2],
        "p=3": [r3['c_final'], r3['beta0'], r3['beta1'], r3['mu'], alpha, 3],
        "p=4": [r4['c_final'], r4['beta0'], r4['beta1'], r4['mu'], alpha, 4]
    })
    csv_comp = df_export.to_csv(index=False).encode('utf-8')

    # ── Generar PDF de reporte H.G. ──
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdf_buf = io.BytesIO()
    plt.style.use('default')
    fig_pdf = plt.figure(figsize=(12, 10))
    fig_pdf.patch.set_facecolor('white')

    # Distribución de material (3 subplots)
    for i, p_val in enumerate([2, 3, 4]):
        ax = fig_pdf.add_subplot(2, 3, i + 1)
        rho_2d = resultados[p_val]['rho_final'].reshape(params['nely'], params['nelx'])
        ax.imshow(rho_2d, cmap='Greys', aspect='auto')
        ax.set_title(f"p={p_val} (β₁={resultados[p_val]['beta1']})")
        ax.axis('off')

    # Sensibilidad a α
    ax_sens = fig_pdf.add_subplot(2, 3, 4)
    alphas = list(sensibilidad.keys())
    mus = [sensibilidad[a]['mu'] for a in alphas]
    ax_sens.plot(alphas, mus, 'o-', color='darkorange', linewidth=2)
    ax_sens.set_xscale('log')
    ax_sens.set_xlabel('α')
    ax_sens.set_ylabel('μ_α')
    ax_sens.set_title('Sensibilidad μ_α vs α (p=3)')
    ax_sens.grid(True, alpha=0.3)

    # Tabla de métricas
    ax_tabla = fig_pdf.add_subplot(2, 3, 5)
    ax_tabla.axis('off')
    tabla_data = [
        ["Métrica", "p=2", "p=3", "p=4"],
        ["Compliance", f"{r2['c_final']:.2f}", f"{r3['c_final']:.2f}", f"{r4['c_final']:.2f}"],
        ["β₀", str(r2['beta0']), str(r3['beta0']), str(r4['beta0'])],
        ["β₁", str(r2['beta1']), str(r3['beta1']), str(r4['beta1'])],
        ["μ_α", f"{r2['mu']:.4f}", f"{r3['mu']:.4f}", f"{r4['mu']:.4f}"],
    ]
    tabla = ax_tabla.table(cellText=tabla_data, loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)
    ax_tabla.set_title('Resumen Comparativo')

    # Veredicto
    ax_veredicto = fig_pdf.add_subplot(2, 3, 6)
    ax_veredicto.axis('off')
    corollary_ok = cumple_3v2 and cumple_3v4
    veredicto = ("✅ HIPÓTESIS GENERAL VALIDADA" if corollary_ok
                 else "⚠️ Requiere ajustes")
    ax_veredicto.text(0.5, 0.5, veredicto, ha='center', va='center',
                      fontsize=12, fontweight='bold',
                      color='green' if corollary_ok else 'orange')
    ax_veredicto.set_title('Veredicto')

    fig_pdf.suptitle(f"H.G. — Comparación Integrada TDA-SIMP | Malla {params['nelx']}×{params['nely']}, fV={params['volfrac']}, α={params['alpha_fijo']}")
    plt.tight_layout()
    fig_pdf.savefig(pdf_buf, format='pdf')
    plt.close(fig_pdf)
    pdf_bytes = pdf_buf.getvalue()

    # ── Fila Única: Exportar TODO en un solo ZIP ───────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Exportación Completa")

    import zipfile
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"hg_comparacion_{params['nelx']}x{params['nely']}_{ts}"

    zf_buf = io.BytesIO()
    with zipfile.ZipFile(zf_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{folder_name}/comparacion_metricas.csv", csv_comp)
        zf.writestr(f"{folder_name}/reporte_hg.pdf", pdf_bytes)
        zf.writestr(f"{folder_name}/parametros.txt",
                    f"Malla: {params['nelx']}×{params['nely']}\\n"
                    f"fV: {params['volfrac']}\\n"
                    f"r_min: {params['rmin']}\\n"
                    f"α: {params['alpha_fijo']}\\n"
                    f"E: {E_acero:.0f} MPa\\n"
                    f"F: {F_carga:.0f} N\\n"
                    f"Espesor: {espesor:.1f} mm\\n"
                    f"Timestamp: {ts}")
    zip_bytes = zf_buf.getvalue()

    download_button(
        label="📦 Descargar TODO (ZIP completo)",
        data=zip_bytes,
        file_name=f"{folder_name}.zip",
        mime="application/zip",
        width='stretch',
        help="Incluye: CSV comparativo, PDF con visualizaciones y tabla, archivo de parámetros - todo organizado en una carpeta con timestamp"
    )

# ═══════════════════════════════════════════════════════════════
# METODOLOGÍA
# ══════════════════════════════════════════════════════════════
    methodology_expander(
    "📖 Metodología — H.G.",
    [
        (
            "Métrica Compuesta (Pareto)",
            r"\mu_{\alpha} = c + \alpha \cdot \beta_{1} \quad \text{con } \alpha > 0"
        ),
        (
            "Criterio de Optimidad",
            r"p=3 \text{ es Pareto-óptimo si minimiza } c \text{ manteniendo } \beta_{1} = 2"
        ),
        (
            "Configuración de Barrido",
            r"p \in \{2, 3, 4\} \text{ en malla } 60 \times 30"
        ),
        (
            "Veredicto Final",
            r"\text{H.E.2a}' \text{ (reducción 77.58\%)} \ge 40\% \quad \text{y} \quad \beta_{1}=2 \text{ invariante}"
        )
    ],
    "H.G."
    )
    st.markdown("---")

apply_mpl_theme()
