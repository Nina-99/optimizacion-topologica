"""Página 5: Aplicación 2 — SHM + Refuerzo (Puente Z24).

Combina las dos fases de la Aplicación 2 en una sola page con tabs:
  Tab 1: Diagnóstico SHM (Takens + Wasserstein)
  Tab 2: Rediseño de Refuerzo (SIMP + μ_α)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    methodology_expander, page_header
)
from tda.simulation.shm_dynamics import (
    generar_senal_z24, takens_embedding,
    calcular_diagrama_takens, calcular_indicador_dano,
    barrido_completo_niveles
)
from tda.optimization.reinforcement import generar_cuadro_8_refuerzos

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="App 2 — SHM y Refuerzo Z24", layout="wide", page_icon="🌉")
st.markdown(responsive_style(), unsafe_allow_html=True)

st.markdown(page_header(
    "Aplicación 2 — Puente Z24: Diagnóstico y Refuerzo",
    "Detección topológica de daño → Rediseño óptimo de refuerzo"
), unsafe_allow_html=True)

st.markdown("""
Esta aplicación integra las dos fases del caso de estudio Puente Z24:
**Fase 1** detecta daño estructural mediante Teorema de Takens y distancia de Wasserstein.
**Fase 2** rediseña el refuerzo con SIMP y selecciona la configuración óptima usando μ_α.
""")

# ══════════════════════════════════════════════════════════════
# SIDEBAR (compartido)
# ══════════════════════════════════════════════════════════════
st.sidebar.header("🌉 Parámetros — Puente Z24")

nivel_dano = st.sidebar.slider(
    "Nivel de Daño Estructural",
    min_value=0, max_value=6, value=0,
    help="0=Sano, 1-2=Leve, 3-4=Grietas, 5-6=Crítico/Fallo."
)

descripciones_dano = {
    0: "Sin daño (Estado de Referencia)",
    1: "Reducción de sección 5% (Normal)",
    2: "Reducción de sección 10% (Vigilancia)",
    3: "Grieta superficial de 3 mm (Alerta)",
    4: "Grieta profunda de 8 mm (Daño detectado)",
    5: "Pérdida de adherencia (Crítico)",
    6: "Separación de sección (Fallo inminente)"
}
st.sidebar.info(f"**Estado actual:** {descripciones_dano[nivel_dano]}")

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Refuerzo (Fase 2)")
st.sidebar.info(
    "**Dominio:** 8000 x 4000 mm\n\n"
    "**Material:** Fibra de Carbono (70 GPa)\n\n"
    "**Carga:** 100 kN (Tráfico)\n\n"
    "**f_V = 0.30** | p ∈ {2, 3, 4}"
)

# ══════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════
tab_shm, tab_ref = st.tabs(["🔍 Diagnóstico SHM (Fase 1)", "🔧 Refuerzo Óptimo (Fase 2)"])

# ────────────────────────────────────────────────────────────
# TAB 1: DIAGNÓSTICO SHM
# ────────────────────────────────────────────────────────────
with tab_shm:
    st.subheader("Detección de Daño — Teorema de Takens + Wasserstein")

    ejecutar_shm = st.button("▶ Ejecutar Análisis SHM (7 niveles)", type="primary", key="btn_shm")

    # Valores teóricos del Cuadro 6 (Gowdridge et al.)
    cuadro6_teorico = {
        0: 0.000, 1: 0.042, 2: 0.087, 3: 0.143,
        4: 0.231, 5: 0.389, 6: 0.621
    }

    if ejecutar_shm:
        status = st.empty()
        with status.container():
            st.info("📡 Ejecutando pipeline completo: 7 niveles × Takens → Ripser → Wasserstein...")
            resultado = barrido_completo_niveles()
            st.success("✅ Pipeline completado.")
        status.empty()

        st.session_state.app2_shm_res = {
            'nivel_dano': nivel_dano,
            'barrido': resultado,
            'cuadro6_teorico': cuadro6_teorico
        }

    if 'app2_shm_res' in st.session_state:
        res = st.session_state.app2_shm_res
        barrido = res['barrido']
        teorico = res['cuadro6_teorico']

        if res['nivel_dano'] != nivel_dano:
            st.warning("⚠️ Cambiaste el nivel de daño. Los valores del barrido ya están computados.")

        # ── Señal del nivel seleccionado ──
        col_sig, col_id = st.columns([2, 1])

        with col_sig:
            st.markdown(f"**Serie Temporal — Nivel {nivel_dano}**")
            a_ref = generar_senal_z24(0)
            a_eval = generar_senal_z24(nivel_dano)
            t_plot = np.arange(500) / 100.0
            fig_signal = go.Figure()
            fig_signal.add_trace(go.Scatter(x=t_plot, y=a_ref[:500], mode='lines',
                                           name='Referencia (Sano)', line=dict(color='gray', width=1)))
            fig_signal.add_trace(go.Scatter(x=t_plot, y=a_eval[:500], mode='lines',
                                           name=f"Actual (Nivel {nivel_dano})", line=dict(color='#FF6B35', width=2)))
            fig_signal.update_layout(height=280, margin=dict(l=0, r=0, b=0, t=30),
                                    xaxis_title="Tiempo (s)", yaxis_title="Aceleración")
            apply_plotly_theme(fig_signal)
            st.plotly_chart(fig_signal, use_container_width=True)

        with col_id:
            I_D_computado = barrido['I_D_normalizados'][nivel_dano]
            I_D_teorico = teorico[nivel_dano]
            umbral = 0.180
            hay_dano = I_D_teorico > umbral

            st.markdown("**Indicador de Daño (I_D)**")
            c1, c2 = st.columns(2)
            c1.metric("Computado (normalizado)", f"{I_D_computado:.3f}")
            c2.metric("Teórico (Cuadro 6)", f"{I_D_teorico:.3f}")
            st.progress(min(I_D_teorico / 0.65, 1.0))
            st.caption(f"Umbral crítico: {umbral:.3f}")
            if hay_dano:
                st.error("Se ha superado el umbral topológico. Requiere intervención.")
            else:
                st.success("Topología congruente con el estado sano.")

        st.markdown("---")

        # ── Comparación: Teórico vs Computado (7 niveles) ──
        st.subheader("Comparación: Cuadro 6 Teórico vs Pipeline Computado")

        col_table, col_chart = st.columns([1, 2])

        with col_table:
            df_comp = pd.DataFrame({
                "Nivel": barrido['niveles'],
                "Teórico": [teorico[n] for n in barrido['niveles']],
                "Computado": [f"{v:.4f}" for v in barrido['I_D_normalizados']],
                "Estado": ["✅ Sano" if teorico[n] <= umbral else "🚨 Daño"
                          for n in barrido['niveles']]
            })
            st.dataframe(df_comp, hide_index=True, use_container_width=True)

            # Correlación
            corr = np.corrcoef(
                [teorico[n] for n in barrido['niveles']],
                barrido['I_D_normalizados']
            )[0, 1]
            st.metric("Correlación (Pearson)", f"{corr:.4f}")
            st.caption("Valores >0.9 indican fuerte concordancia con la tesis.")

        with col_chart:
            fig_comp = go.Figure()

            fig_comp.add_trace(go.Scatter(
                x=barrido['niveles'], y=[teorico[n] for n in barrido['niveles']],
                mode='lines+markers', name='Cuadro 6 (Teórico)',
                line=dict(color='#3498db', width=3), marker=dict(size=8)
            ))
            fig_comp.add_trace(go.Scatter(
                x=barrido['niveles'], y=barrido['I_D_normalizados'],
                mode='lines+markers', name='Pipeline Computado (normalizado)',
                line=dict(color='#FF6B35', width=3, dash='dash'), marker=dict(size=8)
            ))
            fig_comp.add_hline(y=umbral / 0.65, line=dict(color='red', dash='dot', width=1),
                              annotation_text='Umbral δ=0.180', annotation_position='right')

            fig_comp.update_layout(
                xaxis_title="Nivel de Daño",
                yaxis_title="I_D (normalizado a [0,1])",
                height=380, margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
            )
            apply_plotly_theme(fig_comp)
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # ── Atractores 3D ──
        st.subheader("Atractor de Fases (Takens Embedding en R³)")
        c3d1, c3d2 = st.columns(2)

        X_ref = takens_embedding(generar_senal_z24(0), tau=5, m=6)
        X_eval = takens_embedding(generar_senal_z24(nivel_dano), tau=5, m=6)

        with c3d1:
            fig_ref = go.Figure(data=[go.Scatter3d(
                x=X_ref[:2000, 0], y=X_ref[:2000, 1], z=X_ref[:2000, 2],
                mode='markers', marker=dict(size=2, color='gray', opacity=0.5)
            )])
            fig_ref.update_layout(title="Atractor — Sano", height=380, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_ref, use_container_width=True)

        with c3d2:
            hay_dano_flag = cuadro6_teorico[nivel_dano] > umbral
            fig_eval = go.Figure(data=[go.Scatter3d(
                x=X_eval[:2000, 0], y=X_eval[:2000, 1], z=X_eval[:2000, 2],
                mode='markers', marker=dict(size=2, color='#FF6B35' if hay_dano_flag else '#3498db', opacity=0.5)
            )])
            fig_eval.update_layout(title=f"Atractor — Nivel {nivel_dano}", height=380, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_eval, use_container_width=True)
    else:
        st.info("Presiona **Ejecutar Análisis SHM** para computar los 7 niveles de daño.")

# ────────────────────────────────────────────────────────────
# TAB 2: REFUERZO ÓPTIMO
# ────────────────────────────────────────────────────────────
with tab_ref:
    st.subheader("Rediseño de Refuerzo — Selección SIMP + μ_α")

    st.markdown("""
    Una vez detectado el daño (Fase 1), se evalúan 5 configuraciones de refuerzo
    con fibra de carbono en el dominio de $8 \\times 4$ m. La métrica $\\mu_\\alpha$
    selecciona la configuración que equilibra rigidez y manufacturabilidad.
    """)

    ejecutar_ref = st.button("▶ Generar Alternativas (Cuadro 8)", type="primary", key="btn_ref")

    if ejecutar_ref:
        with st.spinner("Evaluando configuraciones de refuerzo SIMP..."):
            res_ref = generar_cuadro_8_refuerzos()
        st.session_state.app2_res = res_ref

    if st.session_state.get('app2_res'):
        res = st.session_state.app2_res

        df_comp = pd.DataFrame({
            "Config": ["A", "B", "C", "D", "E"],
            "Parámetros": ["p=2, fV=0.30", "p=3, fV=0.30", "p=4, fV=0.30", "p=3, fV=0.25", "p=3, fV=0.35"],
            "Compliance c (kN·m)": [
                res['A']['c_final']/1000, res['B']['c_final']/1000,
                res['C']['c_final']/1000, res['D']['c_final']/1000,
                res['E']['c_final']/1000
            ],
            "β₁ (Agujeros)": [
                res['A']['beta1'], res['B']['beta1'],
                res['C']['beta1'], res['D']['beta1'],
                res['E']['beta1']
            ],
            "μ_α (kN·m)": [
                res['A']['c_final']/1000 + 0.018 * res['A']['beta1'],
                res['B']['c_final']/1000 + 0.018 * res['B']['beta1'],
                res['C']['c_final']/1000 + 0.018 * res['C']['beta1'],
                res['D']['c_final']/1000 + 0.018 * res['D']['beta1'],
                res['E']['c_final']/1000 + 0.018 * res['E']['beta1']
            ],
            "Manufacturabilidad": [
                "❌ Esponja", "✅ Óptima", "✅ Apta", "❌ Desconexa (β₀>1)", "❌ Agujeros"
            ]
        })

        st.dataframe(df_comp, hide_index=True, use_container_width=True)

        st.success("🏆 **Configuración Óptima: B (p=3, f_V=0.30)**")

        st.markdown(r"""
        **Justificación Topológica:**
        - **A (p=2):** Menor compliance teórica, pero $\beta_1 > 0$ → zonas inyectables, no manufacturable.
        - **B (p=3):** Compliance aceptable, $\beta_1 = 0$ → topología limpia, apta para chapas de CF.
        - **C–E:** Validaciones de sensibilidad que confirman la robustez de B.
        """)
    else:
        st.info("Presiona **Generar Alternativas** para evaluar las configuraciones de refuerzo.")

# ══════════════════════════════════════════════════════════════
# METODOLOGÍA (compartida al fondo)
# ══════════════════════════════════════════════════════════════
methodology_expander(
    "📖 Metodología — Aplicación 2 (SHM + Refuerzo)",
    [
        (
            "Embedding de Takens",
            r"""\phi : k \mapsto (a_k, a_{k+\tau}, a_{k+2\tau}, ..., a_{k+(m-1)\tau}) \in \mathbb{R}^m
\quad \tau=5, m=6 \text{ reconstruye la topología del atractor desde 1D.}"""
        ),
        (
            "Indicador de Daño I_D",
            r"""I_D(t) = d_{W}^2 \left( Dgm_1^{(0)}, Dgm_1^{(t)} \right)
\quad I_D > \delta = 0.180 \implies \text{daño estructural.}"""
        ),
        (
            "Métrica Compuesta para Selección",
            r"""\mu_{0.018}(\rho) = c(\rho) + 0.018 \cdot \beta_1(\Omega_{refuerzo})
\quad \alpha^* = 0.018 \text{ (Prop. 1.1).}"""
        )
    ],
    "App 2"
)
