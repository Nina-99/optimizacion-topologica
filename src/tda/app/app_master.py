"""Página principal — Plataforma de Optimización SIMP y Análisis Topológico.

Landing page con navegación a las 3 aplicaciones principales:
  1. TDA vs K-Medias (H.E.1) — Robustez topológica vs clasificación euclidiana
  2. Optimización Topológica de Vigas — SIMP 1D para hormigón armado
  3. Optimización SIMP + TDA (H.E.2) — Métrica compuesta μ_α
"""

import streamlit as st

# ==========================================
# CONFIGURACIÓN DE PÁGINA (DEBE SER EL PRIMER COMANDO STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="TDA-SIMP Master Suite",
    layout="wide",
    page_icon="📐"
)

# ==========================================
# LANDING PAGE
# ==========================================
st.title("Plataforma de Optimización SIMP y Análisis Topológico (TDA)")
st.markdown("---")

# ── Hero section ──
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
">
    <h1 style="margin:0; font-weight:700; font-size:2.2rem;">📐 Bienvenido a la Suite TDA-SIMP</h1>
    <p style="margin:0.8rem 0 0 0; opacity:0.9; font-size:1.1rem; max-width:700px; margin-left:auto; margin-right:auto;">
        Optimización Topológica, Homología Persistente y Métricas Compuestas 
        para el diseño de estructuras eficientes y manufacturables.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Cards de navegación ──
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background: #f0f4f8;
        padding: 1.8rem 1.2rem;
        border-radius: 14px;
        text-align: center;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 2px solid #d5dbdb;
        transition: all 0.2s;
    ">
        <div style="font-size:3rem; margin-bottom:0.5rem;">📊</div>
        <h3 style="margin:0.3rem 0; color:#2c3e50;">TDA vs K-Medias</h3>
        <p style="margin:0.3rem 0 0 0; font-size:0.9rem; color:#7f8c8d;">
            H.E.1 — Robustez de invariantes topológicos<br>frente a ruido gaussiano
        </p>
        <p style="margin:1rem 0 0 0; font-size:0.8rem; color:#3498db; font-weight:600;">
            ← Seleccionar en la barra lateral
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: #fef9e7;
        padding: 1.8rem 1.2rem;
        border-radius: 14px;
        text-align: center;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 2px solid #f5d76e;
        transition: all 0.2s;
    ">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🏗️</div>
        <h3 style="margin:0.3rem 0; color:#2c3e50;">Optimización de Vigas</h3>
        <p style="margin:0.3rem 0 0 0; font-size:0.9rem; color:#7f8c8d;">
            SIMP 1D — Minimización de volumen<br>con restricción de rigidez
        </p>
        <p style="margin:1rem 0 0 0; font-size:0.8rem; color:#e67e22; font-weight:600;">
            ← Seleccionar en la barra lateral
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background: #eafaf1;
        padding: 1.8rem 1.2rem;
        border-radius: 14px;
        text-align: center;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 2px solid #82e0aa;
        transition: all 0.2s;
    ">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🧮</div>
        <h3 style="margin:0.3rem 0; color:#2c3e50;">SIMP + TDA</h3>
        <p style="margin:0.3rem 0 0 0; font-size:0.9rem; color:#7f8c8d;">
            H.E.2 — Métrica compuesta μ_α = c + α·β₁<br>con análisis topológico
        </p>
        <p style="margin:1rem 0 0 0; font-size:0.8rem; color:#27ae60; font-weight:600;">
            ← Seleccionar en la barra lateral
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; font-size:0.85rem; padding:1rem 0;">
    <p style="margin:0;">
        Jorge Larry Copa Cruz · Maestría en Matemática · Universidad Autónoma Gabriel René Moreno · 2026
    </p>
    <p style="margin:0.3rem 0 0 0;">
        Tecnologías: Streamlit · Python · NumPy · Matplotlib · Plotly · Ripser · Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)
