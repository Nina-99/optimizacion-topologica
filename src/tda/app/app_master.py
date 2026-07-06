"""Página principal — Plataforma de Optimización SIMP y Análisis Topológico.

Landing page con navegación a las 3 aplicaciones principales:
  1. TDA vs K-Medias (H.E.1) — Robustez topológica vs clasificación euclidiana
  2. Optimización Topológica de Vigas — SIMP 1D para hormigón armado
  3. Optimización SIMP + TDA (H.E.2) — Métrica compuesta μ_α
"""
# ══════════════════════════════════════════════════════════════
# FIX PyInstaller: forzar backends matplotlib ANTES de cualquier otro import.
# Sin esto, en el .exe compilado savefig() y PdfPages fallan porque
# PyInstaller no empaqueta los backends cargados dinámicamente.
# ══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")  # sin GUI, necesario en .exe (también funciona en desarrollo)
import matplotlib.backends.backend_pdf  # fuerza empaquetado por PyInstaller
import matplotlib.backends.backend_agg
import os
import sys as _sys
if getattr(_sys, "frozen", False):
    _mei = _sys._MEIPASS
    os.environ.setdefault("MATPLOTLIBDATA",
                          os.path.join(_mei, "matplotlib", "mpl-data"))
del os, _sys

# ══════════════════════════════════════════════════════════════

import streamlit as st

from tda.app.theme import landing_card, footer_style, hero_section, responsive_style

# ==========================================
# CONFIGURACIÓN DE PÁGINA (DEBE SER EL PRIMER COMANDO STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="TDA-SIMP Master Suite",
    layout="wide",
    page_icon="📐"
)

st.markdown(responsive_style(), unsafe_allow_html=True)

# ── Sidebar: configuración de exportación (.exe) ──
from tda.app.download_utils import export_settings_ui
export_settings_ui()

# ==========================================
# LANDING PAGE
# ==========================================
st.title("Plataforma de Optimización SIMP y Análisis Topológico (TDA)")
st.markdown("---")

# ── Hero section ──
st.markdown(hero_section(
    "📐 Bienvenido a la Suite TDA-SIMP",
    "Optimización Topológica, Homología Persistente y Métricas Compuestas "
    "para el diseño de estructuras eficientes y manufacturables."
), unsafe_allow_html=True)

# ── Cards de navegación ──
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(landing_card(
        icon="📊",
        title="TDA vs K-Medias",
        description="H.E.1 — Robustez de invariantes topológicos<br>frente a ruido gaussiano",
        variant="info"
    ), unsafe_allow_html=True)
    st.page_link("pages/1_TDA_vs_KMedias.py",
                 label="📊 Abrir TDA vs K-Medias",
                 use_container_width=True)

with col2:
    st.markdown(landing_card(
        icon="🏗️",
        title="Optimización de Vigas",
        description="SIMP 1D — Minimización de volumen<br>con restricción de rigidez",
        variant="warning"
    ), unsafe_allow_html=True)
    st.page_link("pages/2_Optimizacion_Topologica.py",
                 label="🏗️ Abrir Optimización de Vigas",
                 use_container_width=True)

with col3:
    st.markdown(landing_card(
        icon="🧮",
        title="SIMP + TDA",
        description="H.E.2 — Métrica compuesta μ_α = c + α·β₁<br>con análisis topológico",
        variant="success"
    ), unsafe_allow_html=True)
    st.page_link("pages/3_Optimizacion_SIMP.py",
                 label="🧮 Abrir SIMP + TDA",
                 use_container_width=True)

# ── Footer ──
st.markdown("---")
st.markdown(f"""
<div style="{footer_style()}">
    <p style="margin:0;">
        Jorge Larry Copa Cruz · Maestría en Matemática · Universidad Autónoma Gabriel René Moreno · 2026
    </p>
    <p style="margin:0.3rem 0 0 0;">
        Tecnologías: Streamlit · Python · NumPy · Matplotlib · Plotly · Ripser · Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)
