"""Página principal — Plataforma de Optimización SIMP y Análisis Topológico.

Landing page con navegación a los 4 módulos:
  1. H.E.1 — Robustez TDA vs Euclidianos
  2. H.E.2 — Optimización SIMP + Métrica Compuesta μ_α
  3. H.G. — Comparación Integrada TDA-SIMP
  4. Ejemplo — Viga 1D
"""
# ══════════════════════════════════════════════════════════════
# FIX PyInstaller: forzar backends matplotlib ANTES de cualquier otro import.
# ══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_pdf
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

from tda.app.theme import landing_card, footer_style, update_hero_orange, responsive_style, ORANGE

# ==========================================
# CONFIGURACIÓN DE PÁGINA (DEBE SER EL PRIMER COMANDO STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="Plataforma TDA-SIMP",
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

# ── Hero section premium con logos dentro de la tarjeta ──
import os as _os
import base64 as _b64

_logo_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "img")

def _img_to_base64(path):
    if _os.path.exists(path):
        with open(path, "rb") as f:
            return _b64.b64encode(f.read()).decode()
    return None

_uagrm_b64 = _img_to_base64(_os.path.join(_logo_dir, "logoUAGRM.png"))
_upi_b64 = _img_to_base64(_os.path.join(_logo_dir, "Logo-UPI.png"))

_uagrm_tag = f'<img src="data:image/png;base64,{_uagrm_b64}" style="height:170px;" />' if _uagrm_b64 else ""
_upi_tag = f'<img src="data:image/png;base64,{_upi_b64}" style="height:170px;" />' if _upi_b64 else ""

_bg_animation = "heroGradient 8s ease infinite"

st.markdown(update_hero_orange(f"""
<style>
@keyframes heroGradient {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
</style>
<div style="
    background:linear-gradient(135deg, #2c1a00 0%, #3d2200 40%, #1a0f00 70%, #2c1a00 100%);
    background-size:200% 200%;
    animation:{_bg_animation};
    padding:2rem 2.5rem;
    border-radius:20px;
    color:white;
    margin-bottom:2rem;
    box-shadow:0 12px 40px rgba(0,0,0,0.25);
    display:flex;
    align-items:center;
    justify-content:space-between;
    border-left:5px solid {ORANGE};
">
    <div style="flex:0 0 auto;margin-right:1.5rem;">
        {_uagrm_tag}
    </div>
    <div style="flex:1;text-align:center;">
        <h1 style="margin:0;font-weight:800;font-size:2.2rem;letter-spacing:-0.02em;color:{ORANGE};">Plataforma TDA-SIMP</h1>
        <p style="margin:0.6rem 0 0 0;opacity:0.9;font-size:1rem;max-width:600px;margin-left:auto;margin-right:auto;line-height:1.5;">
            Optimización Topológica, Homología Persistente y Métricas Compuestas
            para el diseño de estructuras eficientes y manufacturables.
        </p>
        <p style="margin:0.3rem 0 0 0;opacity:0.7;font-size:0.85rem;">
            Jorge Larry Copa Cruz · Maestría en Matemática · UAGRM · 2026
        </p>
    </div>
    <div style="flex:0 0 auto;margin-left:1.5rem;">
        {_upi_tag}
    </div>
</div>
"""), unsafe_allow_html=True)

# ── Cards de navegación ──
st.markdown("### Módulos de Análisis")

# CSS para integrar el botón de navegación con la tarjeta
st.markdown("""
<style>
div[data-testid="stPageLink"] a {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease !important;
}
div[data-testid="stPageLink"] a:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Fila de Tarjetas de Acceso ──
cols = st.columns(4)

with cols[0]:
    st.markdown(landing_card(
        icon="📐",
        title="H.E.1 — Robustez",
        description="Validación de invariantes topológicos bajo ruido gaussiano y comparación con descriptores euclidianos.",
        variant="orange"
    ), unsafe_allow_html=True)
    st.page_link("pages/1_H.E.1_Robustez_TDA_vs_Euclidianos.py", label="📐 Abrir H.E.1", width='stretch')

with cols[1]:
    st.markdown(landing_card(
        icon="🧮",
        title="H.E.2 — Optimización",
        description="Implementación de SIMP con métrica compuesta μ_α para optimización de compliance y topología.",
        variant="orange"
    ), unsafe_allow_html=True)
    st.page_link("pages/2_H.E.2_Optimizacion_SIMP_Metrica_Compuesta.py", label="🧮 Abrir H.E.2", width='stretch')

with cols[2]:
    st.markdown(landing_card(
        icon="🔬",
        title="H.G. — Comparación",
        description="Síntesis integral de TDA+SIMP: Validación de la Hipótesis General y Pareto-optimalidad.",
        variant="orange"
    ), unsafe_allow_html=True)
    st.page_link("pages/3_H.G._Comparacion_Integrada_TDA-SIMP.py", label="🔬 Abrir H.G.", width='stretch')

with cols[3]:
    st.markdown(landing_card(
        icon="🧪",
        title="Ejemplo Viga 1D vs 2D",
        description="Laboratorio interactivo: Comparación de optimización analítica 1D vs topológica SIMP 2D.",
        variant="orange"
    ), unsafe_allow_html=True)
    st.page_link("pages/4_Ejemplo_Viga_1D_vs_2D.py", label="🧪 Abrir Ejemplo", width='stretch')


# ── Footer ──
st.markdown("---")
st.markdown(f"""
<div style="{footer_style()}">
    <p style="margin:0;color:{ORANGE};font-weight:700;">
        Jorge Larry Copa Cruz · Maestría en Matemática · UAGRM · 2026
    </p>
    <p style="margin:0.3rem 0 0 0;">
        Tecnologías: Streamlit · Python · NumPy · Matplotlib · Plotly · Ripser · Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)
