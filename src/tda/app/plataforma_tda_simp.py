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
import sys as _sys
import base64 as _b64

# Handle PyInstaller bundle path
def _get_base_dir():
    if getattr(_sys, "frozen", False):
        return _sys._MEIPASS
    return _os.path.dirname(__file__)

_logo_dir = _os.path.join(_get_base_dir(), "..", "..", "img")
# Fallback: try from project root (for PyInstaller --add-data src\tda;tda)
if not _os.path.isdir(_logo_dir):
    _logo_dir = _os.path.join(_get_base_dir(), "img")
# Fallback: try from current working directory
if not _os.path.isdir(_logo_dir):
    _logo_dir = _os.path.join(".", "img")

def _img_to_base64(path):
    if _os.path.exists(path):
        with open(path, "rb") as f:
            return _b64.b64encode(f.read()).decode()
    return None

_uagrm_b64 = _img_to_base64(_os.path.join(_logo_dir, "logoUAGRM.png"))
_upi_b64 = _img_to_base64(_os.path.join(_logo_dir, "Logo-UPI.png"))

_uagrm_tag = f'<img src="data:image/png;base64,{_uagrm_b64}" style="height:120px;max-width:100%;object-fit:contain;" />' if _uagrm_b64 else ""
_upi_tag = f'<img src="data:image/png;base64,{_upi_b64}" style="height:120px;max-width:100%;object-fit:contain;" />' if _upi_b64 else ""

_bg_animation = "heroGradient 8s ease infinite"

st.markdown(update_hero_orange(f"""
<style>
@keyframes heroGradient {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
.hero-container {{
    background:linear-gradient(135deg, #2c1a00 0%, #3d2200 40%, #1a0f00 70%, #2c1a00 100%);
    background-size:200% 200%;
    animation:{_bg_animation};
    padding:2rem 2.5rem;
    border-radius:20px;
    color:white;
    margin-bottom:2rem;
    box-shadow:0 12px 40px rgba(0,0,0,0.25);
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    justify-content:center;
    gap:1rem;
    border-left:5px solid {ORANGE};
    overflow:hidden;
}}
.hero-logo img {{
    height:120px;
    max-width:100%;
    object-fit:contain;
}}
.hero-title {{
    flex:1 1 300px;
    text-align:center;
    min-width:280px;
}}
.hero-title h1 {{
    margin:0;
    font-weight:800;
    font-size:2.2rem;
    letter-spacing:-0.02em;
    color:{ORANGE};
}}
.hero-title p {{
    margin:0.6rem 0 0 0;
    opacity:0.9;
    font-size:1rem;
    max-width:600px;
    margin-left:auto;
    margin-right:auto;
    line-height:1.5;
}}
.hero-title .subtitle {{
    margin:0.3rem 0 0 0;
    opacity:0.7;
    font-size:0.85rem;
}}
@media (max-width: 768px) {{
    .hero-container {{
        padding:1.2rem 1rem;
        flex-direction:column;
        text-align:center;
    }}
    .hero-logo img {{
        height:80px;
    }}
    .hero-title h1 {{
        font-size:1.5rem;
    }}
    .hero-title p {{
        font-size:0.9rem;
    }}
}}
</style>
<div class="hero-container">
    <div class="hero-logo">
        {_uagrm_tag}
    </div>
    <div class="hero-title">
        <h1>Plataforma TDA-SIMP</h1>
        <p>
            Optimizacion Topologica, Homologia Persistente y Metricas Compuestas
            para el diseno de estructuras eficientes y manufacturables.
        </p>
        <p class="subtitle">
            Jorge Larry Copa Cruz · Maestria en Matematica · UAGRM · 2026
        </p>
    </div>
    <div class="hero-logo">
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
