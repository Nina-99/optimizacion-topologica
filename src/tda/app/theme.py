"""Utilidades de theming para la app Streamlit TDA-SIMP.

Soporte de modo oscuro/claro con componentes adaptables.
Incluye: tipografía custom, cards premium, progress por fases, badges.
"""

import streamlit as st
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# PALETA DE COLORES — TEMA OSCURO NARANJA
# ═══════════════════════════════════════════════════════════════

ORANGE = "#FF6B35"
ORANGE_HOVER = "#ff8555"
ORANGE_DARK = "#c44b1a"
DARK_BG = "#121212"
DARK_CARD = "#1e1e1e"
DARK_BORDER = "#333333"
WHITE = "#ffffff"
WHITE_DIM = "#b0b0b0"
GREEN_OK = "#27ae60"
RED_BAD = "#e74c3c"
BLUE_INFO = "#2980b9"
GOLD_WARN = "#f39c12"


# ── Deteccion de tema ───────────────────────────────────────

def is_dark():
    """Detecta si Streamlit está en modo oscuro."""
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


# ── Tipografía custom ───────────────────────────────────────────────

def custom_fonts():
    """Inyecta Google Fonts (Inter + JetBrains Mono) y estilos base.

    Gracefully falls back to system fonts when offline (e.g., in .exe builds).
    Llamar una sola vez al inicio de cada pagina o en responsive_style().
    """
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Tipografia base — fallback a fuentes del sistema si offline */
body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Numeros de metricas: JetBrains Mono */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace !important;
    font-weight: 600 !important;
}

/* Títulos con peso uniforme */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

/* Sidebar más compacto */
[data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
}

/* Botones primarios con gradiente sutil */
.stButton > button[kind="primary"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

/* Progress bar personalizada */
.stProgress > div > div {
    border-radius: 8px !important;
    height: 10px !important;
}
</style>
"""


# ── Matplotlib ──────────────────────────────────────────────────────

def apply_mpl_theme():
    """Configura matplotlib para renderizado inline según el tema."""
    if is_dark():
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
def apply_plotly_theme(fig, leyenda="derecha"):
    """Aplica el template de Plotly según el tema y ubica la leyenda.

    ``leyenda``:
      - ``"derecha"`` (default): leyenda vertical fuera del área de trazado,
        a la derecha, para no tapar el gráfico.
      - ``"secundaria"``: idem pero más desplazada, para gráficos con eje Y a
        ambos lados (no choca con el eje secundario).
      - ``None``: no toca leyenda ni márgenes (heatmaps y gráficos sin leyenda).
    """
    if is_dark():
        fig.update_layout(template="plotly_dark")
    else:
        fig.update_layout(template="plotly_white")
    if leyenda == "derecha":
        fig.update_layout(
            legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top"),
            margin=dict(r=175),
        )
    elif leyenda == "secundaria":
        fig.update_layout(
            legend=dict(orientation="v", x=1.16, xanchor="left", y=1, yanchor="top"),
            margin=dict(r=260),
        )
    return fig


def get_plot_colors():
    """Retorna colores adaptativos para plots según el tema actual.

    Returns
    -------
    dict
        Claves: bg, paper_bg, grid, font, text
    """
    if is_dark():
        return {
            "bg": "#1a1a2e",
            "paper_bg": "#0f0f23",
            "grid": "#2c3e50",
            "font": "#ecf0f1",
            "text": "#bdc3c7",
        }
    return {
        "bg": "#fafafa",
        "paper_bg": "#ffffff",
        "grid": "#ecf0f1",
        "font": "#2c3e50",
        "text": "#7f8c8d",
    }


# ── Paletas de color ────────────────────────────────────────────────

_LANDING_PALETTES = {
    "info": {
        "light": {
            "bg": "#f0f4f8", "border": "#d5dbdb", "icon_bg": "#e8f4fd",
            "title": "#2c3e50", "text": "#7f8c8d", "link": "#3498db",
            "gradient": "linear-gradient(135deg, #f0f4f8 0%, #e8f4fd 100%)",
        },
        "dark": {
            "bg": "#1a2634", "border": "#3498db", "icon_bg": "#1a3a5c",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#5dade2",
            "gradient": "linear-gradient(135deg, #1a2634 0%, #1a3a5c 100%)",
        },
    },
    "warning": {
        "light": {
            "bg": "#fef9e7", "border": "#f5d76e", "icon_bg": "#fef3cd",
            "title": "#2c3e50", "text": "#7f8c8d", "link": "#e67e22",
            "gradient": "linear-gradient(135deg, #fef9e7 0%, #fef3cd 100%)",
        },
        "dark": {
            "bg": "#2c1f0a", "border": "#e67e22", "icon_bg": "#3d2b10",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#f39c12",
            "gradient": "linear-gradient(135deg, #2c1f0a 0%, #3d2b10 100%)",
        },
    },
    "success": {
        "light": {
            "bg": "#eafaf1", "border": "#82e0aa", "icon_bg": "#d5f5e3",
            "title": "#2c3e50", "text": "#7f8c8d", "link": "#27ae60",
            "gradient": "linear-gradient(135deg, #eafaf1 0%, #d5f5e3 100%)",
        },
        "dark": {
            "bg": "#0a2a1a", "border": "#27ae60", "icon_bg": "#0d3d22",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#2ecc71",
            "gradient": "linear-gradient(135deg, #0a2a1a 0%, #0d3d22 100%)",
        },
    },
    "accent": {
        "light": {
            "bg": "#f5eef8", "border": "#d2b4de", "icon_bg": "#ebdef0",
            "title": "#2c3e50", "text": "#7f8c8d", "link": "#8e44ad",
            "gradient": "linear-gradient(135deg, #f5eef8 0%, #ebdef0 100%)",
        },
        "dark": {
            "bg": "#1a0f2e", "border": "#8e44ad", "icon_bg": "#2a1a45",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#a569bd",
            "gradient": "linear-gradient(135deg, #1a0f2e 0%, #2a1a45 100%)",
        },
    },
    "orange": {
        "light": {
            "bg": "#fff5eb", "border": "#FF6B35", "icon_bg": "#ffe8d6",
            "title": "#2c1a00", "text": "#7f5520", "link": "#FF6B35",
            "gradient": "linear-gradient(135deg, #fff5eb 0%, #ffe8d6 100%)",
        },
        "dark": {
            "bg": "#2c1a00", "border": "#FF6B35", "icon_bg": "#3d2200",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#FF6B35",
            "gradient": "linear-gradient(135deg, #2c1a00 0%, #3d2200 100%)",
        },
    },
}

_METRIC_PALETTES = {
    "beta0": {
        "light": {"bg": "#e8f4f8", "h3": "#2980b9", "small": "#7f8c8d", "icon_color": "#3498db"},
        "dark": {"bg": "#1a2634", "h3": "#5dade2", "small": "#bdc3c7", "icon_color": "#5dade2"},
    },
    "beta1": {
        "light": {"bg": "#fef9e7", "h3": "#d35400", "small": "#7f8c8d", "icon_color": "#e67e22"},
        "dark": {"bg": "#2c1f0a", "h3": "#e67e22", "small": "#bdc3c7", "icon_color": "#f39c12"},
    },
    "mu": {
        "light": {"bg": "#eafaf1", "h3": "#27ae60", "small": "#7f8c8d", "icon_color": "#2ecc71"},
        "dark": {"bg": "#0a2a1a", "h3": "#2ecc71", "small": "#bdc3c7", "icon_color": "#58d68d"},
    },
    "manufacturing": {
        "light": {"bg": "#f4f6f7", "h3": "#7f8c8d", "small": "#7f8c8d", "icon_color": "#95a5a6"},
        "dark": {"bg": "#2c2c2c", "h3": "#95a5a6", "small": "#bdc3c7", "icon_color": "#bdc3c7"},
    },
    "compliance": {
        "light": {"bg": "#ebf5fb", "h3": "#2471a3", "small": "#7f8c8d", "icon_color": "#2e86c1"},
        "dark": {"bg": "#15263b", "h3": "#5dade2", "small": "#bdc3c7", "icon_color": "#85c1e9"},
    },
    "orange": {
        "light": {"bg": "#fff5eb", "h3": "#FF6B35", "small": "#7f5520", "icon_color": "#FF6B35"},
        "dark": {"bg": "#2c1a00", "h3": "#FF6B35", "small": "#bdc3c7", "icon_color": "#FF6B35"},
    },
}

# Iconos SVG inline por variante de métrica
_METRIC_ICONS = {
    "beta0": (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="2"/>'
        '<circle cx="12" cy="12" r="4" fill="{color}"/></svg>'
    ),
    "beta1": (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="2"/>'
        '<circle cx="12" cy="12" r="5" stroke="{color}" stroke-width="2" fill="none"/>'
        '<line x1="12" y1="2" x2="12" y2="7" stroke="{color}" stroke-width="2"/></svg>'
    ),
    "mu": (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="3" y="3" width="18" height="18" rx="3" stroke="{color}" stroke-width="2"/>'
        '<line x1="7" y1="12" x2="17" y2="12" stroke="{color}" stroke-width="2"/>'
        '<line x1="12" y1="7" x2="12" y2="17" stroke="{color}" stroke-width="2"/></svg>'
    ),
    "manufacturing": (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M9 3L9 21M15 3L15 21M3 9H21M3 15H21" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>'
    ),
    "compliance": (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M4 20L8 12L12 16L16 8L20 4" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    ),
}


# ── Componentes HTML ────────────────────────────────────────────────

def landing_card(icon, title, description, link_text="", variant="info"):
    """Genera HTML premium para una card de la landing page.

    Incluye gradiente de fondo, icono con fondo circular, hover con sombra.
    """
    theme = "dark" if is_dark() else "light"
    p = _LANDING_PALETTES.get(variant, _LANDING_PALETTES["info"])[theme]

    style_card = (
        f"background:{p['gradient']};"
        f"padding:2rem 1.5rem;border-radius:16px;"
        f"text-align:center;height:240px;display:flex;flex-direction:column;"
        f"justify-content:center;align-items:center;"
        f"border:1.5px solid {p['border']};"
        f"box-shadow:0 4px 20px rgba(0,0,0,0.08);"
        f"transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);"
        f"cursor:pointer;position:relative;overflow:hidden;"
    )
    style_icon_circle = (
        f"width:64px;height:64px;border-radius:50%;"
        f"background:{p['icon_bg']};"
        f"display:flex;align-items:center;justify-content:center;"
        f"margin-bottom:0.8rem;font-size:2rem;"
        f"box-shadow:0 2px 8px rgba(0,0,0,0.06);"
    )
    style_title = f"margin:0;color:{p['title']};font-size:1.1rem;font-weight:700;"
    style_desc = f"margin:0.4rem 0 0 0;font-size:0.85rem;color:{p['text']};line-height:1.4;"
    style_link = (
        f"margin:0.8rem 0 0 0;font-size:0.78rem;color:{p['link']};"
        f"font-weight:600;text-transform:uppercase;letter-spacing:0.05em;"
    )
    style_badge = (
        f"position:absolute;top:12px;right:12px;"
        f"background:{p['link']};color:white;font-size:0.65rem;"
        f"padding:2px 8px;border-radius:10px;font-weight:600;"
    )

    link_part = f'<p style="{style_link}">{link_text} →</p>' if link_text else ""

    return (
        f'<div style="{style_card}" '
        f'onmouseover="this.style.boxShadow=\'0 8px 32px rgba(0,0,0,0.15)\';this.style.transform=\'translateY(-3px)\'" '
        f'onmouseout="this.style.boxShadow=\'0 4px 20px rgba(0,0,0,0.08)\';this.style.transform=\'translateY(0)\'">'
        f'<div style="{style_icon_circle}">{icon}</div>'
        f'<h3 style="{style_title}">{title}</h3>'
        f'<p style="{style_desc}">{description}</p>'
        f'{link_part}'
        f'</div>'
    )


def metric_card(value, title, subtitle, variant="beta0", value_color=None):
    """Genera HTML premium para una card de métrica TDA.

    Incluye icono SVG, sombra, hover con elevación.
    """
    theme = "dark" if is_dark() else "light"
    p = _METRIC_PALETTES.get(variant, _METRIC_PALETTES["beta0"])[theme]

    if value_color:
        val_color = value_color
    else:
        val_color = {"light": "#2c3e50", "dark": "#ecf0f1"}.get(theme, "#2c3e50")

    icon_html = _METRIC_ICONS.get(variant, "").format(color=p["icon_color"])

    style_card = (
        f"background:{p['bg']};padding:20px;border-radius:14px;text-align:center;"
        f"box-shadow:0 2px 12px rgba(0,0,0,0.06);"
        f"transition:all 0.2s cubic-bezier(0.4, 0, 0.2, 1);"
        f"border:1px solid rgba(0,0,0,0.04);"
    )
    style_icon = f"margin-bottom:8px;opacity:0.9;"
    style_val = f"font-size:2.2em;font-weight:700;margin:0;color:{val_color};font-family:'JetBrains Mono',monospace;"
    style_title = f"margin:4px 0 0 0;color:{p['h3']};font-size:0.85rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;"
    style_small = f"color:{p['small']};font-size:0.8rem;margin-top:4px;"

    return (
        f'<div style="{style_card}" '
        f'onmouseover="this.style.boxShadow=\'0 6px 24px rgba(0,0,0,0.12)\';this.style.transform=\'translateY(-2px)\'" '
        f'onmouseout="this.style.boxShadow=\'0 2px 12px rgba(0,0,0,0.06)\';this.style.transform=\'translateY(0)\'">'
        f'<div style="{style_icon}">{icon_html}</div>'
        f'<p style="{style_val}">{value}</p>'
        f'<h3 style="{style_title}">{title}</h3>'
        f'<p style="{style_small}">{subtitle}</p>'
        f'</div>'
    )
def footer_style():
    """Devuelve el estilo CSS para el footer según el tema."""
    color = "#7f8c8d" if is_dark() else "#95a5a6"
    return f"text-align:center;color:{color};font-size:0.85rem;padding:1rem 0;"


# ═══════════════════════════════════════════════════════════════
# DIAGNÓSTICOS: explicación condicional de hipótesis
# ═══════════════════════════════════════════════════════════════

def _badge(icono: str, label: str, color: str) -> str:
    """Badge inline tiny para resumen de estado."""
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:6px;'
        f'font-size:0.78rem;font-weight:600;margin-right:4px;'
        f'background:{color}22;color:{color};border:1px solid {color}44">'
        f'{icono} {label}</span>'
    )


def report_header(title, subtitle=""):
    """Genera HTML para un header con gradiente violeta (reportes)."""
    style_hdr = (
        "background:linear-gradient(135deg, #2c1a00 0%, #3d2200 100%);"
        "padding:24px;border-radius:14px;margin-bottom:20px;"
        f"border-left:4px solid {ORANGE};"
        "box-shadow:0 4px 20px rgba(255,107,53,0.2);"
    )
    sub = (
        f'<p style="color:rgba(255,255,255,0.85);text-align:center;margin:5px 0 0;font-size:0.9rem;">{subtitle}</p>'
        if subtitle else ""
    )
    return (
        f'<div style="{style_hdr}">'
        f'<h3 style="color:white;margin:0;text-align:center;font-weight:700;">{title}</h3>'
        f'{sub}'
        f'</div>'
    )
def responsive_style():
    """Inyecta CSS inline para mejorar la visualización en pantallas angostas.

    Incluye: tipografía custom, responsive, animaciones, progress bars.
    """
    return custom_fonts() + """
<style>
/* ═══════════════════════════════════════════════════════════════
   RESPONSIVE UNIVERSAL — cualquier dispositivo
   ═══════════════════════════════════════════════════════════════ */

/* Nada se sale del ancho */
.stPlotlyChart, .js-plotly-plot, .stImage, img, .stDataFrame, .stTable {
    max-width: 100% !important;
}
img { max-width: 100%; height: auto; }
.stDataFrame, .stTable { overflow-x: auto !important; }
.js-plotly-plot .plotly, .js-plotly-plot .plot-container { width: 100% !important; }

/* Tipografía fluida */
h1, .stMarkdown h1 { font-size: clamp(1.3rem, 4.5vw, 2.2rem) !important; }
h2, .stMarkdown h2 { font-size: clamp(1.15rem, 3.6vw, 1.7rem) !important; }
h3, .stMarkdown h3 { font-size: clamp(1.0rem, 3vw, 1.35rem) !important; }

/* Padding del área principal fluido */
.block-container {
    padding-left: clamp(0.6rem, 4vw, 4rem) !important;
    padding-right: clamp(0.6rem, 4vw, 4rem) !important;
    padding-top: 2rem !important;
    max-width: 100% !important;
}

/* ── MÓVIL (≤768px) ── */
@media (max-width: 768px) {
    /* Todas las columnas se apilan */
    [data-testid="stHorizontalBlock"] > div {
        min-width: 100% !important;
        flex: 1 1 100% !important;
        padding-right: 0 !important;
        padding-left: 0 !important;
    }
    /* Métricas */
    [data-testid="stMetric"] { padding: 0.5rem 0.4rem !important; }
    [data-testid="stMetric"] label { font-size: 0.75rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    /* Botones / descargas full width */
    .stButton > button, .stDownloadButton > button { width: 100% !important; }
    /* Cabecera */
    [data-testid="stHeader"] { padding: 0.4rem 0.4rem !important; }
    /* Sidebar compactada */
    [data-testid="stSidebar"] { min-width: 260px !important; }
    [data-testid="stSidebar"][aria-expanded="false"] { min-width: 0 !important; }
    /* Tablas y tabs */
    .stDataFrame { font-size: 0.8rem !important; }
    .stTabs [data-baseweb="tab-list"] { flex-wrap: nowrap !important; overflow-x: auto !important; }
    /* Títulos */
    h1, .stMarkdown h1 { font-size: 1.4rem !important; }
    /* Leyendas de Plotly más chicas */
    .js-plotly-plot .legend { font-size: 10px !important; }
}

/* ── MÓVIL CHICO (≤480px) ── */
@media (max-width: 480px) {
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 0.95rem !important; }
    h1, .stMarkdown h1 { font-size: 1.2rem !important; }
    .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    .stDataFrame { font-size: 0.72rem !important; }
}

/* ── TABLET (769–1024px) ── */
@media (min-width: 769px) and (max-width: 1024px) {
    [data-testid="stHorizontalBlock"] > div { min-width: 45% !important; }
}

/* ═══════════════════════════════════════════════════════════════
   GENERAL
   ═══════════════════════════════════════════════════════════════ */

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* Botones de descarga */
.stDownloadButton {
    margin-bottom: 0.5rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
    font-size: 0.95rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.8;
}

/* Section dividers */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(0,0,0,0.1), transparent);
    margin: 1.5rem 0;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Orange accent overrides for dark theme */
[data-testid="stButton"] button {
    border-color: #FF6B35;
}
[data-testid="stButton"]:hover {
    background-color: #FF6B35;
    color: #121212;
}
hr {
    background: linear-gradient(to right, transparent, #FF6B35, transparent) !important;
}

/* ═══════════════════════════════════════════════════════════════
   DESCRIPCIÓN ↔ FIGURA — sin huecos que las desvinculen
   ═══════════════════════════════════════════════════════════════ */
/* Compacta el aire vertical entre bloques (la figura queda pegada a su rótulo) */
[data-testid="stVerticalBlock"] { gap: 0.55rem !important; }
/* Encabezado de figura: sin margen inferior residual */
[data-testid="stMarkdownContainer"] > p:last-child { margin-bottom: 0.1rem !important; }
/* La figura arranca inmediatamente debajo de su descripción */
[data-testid="stPlotlyChart"] { margin-top: -0.15rem !important; }
/* Si la descripción precede a una figura, se anulan los márgenes entre ambas */
[data-testid="stMarkdown"]:has(+ [data-testid="element-container"] [data-testid="stPlotlyChart"]) {
    margin-bottom: -0.55rem !important;
}
.element-container:has([data-testid="stPlotlyChart"]) { margin-top: -0.1rem !important; }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# NAVEGACIÓN Y ESTRUCTURA DE PÁGINA
# ═══════════════════════════════════════════════════════════════

def sidebar_nav(current_page, page_map):
    """Genera navegación en sidebar con page_link de Streamlit.

    Parameters
    ----------
    current_page : str
        Nombre del archivo de la página actual (ej. "pages/2_H.E.2_...")
    page_map : dict
        {icon: str, title: str, path: str} para cada página
    """
    st.sidebar.markdown("---")

    for icon, title, path in page_map:
        if path == current_page:
            st.sidebar.markdown(f"**{title}**")
        else:
            st.sidebar.page_link(path, label=f"{title}")


def breadcrumbs(parts, separator="→"):
    """Genera breadcrumbs HTML estilizados con acento naranja.

    Parameters
    ----------
    parts : list of str
        Ejemplo: ["Tesis", "H.E.2", "Resultados"]
    separator : str
        Separador entre partes
    """
    items = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            items.append(f'<span style="color:{ORANGE_HOVER};cursor:pointer;">{part}</span>')
        else:
            items.append(f'<span style="color:{ORANGE};font-weight:700;">{part}</span>')
    breadcrumb_html = f" <span style='color:#666;'>{separator}</span> ".join(items)
    css = """
    <style>
    .breadcrumb-bar {
        background: #1a1a1a;
        padding: 8px 16px;
        border-radius: 8px;
        border-left: 3px solid #FF6B35;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    </style>
    <div class="breadcrumb-bar">""" + breadcrumb_html + "</div>"
    return css


def methodology_expander(title, content_items, page_name):
    """Genera un expandable con el plan de acción completo de una hipótesis.

    Parameters
    ----------
    title : str
        Título del expandable (ej. "📖 Plan de Acción")
    content_items : list of (label, text) tuples
        Contenido a mostrar: fórmulas, referencias, algoritmo
    page_name : str
        Nombre de la página para el título

    Returns
    -------
    str HTML o markdown para st.markdown()
    """
    def _is_latex(text):
        """Detecta si el contenido contiene sintaxis LaTeX."""
        latex_markers = ['\\frac', '\\sqrt', '\\beta', '\\mu', '\\alpha',
                         '\\leq', '\\geq', '\\rightarrow', '\\text{',
                         '\\mapsto', '\\mathbb', '\\in', '\\quad',
                         '\\Rightarrow', '\\implies', '\\sum', '\\int',
                         '\\dot', '\\ddot', '\\partial', '\\nabla',
                         '\\left(', '\\right)', '\\left[', '\\right]',
                         '\\left\\{', '\\right\\}', '\\cdot', '\\times',
                         '\\sigma', '\\delta', '\\rho', '\\phi']
        return any(m in text for m in latex_markers)

    with st.expander(title, expanded=False):
        st.markdown(f"### {page_name} — Protocolo y compuertas")

        for label, text in content_items:
            if label == "fórmulas":
                st.markdown("#### Fórmulas Principales")
                st.latex(text)
            elif label == "referencias":
                st.markdown("#### Referencias")
                for ref in text:
                    st.markdown(f"- {ref}")
            elif label == "algoritmo":
                st.markdown("#### Algoritmo")
                st.code(text, language="text")
            else:
                st.markdown(f"**{label}**")
                st.markdown(text)

        st.markdown("---")
        st.caption("Trazabilidad con el Cap. IV del documento.")


# ═══════════════════════════════════════════════════════════════
# HELPERS DE PÁGINA
# ═══════════════════════════════════════════════════════════════

def page_header(title, subtitle=""):
    """Header premium con gradiente naranja para cada página."""
    return f"""
<div style="
    background: linear-gradient(135deg, #2c1a00 0%, #3d2200 40%, #1a0f00 70%);
    padding: clamp(1rem, 3vw, 1.5rem) clamp(1rem, 4vw, 2rem);
    border-radius: 16px;
    border-left: 4px solid {ORANGE};
    margin-bottom: 1.5rem;
">
    <h2 style="margin:0;color:{ORANGE};font-weight:800;font-size:clamp(1.1rem, 4vw, 1.6rem);">{title}</h2>
    {f'<p style="margin:0.3rem 0 0 0;color:{WHITE_DIM};font-size:clamp(0.8rem, 2.5vw, 0.95rem);">{subtitle}</p>' if subtitle else ''}
</div>
"""
def update_hero_orange(hero_html):
    """Reemplaza el gradiente azul del hero por naranja oscuro."""
    return hero_html.replace(
        "linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 70%, #1a1a2e 100%)",
        "linear-gradient(135deg, #2c1a00 0%, #3d2200 40%, #1a0f00 70%, #2c1a00 100%)"
    ).replace("#1a1a2e", "#2c1a00").replace("#16213e", "#3d2200").replace("#0f3460", "#1a0f00")
