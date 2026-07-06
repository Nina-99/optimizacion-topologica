"""Utilidades de theming para la app Streamlit TDA-SIMP.

Soporte de modo oscuro/claro con componentes adaptables.
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ── Detección de tema ───────────────────────────────────────────────

def is_dark():
    """Detecta si Streamlit está en modo oscuro.

    Returns
    -------
    bool
        True si el tema base es "dark", False en caso contrario.
    """
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


# ── Matplotlib ──────────────────────────────────────────────────────

def apply_mpl_theme():
    """Configura matplotlib para renderizado inline según el tema.

    Llamar ANTES de crear figuras que se muestren con st.pyplot().
    Para figuras exportadas (PDF/PNG), usar explícitamente fondo blanco.
    """
    if is_dark():
        plt.style.use("dark_background")
    else:
        plt.style.use("default")


def figure_for_export():
    """Crea una figura matplotlib con fondo blanco siempre (para exportación).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure()
    fig.patch.set_facecolor("white")
    return fig


# ── Plotly ──────────────────────────────────────────────────────────

def apply_plotly_theme(fig):
    """Aplica el template de Plotly según el tema actual.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figura a la que aplicar el tema.
    """
    if is_dark():
        fig.update_layout(template="plotly_dark")
    else:
        fig.update_layout(template="plotly_white")
    return fig


# ── Paletas de color ────────────────────────────────────────────────

_LANDING_PALETTES = {
    "info": {
        "light": {
            "bg": "#f0f4f8", "border": "#d5dbdb",
            "title": "#2c3e50", "text": "#7f8c8d", "link": "#3498db",
        },
        "dark": {
            "bg": "#1a2634", "border": "#3498db",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#5dade2",
        },
    },
    "warning": {
        "light": {
            "bg": "#fef9e7", "border": "#f5d76e",
            "title": "#2c3e50", "text": "#7f8c8d", "link": "#e67e22",
        },
        "dark": {
            "bg": "#2c1f0a", "border": "#e67e22",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#f39c12",
        },
    },
    "success": {
        "light": {
            "bg": "#eafaf1", "border": "#82e0aa",
            "title": "#2c3e50", "text": "#7f8c8d", "link": "#27ae60",
        },
        "dark": {
            "bg": "#0a2a1a", "border": "#27ae60",
            "title": "#ecf0f1", "text": "#bdc3c7", "link": "#2ecc71",
        },
    },
}

_METRIC_PALETTES = {
    "beta0": {
        "light": {"bg": "#e8f4f8", "h3": "#2980b9", "small": "#7f8c8d"},
        "dark": {"bg": "#1a2634", "h3": "#5dade2", "small": "#bdc3c7"},
    },
    "beta1": {
        "light": {"bg": "#fef9e7", "h3": "#d35400", "small": "#7f8c8d"},
        "dark": {"bg": "#2c1f0a", "h3": "#e67e22", "small": "#bdc3c7"},
    },
    "mu": {
        "light": {"bg": "#eafaf1", "h3": "#27ae60", "small": "#7f8c8d"},
        "dark": {"bg": "#0a2a1a", "h3": "#2ecc71", "small": "#bdc3c7"},
    },
    "manufacturing": {
        "light": {"bg": "#f4f6f7", "h3": "#7f8c8d", "small": "#7f8c8d"},
        "dark": {"bg": "#2c2c2c", "h3": "#95a5a6", "small": "#bdc3c7"},
    },
}


# ── Componentes HTML ────────────────────────────────────────────────

def landing_card(icon, title, description, link_text="", variant="info"):
    """Genera HTML para una card de la landing page.

    Parameters
    ----------
    icon : str
        Emoji o ícono (ej. "📊")
    title : str
        Título de la card
    description : str
        Descripción (soporta <br>)
    link_text : str, optional
        Texto opcional del hint de navegación
    variant : str
        "info", "warning" o "success"

    Returns
    -------
    str
        HTML listo para st.markdown(unsafe_allow_html=True)
    """
    theme = "dark" if is_dark() else "light"
    p = _LANDING_PALETTES.get(variant, _LANDING_PALETTES["info"])[theme]

    style_card = (
        f"background:{p['bg']};padding:1.8rem 1.2rem;border-radius:14px;"
        f"text-align:center;height:220px;display:flex;flex-direction:column;"
        f"justify-content:center;border:2px solid {p['border']};transition:all 0.2s;"
    )
    style_icon = "font-size:3rem;margin-bottom:0.5rem;"
    style_title = f"margin:0.3rem 0;color:{p['title']};"
    style_desc = f"margin:0.3rem 0 0 0;font-size:0.9rem;color:{p['text']};"
    style_link = f"margin:1rem 0 0 0;font-size:0.8rem;color:{p['link']};font-weight:600;"

    link_part = f'<p style="{style_link}">{link_text}</p>' if link_text else ""

    return (
        f'<div style="{style_card}">'
        f'<div style="{style_icon}">{icon}</div>'
        f'<h3 style="{style_title}">{title}</h3>'
        f'<p style="{style_desc}">{description}</p>'
        f'{link_part}'
        f'</div>'
    )


def metric_card(value, title, subtitle, variant="beta0", value_color=None):
    """Genera HTML para una card de métrica TDA.

    Parameters
    ----------
    value : str
        Valor numérico a mostrar
    title : str
        Nombre/símbolo de la métrica (ej. "β₀")
    subtitle : str
        Descripción breve
    variant : str
        "beta0", "beta1", "mu" o "manufacturing"
    value_color : str, optional
        Color específico para el valor (para colores dinámicos)

    Returns
    -------
    str
        HTML listo para st.markdown(unsafe_allow_html=True)
    """
    theme = "dark" if is_dark() else "light"
    p = _METRIC_PALETTES.get(variant, _METRIC_PALETTES["beta0"])[theme]

    if value_color:
        val_color = value_color
    else:
        val_color = {"light": "#2c3e50", "dark": "#ecf0f1"}.get(theme, "#2c3e50")

    style_card = f"background:{p['bg']};padding:15px;border-radius:10px;text-align:center;"
    style_val = f"font-size:2em;font-weight:bold;margin:0;color:{val_color};"
    style_small = f"color:{p['small']};"

    return (
        f'<div style="{style_card}">'
        f'<h3 style="margin:0;color:{p["h3"]};">{title}</h3>'
        f'<p style="{style_val}">{value}</p>'
        f'<small style="{style_small}">{subtitle}</small>'
        f'</div>'
    )


def footer_style():
    """Devuelve el estilo CSS para el footer según el tema."""
    color = "#7f8c8d" if is_dark() else "#95a5a6"
    return f"text-align:center;color:{color};font-size:0.85rem;padding:1rem 0;"


def hero_section(title, subtitle=""):
    """Genera HTML para el hero banner con gradiente oscuro.

    Funciona en ambos modos (el gradiente oscuro contrasta bien
    tanto en fondo claro como oscuro).
    """
    style_hero = (
        "background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);"
        "padding:2.5rem 2rem;border-radius:16px;color:white;margin-bottom:2rem;"
        "text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.2);"
    )
    sub = (
        f'<p style="margin:0.8rem 0 0 0;opacity:0.9;font-size:1.1rem;'
        f'max-width:700px;margin-left:auto;margin-right:auto;">{subtitle}</p>'
        if subtitle else ""
    )
    return (
        f'<div style="{style_hero}">'
        f'<h1 style="margin:0;font-weight:700;font-size:2.2rem;">{title}</h1>'
        f'{sub}'
        f'</div>'
    )


def report_header(title, subtitle=""):
    """Genera HTML para un header con gradiente violeta (reportes).

    El gradiente oscuro funciona en ambos modos (claro/oscuro).

    Parameters
    ----------
    title : str
        Título del header
    subtitle : str, optional
        Subtítulo opcional

    Returns
    -------
    str
        HTML listo para st.markdown(unsafe_allow_html=True)
    """
    style_hdr = (
        "background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);"
        "padding:20px;border-radius:12px;margin-bottom:20px;"
    )
    sub = (
        f'<p style="color:rgba(255,255,255,0.85);text-align:center;margin:5px 0 0;">{subtitle}</p>'
        if subtitle else ""
    )
    return (
        f'<div style="{style_hdr}">'
        f'<h3 style="color:white;margin:0;text-align:center;">{title}</h3>'
        f'{sub}'
        f'</div>'
    )


# ── Responsive CSS ──────────────────────────────────────────────────

def responsive_style():
    """Inyecta CSS inline para mejorar la visualización en pantallas angostas.

    Ajusta columnas, tarjetas y botones para que no se vean apretados
    en viewports menores a 768px de ancho.

    Llamar al inicio de cada página::

        st.markdown(responsive_style(), unsafe_allow_html=True)
    """
    return """
<style>
/* Columnas responsivas: en mobile se apilan en una sola columna */
@media (max-width: 768px) {
    /* Forzar que los elementos dentro de st.columns se apilen */
    .stColumn > div {
        min-width: 100% !important;
        flex: 0 0 100% !important;
    }
    .stColumn {
        gap: 0 !important;
    }
    /* Botones de descarga ocupan todo el ancho */
    .stDownloadButton button {
        width: 100% !important;
    }
    /* Tarjetas: padding reducido */
    div[data-testid="stVerticalBlock"] > div > div > div > div > div[style*="border-radius"] {
        padding: 1rem 0.8rem !important;
    }
}
/* Ajustes generales */
.stDownloadButton {
    margin-bottom: 0.5rem;
}
</style>
"""
