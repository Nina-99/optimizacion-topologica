"""Utilidades de theming para la app Streamlit TDA-SIMP.

Soporte de modo oscuro/claro con componentes adaptables.
Incluye: tipografía custom, cards premium, progress por fases, badges.
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ── Detección de tema ───────────────────────────────────────────────

def is_dark():
    """Detecta si Streamlit está en modo oscuro."""
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


# ── Tipografía custom ───────────────────────────────────────────────

def custom_fonts():
    """Inyecta Google Fonts (Inter + JetBrains Mono) y estilos base.

    Llamar una sola vez al inicio de cada página o en responsive_style().
    """
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Tipografía base */
body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Números de métricas: JetBrains Mono */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
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


def figure_for_export():
    """Crea una figura matplotlib con fondo blanco siempre (para exportación)."""
    fig = plt.figure()
    fig.patch.set_facecolor("white")
    return fig


# ── Plotly ──────────────────────────────────────────────────────────

def apply_plotly_theme(fig):
    """Aplica el template de Plotly según el tema actual."""
    if is_dark():
        fig.update_layout(template="plotly_dark")
    else:
        fig.update_layout(template="plotly_white")
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


def stat_badge(number, label):
    """Genera un badge de estadística animada para el hero.

    Parameters
    ----------
    number : str or int
        Número a mostrar (ej. "3", "12", "21")
    label : str
        Descripción del número (ej. "Hipótesis Validadas")

    Returns
    -------
    str
        HTML listo para st.markdown()
    """
    return (
        f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
        f'padding:0.8rem 1.5rem;background:rgba(255,255,255,0.1);'
        f'border-radius:12px;backdrop-filter:blur(4px);'
        f'border:1px solid rgba(255,255,255,0.15);margin:0 0.5rem;">'
        f'<span style="font-size:2rem;font-weight:800;color:white;font-family:\'JetBrains Mono\',monospace;">{number}</span>'
        f'<span style="font-size:0.75rem;color:rgba(255,255,255,0.8);text-transform:uppercase;letter-spacing:0.08em;font-weight:500;">{label}</span>'
        f'</div>'
    )


def phase_progress(fase_actual, total_fases, labels, status_text=""):
    """Barra de progreso compacta con st.progress() y caption inline.

    Parameters
    ----------
    fase_actual : int
        Fase actual (1-indexed). Si fase_actual > total_fases, muestra 100%.
    total_fases : int
        Total de fases
    labels : list of str
        Labels para cada fase (se usa solo el de la fase activa)
    status_text : str, optional
        Texto de estado adicional debajo de la barra
    """
    pct = min(fase_actual, total_fases) / total_fases
    st.progress(pct)
    active_label = labels[fase_actual - 1] if 1 <= fase_actual <= total_fases else labels[-1]
    icon = "✓" if fase_actual > total_fases else "⏳"
    status = status_text if status_text else active_label
    st.caption(f"{icon} **{active_label}** — {status}")


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


def diagnosticar_he2(reduccion: float, beta1: int, mu: float,
                      alpha: float, rmin: float, penal: float,
                      volfrac: float, n_iter: int, converged: bool) -> dict:
    """Diagnóstico de H.E.2: genera badges, razones y sugerencias.

    Returns dict con keys:
        badges_html: str  — HTML inline con badges resumen
        razones: list[str] — explicaciones por qué pasa/falla
        sugerencias: list[str] — qué ajustar
        veredicto: str     — "CUMPLIDA" / "PARCIAL" / "NO CUMPLIDA"
    """
    cumple_red = reduccion >= 40
    cumple_b1 = beta1 <= 2

    badges = []
    razones = []
    sugerencias = []

    # ── Badges ──
    badges.append(_badge("✅" if cumple_red else "❌",
                         f"Reducción {reduccion:.1f}%",
                         "#27ae60" if cumple_red else "#e74c3c"))
    badges.append(_badge("✅" if cumple_b1 else "❌",
                         f"β₁={beta1}",
                         "#27ae60" if cumple_b1 else "#e74c3c"))
    badges.append(_badge("📊", f"μ_α={mu:.4f}", "#2980b9"))

    # ── Razones ──
    if cumple_red:
        razones.append(
            f"La optimización SIMP con p={penal}, f_V={volfrac} redujo la compliance "
            f"un {reduccion:.1f}% respecto al diseño homogéneo inicial, superando "
            f"el umbral del 40% del Documento Completo."
        )
    else:
        if n_iter < 150:
            razones.append(
                f"La reducción ({reduccion:.1f}%) es insuficiente porque la optimización "
                f"solo corrió {n_iter} iteraciones y {'no convergió' if not converged else 'convergió prematuramente'}. "
                f"El optimizador OC necesita más iteraciones para alcanzar el diseño óptimo."
            )
        elif volfrac > 0.6:
            razones.append(
                f"Con f_V={volfrac} (>0.5), el diseño inicial ya tiene poco espacio para "
                f"mejorar la rigidez. La reducción marginal es baja."
            )
        else:
            razones.append(
                f"La reducción ({reduccion:.1f}%) no alcanza el 40%. "
                f"El diseño converge a una topología subóptima."
            )

    if cumple_b1:
        razones.append(
            f"β₁={beta1} ≤ 2: la topología resultante tiene máximo 2 agujeros "
            f"topológicos, cumpliendo el criterio de manufacturabilidad."
        )
    else:
        ciclos_extra = beta1 - 2
        razones.append(
            f"β₁={beta1} > 2: hay {ciclos_extra} agujero(s) extra(s). "
            f"Estos son ciclos espurios generados por la discretización de la malla "
            f"o por el filtro de sensibilidad con rmin={rmin}."
        )

    # ── Sensibilidad de parámetros ──
    if not cumple_red or not cumple_b1:
        razones.append(
            "**Sensibilidad de parámetros:**\n"
            "- `α` ↑ → penaliza más agujeros (β₁ baja) pero puede subir compliance\n"
            "- `α` ↑↑ → over-penaliza, topología muy sólida, compliance sube\n"
            "- `rmin` ↑ → suaviza la topología, reduce ciclos espurios\n"
            "- `p` ↑ → penaliza más materia intermedia, topología más binaria\n"
            "- `max_iter` ↑ → mejor convergencia de compliance"
        )

    # ── Sugerencias ──
    if not cumple_red:
        sugerencias.append(
            f"Aumentar `max_iter` a {max(n_iter, 200)} para mejor convergencia."
        )
        if not converged:
            sugerencias.append(
                f"La optimización no convergió con tol=10⁻⁴. Considerar relajar a tol=10⁻³ "
                f"o aumentar `max_iter`."
            )

    if not cumple_b1:
        alpha_sugerido = min(alpha * 3, 1.0)
        rmin_sugerido = max(rmin, 3.0)
        sugerencias.append(
            f"Aumentar α de {alpha:.3f} a ~{alpha_sugerido:.3f} para penalizar más agujeros."
        )
        sugerencias.append(
            f"Aumentar rmin de {rmin:.1f} a ~{rmin_sugerido:.1f} para suavizar ciclos espurios."
        )

    if cumple_red and cumple_b1:
        sugerencias.append("Los parámetros actuales son adecuados. No se requieren ajustes.")

    # ── Veredicto ──
    if cumple_red and cumple_b1:
        veredicto = "CUMPLIDA"
    elif cumple_red or cumple_b1:
        veredicto = "PARCIAL"
    else:
        veredicto = "NO CUMPLIDA"

    return {
        "badges_html": " ".join(badges),
        "razones": razones,
        "sugerencias": sugerencias,
        "veredicto": veredicto,
    }


def diagnosticar_hg(c_eucl: float, c_tda: float, beta1_eucl: int,
                     beta1_tda: int, mu_tda: float, alpha: float) -> dict:
    """Diagnóstico de H.G: comparación euclidiano vs TDA.

    Returns dict con keys:
        badges_html, razones, sugerencias, veredicto
    """
    mejora_c = c_tda < c_eucl
    mejora_b1 = beta1_tda <= beta1_eucl
    delta_c = c_tda - c_eucl
    delta_b1 = beta1_tda - beta1_eucl

    badges = []
    razones = []
    sugerencias = []

    # ── Badges ──
    icon_c = "↓" if mejora_c else "↑" if delta_c > 0 else "="
    badges.append(_badge(
        "✅" if mejora_c else "⚠️" if delta_c == 0 else "❌",
        f"Compliance {icon_c} {abs(delta_c):.2f}",
        "#27ae60" if mejora_c else "#f39c12" if delta_c == 0 else "#e74c3c"
    ))
    icon_b = "↓" if mejora_b1 else "↑" if delta_b1 > 0 else "="
    badges.append(_badge(
        "✅" if mejora_b1 else "⚠️" if delta_b1 == 0 else "❌",
        f"β₁ {icon_b} {abs(delta_b1)}",
        "#27ae60" if mejora_b1 else "#f39c12" if delta_b1 == 0 else "#e74c3c"
    ))
    badges.append(_badge("📊", f"μ_α={mu_tda:.4f}", "#2980b9"))

    # ── Razones ──
    if mejora_c:
        pct = (1 - c_tda / c_eucl) * 100 if c_eucl > 0 else 0
        razones.append(
            f"SIMP+TDA produce compliance {abs(delta_c):.2f} menor que SIMP euclidiano "
            f"({pct:.1f}% de mejora). La regularización topológica μ_α guía "
            f"el OC hacia diseños más eficientes."
        )
    elif delta_c == 0:
        razones.append(
            "Ambos enfoques producen compliance idéntica. Esto ocurre cuando "
            "α=0 o cuando la topología converge al mismo diseño sin importar la "
            "regularización topológica."
        )
    else:
        razones.append(
            f"SIMP euclidiano produce compliance {abs(delta_c):.2f} menor. "
            f"Esto sugiere que α={alpha} está over-penalizando la función objetivo, "
            f"forzando una topología más simple a costa de rigidez."
        )

    if mejora_b1:
        if delta_b1 < 0:
            razones.append(
                f"β₁ baja de {beta1_eucl} a {beta1_tda}: TDA suprime "
                f"{abs(delta_b1)} agujero(s) espurio(s), mejorando la manufacturabilidad."
            )
        else:
            razones.append(
                f"β₁ se mantiene en {beta1_tda}: ambos enfoques producen "
                f"topología con la misma cantidad de agujeros."
            )
    else:
        razones.append(
            f"β₁ sube de {beta1_eucl} a {beta1_tda}: la regularización TDA "
            f"está introduciendo ciclos adicionales en lugar de suprimirlos. "
            f"Esto indica que α={alpha} es demasiado bajo para esta malla."
        )

    # ── Sensibilidad ──
    razones.append(
        "**Relación α ↔ resultados:**\n"
        "- `α` = 0 → euclidiano puro (sininfluencia topológica)\n"
        "- `α` pequeño (0.01-0.05) → efecto sutil, puede no cambiar β₁\n"
        "- `α` medio (0.05-0.2) → suprime ciclos espurios, aumenta ligeramente c\n"
        "- `α` alto (>0.2) → over-penaliza, topología muy sólida, c sube mucho"
    )

    # ── Sugerencias ──
    if not mejora_c and not mejora_b1:
        sugerencias.append(
            f"α={alpha} no produce mejoras. Probar α=0 (euclidiano puro) como baseline, "
            f"o subir α a ~{min(alpha * 5, 0.5):.3f} para efecto más marcado."
        )
    elif not mejora_c:
        sugerencias.append(
            f"La compliance empeora con TDA. Reducir α de {alpha:.3f} a ~{alpha * 0.5:.3f} "
            f"para mitigar la penalización excesiva."
        )
    elif not mejora_b1:
        sugerencias.append(
            f"Compliance mejora pero β₁ no. Subir α de {alpha:.3f} a ~{min(alpha * 3, 0.3):.3f} "
            f"para forzar supresión de agujeros."
        )
    else:
        sugerencias.append("Ambos criterios se cumplen. Los parámetros son adecuados.")

    if mejora_c and mejora_b1:
        veredicto = "VALIDADA"
    elif mejora_c or mejora_b1:
        veredicto = "PARCIAL"
    else:
        veredicto = "NO VALIDADA"

    return {
        "badges_html": " ".join(badges),
        "razones": razones,
        "sugerencias": sugerencias,
        "veredicto": veredicto,
    }


def hero_section(title, subtitle="", badges=None):
    """Genera HTML para el hero banner premium con gradiente animado.

    Parameters
    ----------
    title : str
        Título principal
    subtitle : str, optional
        Subtítulo descriptivo
    badges : list of tuple, optional
        Lista de (número, label) para mostrar como badges estadísticos
        Ej: [("3", "Hipótesis"), ("12", "Mejoras"), ("21", "Tests")]
    """
    badges_html = ""
    if badges:
        badges_html = '<div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.5rem;margin-top:1.5rem;">'
        for num, label in badges:
            badges_html += stat_badge(num, label)
        badges_html += '</div>'

    style_hero = (
        "background:linear-gradient(135deg, #2c1a00 0%, #3d2200 40%, #1a0f00 70%, #2c1a00 100%);"
        "background-size:200% 200%;"
        "animation:heroGradient 8s ease infinite;"
        "padding:3rem 2rem;border-radius:20px;color:white;margin-bottom:2rem;"
        "text-align:center;box-shadow:0 12px 40px rgba(0,0,0,0.25);"
        "position:relative;overflow:hidden;"
        f"border-left:5px solid {ORANGE};"
    )

    sub = (
        f'<p style="margin:1rem 0 0 0;opacity:0.9;font-size:1.05rem;'
        f'max-width:700px;margin-left:auto;margin-right:auto;line-height:1.6;">{subtitle}</p>'
        if subtitle else ""
    )

    # CSS keyframes para gradiente animado
    animation_css = """
    <style>
    @keyframes heroGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    </style>
    """

    return (
        animation_css +
        f'<div style="{style_hero}">'
        f'<h1 style="margin:0;font-weight:800;font-size:2.4rem;letter-spacing:-0.02em;">{title}</h1>'
        f'{sub}'
        f'{badges_html}'
        f'</div>'
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


def success_banner(title, text=""):
    """Genera un banner de éxito con acento naranja."""
    text_html = f'<p style="margin:4px 0 0;color:{WHITE_DIM};font-size:0.9rem;">{text}</p>' if text else ""
    return (
        f'<div style="background:linear-gradient(135deg, #1a3a0a 0%, #0a2a05 100%);'
        f'border-left:4px solid {ORANGE};border-radius:10px;padding:16px 20px;'
        f'box-shadow:0 2px 12px rgba(255,107,53,0.15);margin:10px 0;">'
        f'<p style="margin:0;font-weight:700;color:{ORANGE};">{title}</p>'
        f'{text_html}</div>'
    )


def warning_banner(title, text=""):
    """Genera un banner de advertencia con acento naranja."""
    text_html = f'<p style="margin:4px 0 0;color:{WHITE_DIM};font-size:0.9rem;">{text}</p>' if text else ""
    return (
        f'<div style="background:linear-gradient(135deg, #3d2200 0%, #2c1a00 100%);'
        f'border-left:4px solid {GOLD_WARN};border-radius:10px;padding:16px 20px;'
        f'box-shadow:0 2px 12px rgba(243,156,18,0.15);margin:10px 0;">'
        f'<p style="margin:0;font-weight:700;color:{GOLD_WARN};">{title}</p>'
        f'{text_html}</div>'
    )


# ── Responsive CSS ──────────────────────────────────────────────────

def responsive_style():
    """Inyecta CSS inline para mejorar la visualización en pantallas angostas.

    Incluye: tipografía custom, responsive, animaciones, progress bars.
    """
    return custom_fonts() + """
<style>
/* Columnas responsivas */
@media (max-width: 768px) {
    .stColumn > div {
        min-width: 100% !important;
        flex: 0 0 100% !important;
    }
    .stColumn {
        gap: 0 !important;
    }
    .stDownloadButton button {
        width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] > div > div > div > div > div[style*="border-radius"] {
        padding: 1rem 0.8rem !important;
    }
}

/* Botones de descarga */
.stDownloadButton {
    margin-bottom: 0.5rem;
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
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
    st.sidebar.title("📐 TDA-SIMP Master")
    st.sidebar.markdown("---")

    for icon, title, path in page_map:
        if path == current_page:
            st.sidebar.markdown(f"**{icon} {title}**")
        else:
            st.sidebar.page_link(path, label=f"{icon} {title}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Dark Theme · Orange Accent")


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
    """Genera un expandable con la metodología completa de una hipótesis.

    Parameters
    ----------
    title : str
        Título del expandable (ej. "📖 Metodología")
    content_items : list of (label, text) tuples
        Contenido a mostrar: fórmulas, referencias, algoritmo
    page_name : str
        Nombre de la página para el título

    Returns
    -------
    str HTML o markdown para st.markdown()
    """
    with st.expander(title, expanded=False):
        st.markdown(f"### {page_name} — Metodología")

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
                st.markdown(text)

        st.markdown("---")
        st.caption("Verificar consistencia con el Documento Completo y Proy.Investigacion.txt")


# ═══════════════════════════════════════════════════════════════
# HELPERS DE PÁGINA
# ═══════════════════════════════════════════════════════════════

def page_header(title, subtitle=""):
    """Header premium con gradiente naranja para cada página."""
    return f"""
<div style="
    background: linear-gradient(135deg, #2c1a00 0%, #3d2200 40%, #1a0f00 70%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    border-left: 4px solid {ORANGE};
    margin-bottom: 1.5rem;
">
    <h2 style="margin:0;color:{ORANGE};font-weight:800;font-size:1.6rem;">{title}</h2>
    {f'<p style="margin:0.3rem 0 0 0;color:{WHITE_DIM};font-size:0.95rem;">{subtitle}</p>' if subtitle else ''}
</div>
"""


def orange_banner(title, text=""):
    """Banner de éxito con acento naranja."""
    text_html = f'<p style="margin:4px 0 0;color:{WHITE_DIM};font-size:0.9rem;">{text}</p>' if text else ""
    return (
        f'<div style="background:linear-gradient(135deg, #3d2200 0%, #2c1a00 100%);'
        f'border-left:4px solid {ORANGE};border-radius:10px;padding:16px 20px;'
        f'box-shadow:0 2px 12px rgba(255,107,53,0.15);margin:10px 0;">'
        f'<p style="margin:0;font-weight:700;color:{ORANGE};">{title}</p>'
        f'{text_html}</div>'
    )


def orange_success_banner(title, text=""):
    """Banner de éxito con acento naranja."""
    text_html = f'<p style="margin:4px 0 0;color:{WHITE_DIM};font-size:0.9rem;">{text}</p>' if text else ""
    return (
        f'<div style="background:linear-gradient(135deg, #3d2200 0%, #2c1a00 100%);'
        f'border-left:4px solid {ORANGE};border-radius:10px;padding:16px 20px;'
        f'box-shadow:0 2px 12px rgba(255,107,53,0.15);margin:10px 0;">'
        f'<p style="margin:0;font-weight:700;color:{ORANGE};">{title}</p>'
        f'{text_html}</div>'
    )


def update_hero_orange(hero_html):
    """Reemplaza el gradiente azul del hero por naranja oscuro."""
    return hero_html.replace(
        "linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 70%, #1a1a2e 100%)",
        "linear-gradient(135deg, #2c1a00 0%, #3d2200 40%, #1a0f00 70%, #2c1a00 100%)"
    ).replace("#1a1a2e", "#2c1a00").replace("#16213e", "#3d2200").replace("#0f3460", "#1a0f00")
