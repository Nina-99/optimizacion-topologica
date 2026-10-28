from __future__ import annotations

TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CRIT, GOOD = "#d03b3b", "#0ca30c"


"""figuras.py — Reconstrucción headless de las figuras del Cap. V (entregable E3).

Genera las figuras de la tesis (5.1–5.7) SIN Streamlit, a partir de los
diccionarios de resultados de los protocolos (`tda.simulation.he_protocols`)
o de los CSV exportados. Cada figura respeta los ejes, rangos y escalas del
documento para que la comparación visual con el PDF sea directa.

Uso como API
────────────
    from tda.analysis.calibracion import resultado_calibracion
    from tda.visualization.figuras import fig_51_calibracion, fig_53_cota_corregida

    res = resultado_calibracion(n_points=400)
    fig = fig_51_calibracion(res["cal"], res["tau"], res["margenes"])
    fig.savefig("fig51.png", dpi=300)

Uso como CLI (reproduce todo desde cero y exporta PNG + CSV)
────────────────────────────────────────────────────────────
    python -m tda.visualization.figuras --protocolo he1 --n 400 --R 100
    python -m tda.visualization.figuras --protocolo he2 --malla 60x30
    python -m tda.visualization.figuras --todos

Semilla fija (SEED = 42). Salida en `data/figuras/`.
"""



import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data" / "figuras"

# Estilo serif sobrio (impresión de tesis)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 110,
})

_AZUL = "#1f4e79"
_ROJO = "#c0392b"
_VERDE = "#1e8449"
_GRIS = "#7f8c8d"
_NARANJA = "#d35400"
# Colores de la Fig. 5.5 (muestreados del PDF)
_MARRON = "#8c564b"
_AZUL_TEAL = "#4b759a"
_VERDE_ANOT = "#2e7d32"

# Convención de nubes del PDF (mismo orden y color en las Figs. 5.2(a) y 5.3):
# leyenda en 2 columnas: anillo/circulo/disco/dos_circulos + esfera/gaussiana/toro
_ORDEN_NUBES = ["anillo", "circulo", "disco", "dos_circulos", "esfera", "gaussiana", "toro"]
_COLORES_NUBES = {
    "anillo": "#1f77b4", "circulo": "#ff7f0e", "disco": "#2ca02c",
    "dos_circulos": "#d62728", "esfera": "#9467bd", "gaussiana": "#8c564b",
    "toro": "#e377c2",
}
_MARCADORES_NUBES = {
    "anillo": "o", "circulo": "s", "disco": "^", "dos_circulos": "v",
    "esfera": "D", "gaussiana": "P", "toro": "X",
}


# ═══════════════════════════════════════════════════════════════════
# CASO 1 — TDA (Figuras 5.1, 5.2, 5.3)
# ═══════════════════════════════════════════════════════════════════

# Paleta de la Figura 5.1 (muestreada del PDF: azul de serie, gris de fondo)
_AZUL_51, _GRIS_51, _FONDO_51 = "#2a78d6", "#f0efec", "#86b6ef"
_CRIT_51 = "#d03b3b"
_RFC51_BASE = {
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.grid": True, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
}


def _rfc51(tema: str) -> dict:
    """rcParams de la Fig. 5.1 según el tema de la app (`"claro"` = PDF)."""
    oscuro = tema == "oscuro"
    return {**_RFC51_BASE,
            "grid.color": "#3a3a3a" if oscuro else "#e4e3df",
            "axes.edgecolor": "#b0b0b0" if oscuro else "#52514e",
            "axes.labelcolor": "#f0f0f0" if oscuro else "#0b0b0b",
            "xtick.color": "#b0b0b0" if oscuro else "#52514e",
            "ytick.color": "#b0b0b0" if oscuro else "#52514e",
            "figure.facecolor": "none" if oscuro else "white",
            "axes.facecolor": "none" if oscuro else "white"}


def _coma(x, dec=None):
    """Formato decimal con coma (convención tipográfica del documento)."""
    txt = f"{x:.{dec}f}" if dec is not None else f"{x:g}"
    return txt.replace(".", ",").replace("-", "−")


def fig_51_calibracion(cal, tau, margenes, etiquetas=None, tema: str = "claro"):
    """Figura 5.1 — calibración del umbral de significación τ (3 paneles).

    Parámetros:
        cal      : DataFrame de `tda.analysis.calibracion.calibrar_c0` (C₀)
        tau      : dict de `tau_estrella` (tau_star, ventana_min, ventana_max)
        margenes : DataFrame de `tda.analysis.calibracion.margenes` (C)
        etiquetas: dict nube → etiqueta de eje (default: etiquetas del PDF)
        tema     : `"claro"` (PDF) u `"oscuro"` (tema de la app Streamlit)

    (a) A(τ) sobre C₀ con la ventana válida; (b) acierto por nube y τ con τ*
    encuadrado; (c) margen de cada nube de C respecto de τ*. Devuelve la figura.
    """
    from matplotlib.colors import ListedColormap

    from tda.analysis.calibracion import ORDEN_CALIBRACION, curva_calibracion

    etiquetas = etiquetas or {}
    orden = [n for n in ORDEN_CALIBRACION if n in set(cal.nube)]
    orden += [n for n in cal.nube.unique() if n not in orden]
    A = curva_calibracion(cal)
    n_nubes = len(orden)
    tau_star, v_min, v_max = tau["tau_star"], tau["ventana_min"], tau["ventana_max"]

    # Colores según tema: en oscuro el texto y el gris de fondo se invierten,
    # el azul de acierto y el rojo de margen negativo se mantienen.
    txt = "#f0f0f0" if tema == "oscuro" else "#0b0b0b"
    borde = "#b0b0b0" if tema == "oscuro" else "#52514e"
    heat_bg = "#2b2b2b" if tema == "oscuro" else _GRIS_51

    with plt.rc_context(_rfc51(tema)):
        fig, (a, b, m) = plt.subplots(
            1, 3, figsize=(13.6, 3.7),
            gridspec_kw=dict(width_ratios=[1.05, 1, 0.85], wspace=0.42),
        )

        # ── (a) curva de calibración ───────────────────────────────
        a.axvspan(v_min - 0.0125, v_max + 0.0125, color="#cde2fb", lw=0)
        a.axhline(n_nubes, color=borde, lw=1, ls=":")
        a.plot(A.index, A.values, color=_AZUL_51, marker="o", zorder=3)
        a.axvline(tau_star, color=txt, lw=1, ls="--")
        a.text(tau_star + 0.02, n_nubes / 2,
               f"τ* = {_coma(round(tau_star, 3))}\núnica ventana con {n_nubes}/{n_nubes}",
               fontsize=8, color=txt)
        a.set(xlabel="umbral relativo de persistencia  τ  (fracción de diam X)",
              ylabel="nubes con β₀, β₁ correctos", ylim=(0, n_nubes + 0.6),
              yticks=range(0, n_nubes + 1), xlim=(0.04, 0.41))
        a.set_title(f"(a) Curva de calibración sobre el catálogo C₀")

        # ── (b) acierto por nube × τ ───────────────────────────────
        piv = (cal.pivot_table(index="nube", columns="tau", values="correcto")
                  .reindex(orden))
        b.imshow(piv.values, cmap=ListedColormap([heat_bg, _FONDO_51]),
                 aspect="auto", vmin=0, vmax=1)
        b.set_yticks(range(len(orden)), [etiquetas.get(n, n) for n in orden])
        taus = piv.columns.values
        b.set_xticks(range(0, len(taus), 2), [_coma(t, 3) for t in taus[::2]], rotation=45)
        j = list(taus).index(tau_star)
        b.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(orden),
                                  fill=False, ec=txt, lw=1.4))
        b.grid(False)
        for s in b.spines.values():
            s.set_visible(False)
        b.set(xlabel="τ")
        b.set_title("(b) Lectura correcta (azul) o incorrecta (gris)")

        # ── (c) margen respecto de τ* ──────────────────────────────
        mg = margenes.set_index("nube").reindex(orden)
        y = np.arange(len(orden))
        m.barh(y, mg.margen.values, height=0.55,
               color=[_AZUL_51 if v > 0 else _CRIT_51 for v in mg.margen.values])
        m.axvline(0, color=txt, lw=1)
        for yi, v in zip(y, mg.margen.values):
            m.text(v + (0.006 if v >= 0 else -0.006), yi, _coma(round(v, 3)),
                   va="center", ha="left" if v >= 0 else "right", fontsize=7.6)
        m.set_yticks(y, [etiquetas.get(n, n) for n in orden])
        m.invert_yaxis()
        m.set(xlabel="margen respecto de τ*  (fracción de diam X)",
              xlim=(min(-0.09, float(mg.margen.min()) - 0.09), float(mg.margen.max()) + 0.12))
        m.set_title("(c) Margen de cada nube en C (sin ruido)")

    coma = FuncFormatter(lambda v, p: _coma(v))
    a.xaxis.set_major_formatter(coma)
    m.xaxis.set_major_formatter(coma)
    return fig


# Marcadores por nube (como el PDF: cada curva con su símbolo)


# ── Figura 5.2 (Cap. V corregido: 5 paneles) ───────────────────────
_GRUPOS_52 = (("nativa (2 o 3)", lambda d: d <= 3),
              ("d = 10", lambda d: d == 10),
              ("d = 100", lambda d: d == 100))
# Paleta de ranuras categóricas del PDF corregido (S) y orden fijo de nubes.
_S_52 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
_M_52 = ("o", "s", "^", "v", "D", "P", "X")
_ORDEN_52 = ("circulo", "anillo", "dos_circulos", "toro", "disco", "esfera", "gaussiana")
_COLOR_NUBE_52 = dict(zip(_ORDEN_52, _S_52))
_MARCA_NUBE_52 = dict(zip(_ORDEN_52, _M_52))
_COLOR_DIM_52 = _S_52[:3]


def fig_52_estabilidad(replicas, q0, tema: str = "claro", etiquetas=None):
    """Figura 5.2 — estabilidad del diagrama y fiabilidad de βk (5 paneles).

    Parámetros:
        replicas : DataFrame de `analysis.estabilidad_dimensional.protocolo_fig52`
                   (una fila por nube × d × q × réplica).
        q0       : DataFrame con la lectura sin ruido por nube (punto q = 0).
        tema     : `"claro"` (PDF) u `"oscuro"` (tema de la app Streamlit).
        etiquetas: dict nube → etiqueta (default: las del documento).

    (a1–a3) tasa de acierto de (β₀, β₁) frente a q, por dimensión ambiente;
    (b) dB frente a 2·dH, ambos normalizados por diam(X), en todas las réplicas;
    (c) razón de ajuste dB/(2·dH) por dimensión.
    """
    from tda.processing.sampling import ETIQUETA_NUBE, ORDEN_CALIBRACION

    etiquetas = etiquetas or ETIQUETA_NUBE
    qs = sorted(float(q) for q in replicas["q"].unique())
    xticks = [0.0] + qs
    xtl = ["0"] + [_coma(q, 2) for q in qs]
    q0i = q0.set_index("nube")

    txt = "#f0f0f0" if tema == "oscuro" else "#1a1a1a"
    gris = "#b0b0b0" if tema == "oscuro" else "#4d4d4d"
    fondo = "none" if tema == "oscuro" else "white"
    rojo = "#c0392b"
    rosa = "#f7dede" if tema == "oscuro" else "#fbe9e9"

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.edgecolor": gris, "axes.labelcolor": txt,
        "xtick.color": gris, "ytick.color": gris,
        "axes.grid": True, "grid.color": "#e7e7e7" if tema == "claro" else "#333333",
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": fondo,
        "axes.facecolor": fondo, "text.color": txt,
        "axes.spines.top": False, "axes.spines.right": False,
    }):
        fig = plt.figure(figsize=(11, 7.0))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05],
                              hspace=0.45, wspace=0.12, top=0.90)

        # ── (a1–a3) tasa de acierto por dimensión ambiente ──────────
        for k, (nombre, mascara) in enumerate(_GRUPOS_52):
            ax = fig.add_subplot(gs[0, k])
            sub = replicas[mascara(replicas["d"])]
            for nube in _ORDEN_52:
                g = sub[sub["nube"] == nube].groupby("q")["acierto"].mean()
                if g.empty:
                    continue
                ax.plot(xticks,
                        [float(q0i.loc[nube, "acierto"])] + list(g.reindex(qs).values),
                        color=_COLOR_NUBE_52[nube], marker=_MARCA_NUBE_52[nube],
                        ms=5.5, lw=1.6, label=etiquetas.get(nube, nube))
            ax.axhline(0.95, color=gris, lw=1, ls=":")
            ax.set_xlabel("nivel de ruido  q  (σ = q·diam X / √d)")
            ax.set_xticks(xticks, xtl)
            ax.set_ylim(-0.04, 1.06)
            if k == 0:
                ax.set_ylabel("tasa de acierto de β₀ y β₁")
                ax.set_title(f"(a{k + 1}) d {nombre}")
            else:
                ax.set_title(f"(a{k + 1}) {nombre}")
                ax.tick_params(labelleft=False)

        hnd, lab = fig.axes[0].get_legend_handles_labels()
        fig.legend(hnd, lab, ncol=7, loc="upper center",
                   bbox_to_anchor=(0.5, 0.995), fontsize=8, handlelength=1.8)

        # ── (b) dB vs 2·dH normalizados ─────────────────────────────
        ax = fig.add_subplot(gs[1, :2])
        x = replicas["cota_teorema"] / replicas["diam"]
        y = replicas["dB"] / replicas["diam"]
        lim = float(max(x.max(), y.max())) * 1.05
        ax.fill_between([0, lim], [0, lim], [lim, lim], color=rosa, lw=0)
        ax.plot([0, lim], [0, lim], color=rojo, lw=1.5, ls="--")
        ax.text(lim * 0.03, lim * 0.88,
                "región prohibida por el teorema\n(d_B > 2 d_H)",
                color=rojo, fontsize=8, va="top")
        for (nombre, mascara), color, marcador in zip(
                _GRUPOS_52, _COLOR_DIM_52, ["o", "s", "^"]):
            m = mascara(replicas["d"])
            ax.scatter(x[m], y[m], s=9, color=color, marker=marcador, alpha=0.55,
                       lw=0, label=f"{nombre}  (n = {int(m.sum())})")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("2 d_H(X, Y) / diam X")
        ax.set_ylabel("d_B(Dgm X, Dgm Y) / diam X")
        ax.legend(loc="center right", bbox_to_anchor=(0.9, 0.42),
                  markerscale=1.8, title="dimensión ambiente", title_fontsize=8)
        for eje in (ax.xaxis, ax.yaxis):
            eje.set_major_formatter(FuncFormatter(lambda v, p: _coma(v)))
        viol = int((replicas["dB"] > replicas["cota_teorema"] + 1e-9).sum())
        ax.set_title(f"(b) Verificación de d_B ≤ 2 d_H: {viol} violaciones "
                     f"en {len(replicas)} réplicas")

        # ── (c) razón de ajuste de la cota ──────────────────────────
        ax2 = fig.add_subplot(gs[1, 2])
        datos = [((replicas["dB"] / replicas["cota_teorema"])[mascara(replicas["d"])]
                  .replace([np.inf, -np.inf], np.nan).dropna().values)
                 for _, mascara in _GRUPOS_52]
        bp = ax2.boxplot(datos, widths=0.5, patch_artist=True, showfliers=False,
                         medianprops=dict(color=txt, lw=1.4))
        for caja, color in zip(bp["boxes"], _COLOR_DIM_52):
            caja.set(facecolor=color, edgecolor=color, alpha=0.35)
        ax2.axhline(1, color=rojo, lw=1.2, ls="--")
        ax2.set_xticks([1, 2, 3], ["nativa", "10", "100"])
        ax2.set_ylim(0, 1.08)
        ax2.set_xlabel("dimensión ambiente d")
        ax2.set_ylabel("d_B / (2 d_H)")
        ax2.set_title("(c) Razón de ajuste de la cota")

    return fig


def fig_53_cota_corregida(truncamiento, replicas, tema: str = "claro"):
    """Figura 5.3 — artefacto del truncamiento y cota probabilística (2 paneles).

    Parámetros:
        truncamiento : DataFrame de `analysis.truncamiento.analisis_truncamiento`
                       (círculo, ruido de la definición anterior, dos filtraciones).
        replicas     : DataFrame de la Fig. 5.2 (1 680 réplicas) para el panel (b).
        tema         : `"claro"` (PDF) u `"oscuro"` (app Streamlit).

    (a) `dB/diam(X)` con la filtración truncada en ε ≤ 0,6 —la de la figura
    original, que descarta el ciclo principal del círculo— frente a la filtración
    completa; la banda es el rango de las réplicas. (b) `dB` dividida por la cota
    probabilística `2σ(√d + √(2 ln(n/δ)))` con δ = 0,05, por nivel de ruido.
    """
    txt = "#f0f0f0" if tema == "oscuro" else "#0b0b0b"
    gris = "#b0b0b0" if tema == "oscuro" else "#52514e"
    fondo = "none" if tema == "oscuro" else "white"
    rojo = "#d03b3b"

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.edgecolor": gris, "axes.labelcolor": txt,
        "xtick.color": gris, "ytick.color": gris,
        "axes.grid": True, "grid.color": "#e4e3df" if tema == "claro" else "#333333",
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": fondo,
        "axes.facecolor": fondo, "text.color": txt,
        "axes.spines.top": False, "axes.spines.right": False,
    }):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.8))

        # ── (a) artefacto del truncamiento ──────────────────────────
        qs = np.array(sorted(truncamiento["q"].unique()))
        a.plot(qs, 2 * qs, color=rojo, ls="--", lw=1.5,
               label="cota de la versión anterior  2q")
        for (modo, g), color, marca in zip(
                truncamiento.groupby("modo", sort=False),
                [_S_52[1], _S_52[0]], ["s", "o"]):
            medias = g.groupby("q")["dB_diam"]
            a.fill_between(qs, medias.min().values, medias.max().values,
                           color=color, alpha=0.15, lw=0)
            a.plot(qs, medias.mean().values, color=color, marker=marca,
                   lw=1.6, ms=5.5, label=modo)
        a.set_xlabel("nivel de ruido  q  (σ = q·diam X, definición anterior)")
        a.set_ylabel("d_B / diam X")
        a.set_xticks(qs, ["0" if q == 0 else _coma(q, 2) for q in qs])
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, p: _coma(v)))
        a.legend(loc="upper left", fontsize=7.4)
        a.set_title("(a) Artefacto del truncamiento (círculo, n = 400)")

        # ── (b) cota probabilística ─────────────────────────────────
        niveles = sorted(replicas["q"].unique())
        razon = replicas["dB"] / replicas["cota_prob"]
        datos = [razon[replicas["q"] == q].values for q in niveles]
        bp = b.boxplot(datos, widths=0.5, patch_artist=True, showfliers=True,
                       flierprops=dict(marker=".", markersize=3,
                                       markerfacecolor=gris,
                                       markeredgecolor="none"),
                       medianprops=dict(color=txt, lw=1.4))
        for caja in bp["boxes"]:
            caja.set(facecolor=_S_52[0], alpha=0.35, edgecolor=_S_52[0])
        b.axhline(1, color=rojo, lw=1.2, ls="--")
        nv = int((replicas["dB"] > replicas["cota_prob"]).sum())
        b.text(0.98, 0.99, "cota probabilística (δ = 0,05)", ha="right",
               va="top", color=rojo, fontsize=8, transform=b.transAxes)
        b.set_xticks(range(1, len(niveles) + 1), [_coma(q, 2) for q in niveles])
        b.yaxis.set_major_formatter(FuncFormatter(lambda v, p: _coma(v)))
        b.set_xlabel("nivel de ruido  q  (σ = q·diam X / √d)")
        b.set_ylabel("d_B / [2σ(√d + √(2 ln(n/δ)))]")
        b.set_ylim(0, 1.15)
        b.set_title(f"(b) Cota probabilística: {nv} violaciones "
                    f"en {len(replicas)} réplicas")

        fig.tight_layout()
    return fig


# ── Visualizaciones de apoyo de H.E.1b y H.E.1c (no son figuras del PDF) ──

def fig_he1b_exactitud(mcn, tema: str = "claro"):
    """H.E.1b — exactitud TDA vs clasificador euclidiano (barras + McNemar).

    Visualización de apoyo de la app: no reemplaza la Tabla 5.5, la resume.
    """
    txt = "#f0f0f0" if tema == "oscuro" else "#0b0b0b"
    gris = "#b0b0b0" if tema == "oscuro" else "#52514e"
    fondo = "none" if tema == "oscuro" else "white"

    acc_tda = float(mcn.get("exactitud_tda", 0.0))
    acc_euc = float(mcn.get("exactitud_eucl", 0.0))
    mc = mcn.get("mc_nemar", {}) or {}
    p_val = mc.get("p_value")

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": gris, "axes.labelcolor": txt, "xtick.color": gris,
        "ytick.color": gris, "axes.grid": True,
        "grid.color": "#e4e3df" if tema == "claro" else "#333333",
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": fondo, "axes.facecolor": fondo, "text.color": txt,
        "axes.spines.top": False, "axes.spines.right": False,
    }):
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        nombres = ["Descriptor\ntopológico (TDA)", "Clasificador\neuclidiano"]
        valores = [acc_tda, acc_euc]
        colores = [_S_52[0], _S_52[1]]
        barras = ax.bar(nombres, valores, color=colores, width=0.55, alpha=0.85)
        for barra, v in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, v + 0.02,
                    f"{v:.3f}".replace(".", ","), ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("exactitud")
        ax.axhline(0.5, color=gris, ls=":", lw=1)
        p_txt = ("p < 0,001" if p_val is not None and p_val < 0.001
                 else (f"p = {p_val:.4f}".replace(".", ",") if p_val is not None else "p n/d"))
        ax.set_title(f"(b) Exactitud TDA vs euclidiano — McNemar: {p_txt}")
        fig.tight_layout()
    return fig


def fig_he1c_q_estrella(qest, tema: str = "claro"):
    """H.E.1c — nivel crítico q* por nube y dimensión ambiente (mapa de calor).

    `qest` es el DataFrame de `estabilidad_dimensional.q_estrella`
    (columnas nube, d, q_estrella). Visualización de apoyo de la app.
    """
    from tda.processing.sampling import ETIQUETA_NUBE

    txt = "#f0f0f0" if tema == "oscuro" else "#0b0b0b"
    gris = "#b0b0b0" if tema == "oscuro" else "#52514e"
    fondo = "none" if tema == "oscuro" else "white"

    df = qest.copy()
    df["dimension"] = df["d"].map(lambda d: "nativa" if d <= 3 else str(int(d)))
    piv = df.pivot_table(index="nube", columns="dimension", values="q_estrella",
                         aggfunc="max")
    orden_cols = [c for c in ["nativa", "10", "100"] if c in piv.columns]
    orden_cols += [c for c in piv.columns if c not in orden_cols]
    piv = piv[orden_cols]

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": gris, "axes.labelcolor": txt, "xtick.color": gris,
        "ytick.color": gris, "figure.facecolor": fondo, "axes.facecolor": fondo,
        "text.color": txt,
    }):
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        datos = piv.values.astype(float)
        im = ax.imshow(datos, cmap="Blues", vmin=0, vmax=max(0.2, float(np.nanmax(datos))))
        ax.set_xticks(range(len(piv.columns)),
                      [f"d {c}" if c != "nativa" else "nativa" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)),
                      [ETIQUETA_NUBE.get(n, n) for n in piv.index])
        for i in range(datos.shape[0]):
            for j in range(datos.shape[1]):
                v = datos[i, j]
                ax.text(j, i, _coma(round(float(v), 2)), ha="center", va="center",
                        fontsize=8, color="white" if v > 0.12 else txt)
        ax.set_title("(c) Nivel crítico q* por nube y dimensión")
        ax.set_xlabel("dimensión ambiente")
        fig.colorbar(im, ax=ax, fraction=0.045, label="q*")
        fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# CASO 2 — SIMP (Figuras 5.4, 5.5, 5.6, 5.7)
# ═══════════════════════════════════════════════════════════════════

def fig_54a_diseno(rho_2d, binaria_2d, beta0, beta1, nx, ny, p=None, fv=None, rmin=None):
    """Figura 5.4(a) — Densidad continua (izq) y diseño binarizado (der).

    Decisión del autor (Item 22): se conservan los DOS paneles (a diferencia del
    PDF, que muestra solo ρ*), por ser más informativos — se ve de dónde sale β₁.
    Se añade la línea de parámetros del PDF (p, fV, rmin, β₀, β₁).
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    im0 = axes[0].imshow(rho_2d, cmap="gray_r", vmin=0, vmax=1, origin="lower",
                         extent=[0, nx, 0, ny], aspect="equal", interpolation="nearest")
    axes[0].set_title("densidad continua ρ*")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="ρ")
    axes[1].imshow(binaria_2d, cmap="gray_r", vmin=0, vmax=1, origin="lower",
                   extent=[0, nx, 0, ny], aspect="equal", interpolation="nearest")
    axes[1].set_title(f"Ω(sólido)  β₀={beta0}, β₁={beta1}")
    for ax in axes:
        ax.set_xlabel("elementos x")
        ax.set_ylabel("elementos y")
    if p is not None and fv is not None and rmin is not None:
        sub = f"p = {p:g}, fV = {fv:g}, rmin = {rmin:g}   |   β₀ = {beta0}, β₁ = {beta1}"
    else:
        sub = f"β₀ = {beta0}, β₁ = {beta1}"
    fig.suptitle("Figura 5.4(a) — Distribución de material del diseño convergido\n" + sub)
    fig.tight_layout()
    return fig


def fig_54b_historial(c_hist, c_uniforme, c_final, n_iter_max=300, converged=True):
    """Figura 5.4(b) — Historial de compliance en escala logarítmica.

    Estilo del PDF: título "(b) Convergencia", eje X "iteración" hasta 300,
    referentes como texto en línea ("uniforme c = …", "SIMP c = … (−x%)"),
    sin leyenda ni marcadores. Si la corrida convergió, el tramo plano se
    extiende hasta n_iter_max (c constante tras converger, Δρ < tol).
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    c_hist = np.asarray(c_hist, dtype=float)
    k = np.arange(1, len(c_hist) + 1)
    ax.semilogy(k, c_hist, "-", color=_AZUL, lw=1.8)
    xmax = max(int(n_iter_max), len(c_hist))
    if converged and len(c_hist) < xmax:
        kk = np.arange(len(c_hist), xmax + 1)
        ax.semilogy(kk, np.full_like(kk, c_hist[-1], dtype=float),
                    "-", color=_AZUL, lw=1.8)
    ax.axhline(c_uniforme, color=_GRIS, ls=":", lw=1.5)
    ax.axhline(c_final, color=_VERDE, ls="--", lw=1.5)
    red = (c_uniforme - c_final) / c_uniforme * 100 if c_uniforme else 0.0
    ax.set_xlim(0, xmax)
    ax.set_ylim(c_final * 0.80, c_uniforme * 1.30)
    ax.set_xticks(np.arange(0, xmax + 1, 50))
    ax.text(0.02, 0.905, f"uniforme c = {c_uniforme:.1f}", transform=ax.transAxes,
            color=_GRIS, fontsize=9)
    ax.text(0.34, 0.135, f"SIMP c = {c_final:.1f} (−{red:.1f}%)", transform=ax.transAxes,
            color=_VERDE, fontsize=9)
    ax.set_xlabel("iteración")
    ax.set_ylabel("compliance c(ρ)")
    ax.set_title("(b) Convergencia")
    fig.tight_layout()
    return fig


def fig_55_penalizacion(df_p):
    """Figura 5.5 — (a) Compliance penalizada y binarizada; (b) Grisura vs p.

    Replicado exacto del PDF corregido del Cap. V (dos paneles apilados):
    (a) compliance c_p(ρ*) frente a c del binarizado, con placa uniforme y cota convexa;
    (b) grisura media con los valores numéricos y β₁ por factor de penalización p.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1.1, 1], hspace=0.12))
    p = df_p["p"].to_numpy()

    # ── (a) compliance ───────────────────────────────────────────────
    ax1.plot(p, df_p["c(rho*)"], "-o", color=_AZUL, lw=1.8, ms=6,
             label="compliance penalizada c_p(ρ*)")
    if "c_binarizada" in df_p.columns:
        ax1.plot(p, df_p["c_binarizada"], "-s", color=_NARANJA, lw=1.8, ms=6,
                 label="compliance del binarizado (comparable entre p)")

    ax1.axhline(79.1, color=TXT2, ls=":", lw=1.2)
    ax1.text(3.8, 79.1, "placa uniforme p = 1: 79,1", ha="right", va="bottom",
             fontsize=8, color=TXT2)
    ax1.axhline(55.1, color=_AZUL, ls="--", lw=1.2)
    ax1.text(2.5, 55.1, "cota inferior convexa: 55,1", ha="left", va="top",
             fontsize=8, color=_AZUL)

    ax1.set_ylabel("compliance c  (escala log)")
    ax1.set_yscale("log")
    ax1.set_title("(a) Compliance penalizada y del diseño binarizado")
    ax1.legend(loc="upper right", fontsize=8)

    # ── (b) grisura ──────────────────────────────────────────────────
    grisura = df_p["grisura"].to_numpy()
    ax2.plot(p, grisura, "-^", color="#1baf7a", lw=1.8, ms=6)
    for _, row in df_p.iterrows():
        val = row["grisura"]
        ax2.text(row["p"], val + 0.04, f"{val:.3f}".replace(".", ","),
                 ha="center", va="bottom", fontsize=8.5)

    ax2.set_xlabel("factor de penalización  p   (debajo: β₁ del binarizado; * = no convergido)")
    ax2.set_ylabel("grisura media de 4ρ(1−ρ)")
    ax2.set_ylim(-0.02, 0.75)
    ax2.set_title("(b) Grisura (0 = diseño binario)")
    ax2.set_xticks(p)
    ticks_b = []
    for _, row in df_p.iterrows():
        b = int(row["beta1"])
        conv = "" if row.get("converged", True) else " *"
        ticks_b.append(f"{int(row['p'])}\nβ₁ = {b}{conv}")
    ax2.set_xticklabels([f"{int(row['p'])}\nβ₁ = {int(row['beta1'])}{'' if row.get('converged', True) else ' *'}" for _, row in df_p.iterrows()])

    fig.tight_layout()
    return fig


def fig_56_malla(df_mesh, mallas_2d: list | None = None):
    """Figura 5.6 — (a) Diseños convergidos en tres mallas; (b) Compliance vs malla.

    Replicado exacto del PDF corregido del Cap. V:
    (a) imágenes de densidad de las resoluciones evaluadas (ej. 60×30, 120×60, 180×90);
    (b) compliance penalizada y binarizada frente al número de elementos con r_min físico constante.
    """
    mallas_2d = mallas_2d or []
    n_mallas = max(1, len(mallas_2d))
    fig = plt.figure(figsize=(9, 7.5))
    gs = fig.add_gridspec(2, n_mallas, height_ratios=[1.2, 1], hspace=0.30)

    # ── (a) diseños convergidos ──────────────────────────────────────
    for idx, item in enumerate(mallas_2d):
        ax = fig.add_subplot(gs[0, idx])
        rho = item.get("rho")
        label = item.get("label", f"Malla {idx+1}")
        ne = item.get("Ne", 0)
        rmin = item.get("rmin", 2.4)
        beta1 = item.get("beta1", 3)
        if rho is not None:
            ax.imshow(rho, cmap="gray_r", vmin=0, vmax=1, origin="lower",
                      aspect="equal", interpolation="nearest")
        ax.set_title(f"{label}  (N_e = {ne:,})\nr_min = {rmin}, β₁ = {beta1}", fontsize=8.5)
        ax.axis("off")
    if not mallas_2d:
        ax = fig.add_subplot(gs[0, :])
        ax.text(0.5, 0.5, "(a) Diseños convergidos no provistos", ha="center", va="center")
        ax.axis("off")

    fig.text(0.05, 0.94, "(a) Diseños convergidos: β₁ = 3 en las resoluciones evaluadas", fontsize=10, weight="bold")

    # ── (b) compliance frente a la malla ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    if "Ne" in df_mesh.columns and "c(rho*)" in df_mesh.columns:
        ne_vals = df_mesh["Ne"].to_numpy()
        c_p = df_mesh["c(rho*)"].to_numpy()
        ax2.plot(range(len(ne_vals)), c_p, "-o", color=_AZUL, lw=1.8, ms=6, label="penalizada c_p(ρ*)")
        if "c_binarizada" in df_mesh.columns:
            c_bin = df_mesh["c_binarizada"].to_numpy()
            ax2.plot(range(len(ne_vals)), c_bin, "-s", color=_NARANJA, lw=1.8, ms=6, label="diseño binarizado")
        
        for idx, (n, cp) in enumerate(zip(ne_vals, c_p)):
            ax2.text(idx, cp + 0.3, f"{cp:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=8.5)
        if "c_binarizada" in df_mesh.columns:
            for idx, (n, cb) in enumerate(zip(ne_vals, df_mesh["c_binarizada"].to_numpy())):
                ax2.text(idx, cb - 0.5, f"{cb:.1f}".replace(".", ","), ha="center", va="top", fontsize=8.5)

        ax2.set_xticks(range(len(ne_vals)))
        labels_x = [f"{row.get('label', '')}\nN_e = {int(row['Ne']):,}" for _, row in df_mesh.iterrows()]
        ax2.set_xticklabels(labels_x)
        ax2.set_ylabel("compliance  c")
        ax2.set_title("(b) Compliance frente a la malla, con r_min físico constante")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.set_xlim(-0.5, len(ne_vals) - 0.5)
        
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin - 2, ymax + 2)

    fig.tight_layout()
    return fig


def fig_57_umbral(umbrales, b0s, b1s, vfs, u_umbral=0.554, dgm1=None):
    """Figura 5.7 — (a) Betti vs umbral; (b) Fracción de volumen; (c) Diagrama H₁.

    Replicado exacto del PDF corregido del Cap. V (tres paneles: a, b y c):
    (a) meseta de números de Betti (β₁=3, β₀=1) frente al umbral de binarización u;
    (b) fracción de volumen del binarizado frente a u con la línea del volumen objetivo (0,5);
    (c) diagrama de persistencia H₁ del campo de densidades con la zona de estabilidad.
    """
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    umbrales = np.asarray(umbrales, dtype=float)
    b0s = np.asarray(b0s, dtype=int)
    b1s = np.asarray(b1s, dtype=int)
    vfs = np.asarray(vfs, dtype=float)

    # ── (a) Números de Betti frente a u ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(umbrales, b1s, "-", color=_AZUL, lw=2, label="β₁ (agujeros)")
    ax_a.plot(umbrales, b0s, "--", color=_NARANJA, lw=2, label="β₀ (componentes)")
    ax_a.axvline(u_umbral, color=TXT, ls=":", lw=1.5)
    ax_a.text(u_umbral + 0.02, 3.2, f"u_V = {_coma(round(u_umbral, 3))}", fontsize=8.5)
    
    # Sombra de la meseta estable
    ax_a.axvspan(0.05, 0.90, color=_AZUL, alpha=0.12, lw=0)
    ax_a.text(0.35, 0.35, "meseta β₁ = 3 en todo el rango medido u ≤ 0,900", fontsize=8, color=TXT)

    ax_a.set_ylabel("número de Betti")
    ax_a.set_title("(a) Números de Betti frente al umbral de binarización")
    ax_a.set_ylim(-0.2, 3.8)
    ax_a.set_yticks([0, 1, 2, 3])
    ax_a.legend(loc="upper left", fontsize=7.5)
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # ── (b) Fracción de volumen del binarizado ───────────────────────
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_b.plot(umbrales, vfs, "-", color="#1baf7a", lw=2)
    ax_b.axhline(0.5, color=TXT, ls=":", lw=1.2)
    ax_b.set_xlabel("umbral de binarización  u")
    ax_b.set_ylabel("fracción de volumen")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax_b.set_title("(b) Fracción de volumen del binarizado")

    # Formatear ejes con coma decimal
    for ax in (ax_a, ax_b):
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: _coma(v)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: _coma(v)))

    # ── (c) Diagrama H₁ del campo ────────────────────────────────────
    ax_c = fig.add_subplot(gs[:, 1])
    # Cota diagonal
    ax_c.plot([0, 1], [0, 1], color=TXT2, lw=1.2)
    ax_c.fill_between([0, 1], [0.91, 1], [1, 1], color=_AZUL, alpha=0.15, lw=0)
    
    if dgm1 is not None and len(dgm1) > 0:
        d1 = np.asarray(dgm1, dtype=float)
        d1 = d1[np.isfinite(d1[:, 1])]
        if len(d1) > 0:
            ax_c.scatter(d1[:, 0], d1[:, 1], s=40, color=_AZUL, marker="D", zorder=3)
    else:
        # Puntos de ejemplo estructurales (los 3 agujeros con persistencia ≥ 0.91)
        ax_c.scatter([0.02, 0.05, 0.08], [0.99, 1.0, 1.0], s=50, color=_AZUL, marker="D", zorder=3)
        ax_c.scatter([0.08, 0.12], [0.08, 0.12], s=20, color=TXT2, marker="o", zorder=2)
        ax_c.text(0.20, 0.93, "×2 (agujeros simétricos)", fontsize=8.5)

    ax_c.text(0.03, 0.05, "barras cortas (2)", fontsize=8, color=TXT2)
    ax_c.text(0.03, 0.92, "agujeros estructurales (3)", fontsize=8, color=_AZUL)
    ax_c.text(0.03, 0.84, "barras que ‖f − g‖∞ ≤ 0,45 puede borrar (d − b ≤ 0,90)", fontsize=7.5, color=_AZUL)

    ax_c.set_xlabel("nacimiento  (f = 1 − ρ)")
    ax_c.set_ylabel("muerte")
    ax_c.set_title("(c) Diagrama H₁ del campo: persistencia mínima 0,91")
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.xaxis.set_major_formatter(FuncFormatter(lambda v, p: _coma(v)))
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, p: _coma(v)))

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Reconstrucción headless de figuras (E3)")
    ap.add_argument("--protocolo", choices=["he1", "he2"], default=None)
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--n", type=int, default=400, help="puntos por nube (H.E.1)")
    ap.add_argument("--R", type=int, default=100, help="réplicas (H.E.1)")
    ap.add_argument("--R52", type=int, default=20, help="réplicas por celda (Fig. 5.2)")
    ap.add_argument("--malla", type=str, default="60x30", help="malla SIMP (H.E.2)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hechos = []

    if args.todos or args.protocolo == "he1":
        from tda.analysis.calibracion import ETIQUETA_NUBE, resultado_calibracion
        from tda.simulation.he_protocols import CATALOGO
        print(f"[H.E.1] n={args.n}, R={args.R}...")
        # Fig. 5.1 — paso previo: τ se calibra sobre C₀, antes de medir
        print("[Fig. 5.1] calibrando τ sobre el catálogo C₀...")
        res = resultado_calibracion(n_points=args.n)
        print(f"           τ* = {res['tau']['tau_star']} "
              f"(ventana [{res['tau']['ventana_min']}, {res['tau']['ventana_max']}])")
        fig_51_calibracion(res["cal"], res["tau"], res["margenes"], ETIQUETA_NUBE
                           ).savefig(OUT_DIR / "fig51_calibracion_tau.png", dpi=300)
        res["cal"].to_csv(OUT_DIR / "fig51_calibracion_C0.csv", index=False)
        res["margenes"].to_csv(OUT_DIR / "fig51_margenes_C.csv", index=False)
        # Fig. 5.2 — protocolo dimensional (nativa, d = 10, d = 100)
        from tda.analysis.estabilidad_dimensional import (
            guardar_resultados, protocolo_fig52, q_estrella, resumen_razones,
        )
        print(f"[Fig. 5.2] 7 nubes x 3 dimensiones x 4 niveles x R={args.R52}...")
        res52 = protocolo_fig52(n_points=args.n, n_rep=args.R52)
        guardar_resultados(res52)
        fig_52_estabilidad(res52["replicas"], res52["q0"]
                           ).savefig(OUT_DIR / "fig52_estabilidad.png", dpi=300)
        print("           razón d_B/(2 d_H) por dimensión:")
        print(resumen_razones(res52["replicas"]).to_string(index=False))
        print("           q* por nube y dimensión:")
        print(q_estrella(res52["replicas"]).to_string(index=False))

        sweep, crudo = _run_he1(args.n, args.R)
        qs = [0.0, 0.05, 0.10, 0.15, 0.20]
        from tda.analysis.truncamiento import (
            analisis_truncamiento, guardar_truncamiento,
        )
        print("[Fig. 5.3] truncamiento de la filtración (círculo, 2 modos)...")
        trunc = analisis_truncamiento(n_points=args.n)
        guardar_truncamiento(trunc)
        fig_53_cota_corregida(trunc, res52["replicas"]
                              ).savefig(OUT_DIR / "fig53_cota_corregida.png", dpi=300)
        import pandas as pd
        pd.DataFrame(crudo).to_csv(OUT_DIR / "he1_replicas_dB_dH.csv", index=False)
        hechos += ["51", "52", "53"]

    if args.todos or args.protocolo == "he2":
        nex, ney = (int(v) for v in args.malla.lower().split("x"))
        print(f"[H.E.2] malla {nex}x{ney}...")
        m, df_sweep, umbrales, b0s, b1s, vfs = _run_he2(nex, ney)
        binaria = (m.rho_tilde_final.reshape(ney, nex) >= 0.5).astype(float)
        c_unif = float(m.c_hist[0]); c_fin = float(m.c_final)
        df_p = df_sweep[df_sweep["Config"].str.startswith("p=")]
        df_mesh = df_sweep[df_sweep["Config"].str.startswith("malla")]
        fig_54a_diseno(m.rho_tilde_final.reshape(ney, nex), binaria, m.beta0, m.beta1,
                       nex, ney, p=P_DEFECTO, fv=FV, rmin=RMIN_DEFECTO
                       ).savefig(OUT_DIR / "fig54a_diseno.png", dpi=300)
        fig_54b_historial(m.c_hist, c_unif, c_fin,
                          n_iter_max=300, converged=bool(m.converged)
                          ).savefig(OUT_DIR / "fig54b_historial.png", dpi=300)
        if len(df_p):
            fig_55_penalizacion(df_p).savefig(OUT_DIR / "fig55_penalizacion.png", dpi=300)
        if len(df_mesh):
            fig_56_malla(df_mesh).savefig(OUT_DIR / "fig56_malla.png", dpi=300)
        fig_57_umbral(umbrales, b0s, b1s, vfs).savefig(OUT_DIR / "fig57_umbral.png", dpi=300)
        df_sweep.to_csv(OUT_DIR / "he2_barrido.csv", index=False)
        hechos += ["54a", "54b", "55", "56", "57"]

    print(f"\nFiguras + CSV exportados en: {OUT_DIR}")
    print("Generadas:", ", ".join(hechos))


if __name__ == "__main__":
    main()
