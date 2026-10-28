"""figuras.py — Reconstrucción headless de las figuras del Cap. V (entregable E3).

Genera las figuras de la tesis (5.1–5.7) SIN Streamlit, a partir de los
diccionarios de resultados de los protocolos (`tda.simulation.he_protocols`)
o de los CSV exportados. Cada figura respeta los ejes, rangos y escalas del
documento para que la comparación visual con el PDF sea directa.

Uso como API
────────────
    from tda.analysis.calibracion import resultado_calibracion
    from tda.visualization.figuras import fig_51_calibracion, fig_53_cota

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

from __future__ import annotations

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


def fig_52a_tasa_acierto(qs, sweep, catalogo, colores=None):
    """Figura 5.2(a) — Tasa de acierto de βk frente al ruido q, por nube.

    Estilo del PDF: franjas con texto dentro ("fiable"/"no fiable"), un
    marcador por nube, ticks en {0, 0.05, ..., 0.20}, sin líneas guía 95/50.
    """
    qs = np.asarray(qs)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(qs.min(), 0.05, color="green", alpha=0.10)
    ax.axvspan(0.15, qs.max(), color="red", alpha=0.10)
    ax.text(0.025, 0.93, "fiable", color=_VERDE, fontsize=9, ha="center")
    ax.text(0.175, 0.93, "no fiable", color=_ROJO, fontsize=9, ha="center")
    for shape in _ORDEN_NUBES:
        if shape not in sweep:
            continue
        ax.plot(qs, sweep[shape]["acierto"], "-",
                marker=_MARCADORES_NUBES.get(shape, "o"), ms=5,
                color=(colores or _COLORES_NUBES).get(shape, _AZUL), label=shape)
    ax.set_xlabel("nivel de ruido q = σₙ/diam(X)")
    ax.set_ylabel("tasa de acierto de βk")
    ax.set_title("(a) Colapso de la lectura de βk")
    ax.set_xlim(0.0, float(qs.max()))
    ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, ncol=2, loc="center right")
    fig.tight_layout()
    return fig


def fig_52b_cota_estabilidad(crudo):
    """Figura 5.2(b) — Cada réplica: dB vs 2·dH, con región prohibida.

    Estilo del PDF: ejes dH(X,Y) / dB(Dgm(X), Dgm(Y)), diagonal a todo el
    eje, texto "región prohibida / por el teorema" sin sombreado, leyenda
    con el n real de réplicas.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    dB = np.array([c["dB"] for c in crudo])
    dH = np.array([c["dH"] for c in crudo])
    ax.scatter(dH, dB, s=10, alpha=0.45, color=_AZUL,
               label=f"réplicas (n = {len(crudo)})")
    xmax = max(float(dH.max()), 1e-9) * 1.05
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 2.0 * xmax)
    ax.plot([0.0, xmax], [0.0, 2.0 * xmax], "--", color=_ROJO, lw=1.5,
            label="cota teórica dB = 2 dH")
    ax.text(0.62 * xmax, 0.86 * 2.0 * xmax, "región prohibida\npor el teorema",
            color=_ROJO, fontsize=9, style="italic", ha="center")
    ax.set_xlabel("dH(X,Y)")
    ax.set_ylabel("dB(Dgm(X), Dgm(Y))")
    ax.set_title("(b) Verificación empírica de la cota de estabilidad")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def fig_53_cota(qs, crudo, catalogo, forma="documento", colores=None):
    """Figura 5.3 — distancia bottleneck relativa frente a la cota.

    ``forma="documento"`` (literal del PDF): eje Y ``dB/diam(X)``, cota diagonal
    ``2q``, rango [0, ~0.45]. Con la métrica exacta, círculo y dos_círculos cruzan
    la cota — es la inconsistencia del documento (Item 11/25).

    ``forma="teorema"`` (correcta, Cohen-Steiner): eje Y ``dB/dH``, cota horizontal
    ``2``; todas las curvas quedan por debajo.

    Mismo esqueleto de datos, dos presets. Ver Correcciones, Item 26.
    """
    qs = np.asarray(qs)
    qmax = float(qs.max())

    def _val(c):
        if forma == "documento":
            return c["dB_diam"]
        return c["dB"] / c["dH"] if c["dH"] > 0 else np.nan

    if forma == "documento":
        cota_x, cota_y = [0.0, qmax], [0.0, 2.0 * qmax]
        cota_label = "cota $2q$"
        ylabel = "$d_{B}\\,/\\,\\mathrm{diam}(X)$"
        title = ("Distancia bottleneck relativa frente a la cota\n"
                 "forma del documento:  $d_{B}/\\mathrm{diam}(X) \\leq 2q$")
        yfloor = 0.42
    else:
        cota_x, cota_y = [0.0, qmax], [2.0, 2.0]
        cota_label = "cota $2$"
        ylabel = "$d_{B}\\,/\\,d_{H}$"
        title = ("Distancia bottleneck relativa frente a la cota\n"
                 "forma del teorema:  $d_{B} \\leq 2\\,d_{H}$")
        yfloor = 2.1

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for shape in _ORDEN_NUBES:
        if shape not in catalogo:
            continue
        y, e = [], []
        for q in qs:
            vals = [_val(c) for c in crudo if c["shape"] == shape and c["q"] == q]
            vals = [v for v in vals if v == v]
            y.append(float(np.mean(vals)) if vals else np.nan)
            e.append(float(np.std(vals)) if vals else 0.0)
        ax.errorbar(qs, y, yerr=e, fmt="-" + _MARCADORES_NUBES.get(shape, "o"), ms=4,
                    capsize=3, lw=1.3,
                    color=(colores or _COLORES_NUBES).get(shape, _AZUL), label=shape)
    ax.plot(cota_x, cota_y, "--", color=_ROJO, lw=1.5, label=cota_label)
    ax.set_xlabel("nivel de ruido $q$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.0, qmax)
    ax.set_xticks([0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
    vals_all = [v for v in (_val(c) for c in crudo) if v == v]
    vtop = max(vals_all + [0.0]) * 1.05
    ax.set_ylim(0.0, max(vtop, yfloor))
    # leyenda del PDF: primero las nubes (en orden), la cota al final
    h, l = ax.get_legend_handles_labels()
    orden = {n: i for i, n in enumerate(list(_ORDEN_NUBES) + [cota_label])}
    pares = sorted(zip(l, h), key=lambda p: orden.get(p[0], 99))
    ax.legend([x[1] for x in pares], [x[0] for x in pares],
              fontsize=7, ncol=2, loc="upper left", framealpha=0.95)
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
    """Figura 5.5 — Compliance (izq) y grisura (der) vs p; anota β₁.

    Estilo del PDF: título "Efecto de la penalización: compliance, binariedad y
    topología"; compliance marrón con círculos, grisura azul discontinua con
    cuadrados; eje derecho "grisura (0 = diseño binario)"; β₁ en verde junto a
    cada punto; leyenda en el centro-derecha. (Colores muestreados del PDF.)
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()
    p = df_p["p"].to_numpy()
    l1, = ax1.plot(p, df_p["c(rho*)"], "-o", color=_MARRON, lw=1.9, ms=7,
                   label="compliance $c(\\rho^*)$")
    l2, = ax2.plot(p, df_p["grisura"], "--s", color=_AZUL_TEAL, lw=1.9, ms=7,
                   label="grisura $4\\tilde{\\rho}(1-\\tilde{\\rho})$")
    for _, row in df_p.iterrows():
        ax1.annotate(f"$\\beta_1 = {int(row['beta1'])}$", (row["p"], row["c(rho*)"]),
                     textcoords="offset points", xytext=(0, -16), ha="center",
                     fontsize=9, color=_VERDE_ANOT)
    ax1.set_xlabel("factor de penalización $p$")
    ax1.set_ylabel("compliance $c(\\rho^*)$", color=_MARRON)
    ax2.set_ylabel("grisura (0 = diseño binario)", color=_AZUL_TEAL)
    ax1.set_xticks(p)
    ax1.set_xlim(0.8, 4.2)
    ax1.set_title("Efecto de la penalización: compliance, binariedad y topología")
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0 - 0.07 * (y1 - y0), y1 + 0.04 * (y1 - y0))
    g0, g1 = ax2.get_ylim()
    ax2.set_ylim(min(df_p["grisura"]) - 0.03, max(df_p["grisura"]) + 0.04)
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=8,
               loc="center right")
    fig.tight_layout()
    return fig


def fig_56_malla(df_mesh):
    """Figura 5.6 — Compliance (izq) y β₁ (der) vs malla; eje X logarítmico (Ne).

    Estilo del PDF: título "Independencia de malla: β₁ constante, c estable";
    compliance marrón con círculos (eje izq), β₁ verde con cuadrados (eje der
    "β₁(Ω_sólido)"); anotaciones r_min junto a cada punto; sin leyenda. El eje X
    del PDF es logarítmico en Ne (verificado por posición de marcadores: 0.631).
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()
    ne = df_mesh["Ne"].to_numpy()
    ax1.set_xscale("log")
    ax1.plot(ne, df_mesh["c(rho*)"], "-o", color=_MARRON, lw=1.9, ms=7)
    ax2.plot(ne, df_mesh["beta1"], "-s", color=_VERDE, lw=1.9, ms=7)
    n = len(df_mesh)
    for i, (_, row) in enumerate(df_mesh.iterrows()):
        off = (-44, -4) if i == n - 1 else (-6, -16)
        ax1.annotate(f"$r_{{min}} = {row['rmin']:g}$", (row["Ne"], row["c(rho*)"]),
                     textcoords="offset points", xytext=off,
                     ha="right" if i == n - 1 else "center",
                     fontsize=8, color=_GRIS)
    ax1.set_xlabel("número de elementos $N_e$")
    ax1.set_xticks(ne)
    ax1.set_xticklabels([f"{int(n):,}" for n in ne])
    ax1.minorticks_off()
    ax1.set_xlim(ne.min() / 1.35, ne.max() * 1.35)
    cmin, cmax = float(df_mesh["c(rho*)"].min()), float(df_mesh["c(rho*)"].max())
    ax1.set_ylim(cmin - 0.15 * (cmax - cmin), cmax + 0.15 * (cmax - cmin))
    ax1.set_ylabel("compliance $c(\\rho^*)$", color=_MARRON)
    ax2.set_ylabel("β₁(Ωsólido)", color=_VERDE)
    ax2.set_ylim(-0.4, 3.4)
    ax2.set_yticks([0, 1, 2, 3])
    ax1.set_title("Independencia de malla: β₁ constante, c estable")
    fig.tight_layout()
    return fig


def fig_57_umbral(umbrales, b0s, b1s, vfs, u_umbral=0.5):
    """Figura 5.7 — β₀, β₁ vs umbral de binarización; fracción de volumen.

    Estilo del PDF: volumen en dash-dot gris con eje derecho ajustado a los
    datos (NO [0,1]); línea roja punteada en u=0.5 con texto dentro del
    gráfico; leyenda solo con β₁/β₀. El eje Y de Betti encuadra todos los
    datos (si coinciden con el PDF queda [1,3] como en el documento).
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax2 = ax1.twinx()
    ax1.plot(umbrales, b1s, "-", color=_VERDE, lw=2.5, label="β₁")
    ax1.plot(umbrales, b0s, "--", color=_AZUL, label="β₀")
    ax2.plot(umbrales, vfs, "-.", color=_GRIS, lw=1.5, label="fracción de volumen")
    ax1.axvline(u_umbral, color=_ROJO, ls=":", lw=1.5)
    bmax = max(list(b0s) + list(b1s) + [3])
    y_lo, y_hi = 0.9, bmax + 0.6
    ax1.set_xlim(0.0, 0.95)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_yticks(range(1, int(bmax) + 1))
    ax2.set_ylim(min(vfs) - 0.03, max(vfs) + 0.03)
    ax1.set_xlabel("umbral de binarización u")
    ax1.set_ylabel("número de Betti")
    ax2.set_ylabel("fracción de volumen umbralizada")
    ax1.set_title("Figura 5.7 — Persistencia de β₁ frente al umbral")
    ax1.text(u_umbral + 0.02, y_lo + 0.62 * (y_hi - y_lo), "umbral\nadoptado",
             color=_ROJO, fontsize=8, va="center")
    ax1.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.01, 0.52))
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# CLI — reproduce protocolos y exporta figuras + CSV
# ═══════════════════════════════════════════════════════════════════

def _run_he1(n_points: int, n_rep: int):
    """H.E.1a/H.E.1c: réplicas con ruido para las Figuras 5.2 y 5.3.

    La calibración de τ (Fig. 5.1) vive en `tda.analysis.calibracion` y se
    ejecuta aparte: es un paso previo, no parte de la medición.
    """
    from tda.simulation.he_protocols import (
        protocolo_he1, CATALOGO, Q_LEVELS, TAU_CALIBRADO, SEED,
    )
    sweep, crudo = protocolo_he1(n_points, Q_LEVELS, n_rep, TAU_CALIBRADO, SEED)
    # std por nube/nivel para las barras de error de Fig 5.3
    for shape in CATALOGO:
        std = []
        for q in Q_LEVELS:
            vals = [c["dB_diam"] for c in crudo if c["shape"] == shape and c["q"] == q]
            std.append(float(np.std(vals)) if vals else 0.0)
        sweep[shape]["ratio_diam_std"] = std
    return sweep, crudo


def _run_he2(nex: int, ney: int):
    import pandas as pd
    from tda.simulation.he_protocols import corrida_simp, barrido_he2, FV, P_DEFECTO, RMIN_DEFECTO
    m = corrida_simp(nex, ney, P_DEFECTO, RMIN_DEFECTO, fV=FV)
    sweep = barrido_he2("Rectangular" if nex != ney else "Cuadrado",
                        nex, ney, P_DEFECTO, RMIN_DEFECTO, fV=FV)
    from tda.core.betti2d import betti_gudhi, betti_euler
    umbrales = np.linspace(0.05, 0.95, 19)
    rho2d = m.rho_tilde_final.reshape(ney, nex)
    b0s, b1s, vfs = [], [], []
    for u in umbrales:
        S = rho2d >= u
        vfs.append(float(S.mean()))
        try:
            # Método B (GUDHI): el conteo Euler mezcla 4-conectividad con la
            # unión geométrica y crea agujeros fantasma al fragmentarse (Item 18)
            b0, b1 = betti_gudhi(S)
        except Exception:
            b0, b1 = betti_euler(S)
        b0s.append(b0); b1s.append(b1)
    return m, pd.DataFrame(sweep), umbrales, b0s, b1s, vfs


def main():
    ap = argparse.ArgumentParser(description="Reconstrucción headless de figuras (E3)")
    ap.add_argument("--protocolo", choices=["he1", "he2"], default=None)
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--n", type=int, default=400, help="puntos por nube (H.E.1)")
    ap.add_argument("--R", type=int, default=100, help="réplicas (H.E.1)")
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
        sweep, crudo = _run_he1(args.n, args.R)
        qs = [0.0, 0.05, 0.10, 0.15, 0.20]
        fig_52a_tasa_acierto(qs, sweep, CATALOGO).savefig(OUT_DIR / "fig52a_tasa_acierto.png", dpi=300)
        fig_52b_cota_estabilidad(crudo).savefig(OUT_DIR / "fig52b_cota_estabilidad.png", dpi=300)
        fig_53_cota(qs, crudo, CATALOGO, forma="documento").savefig(OUT_DIR / "fig53a_forma_documento.png", dpi=300)
        fig_53_cota(qs, crudo, CATALOGO, forma="teorema").savefig(OUT_DIR / "fig53b_forma_teorema.png", dpi=300)
        import pandas as pd
        pd.DataFrame(crudo).to_csv(OUT_DIR / "he1_replicas_dB_dH.csv", index=False)
        hechos += ["51", "52a", "52b", "53"]

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
