"""figuras.py — Reconstrucción headless de las figuras del Cap. V (entregable E3).

Genera las figuras de la tesis (5.1–5.7) SIN Streamlit, a partir de los
diccionarios de resultados de los protocolos (`tda.simulation.he_protocols`)
o de los CSV exportados. Cada figura respeta los ejes, rangos y escalas del
documento para que la comparación visual con el PDF sea directa.

Uso como API
────────────
    from tda.visualization.figuras import fig_51_calibracion, fig_53_cota
    fig = fig_51_calibracion(taus, aciertos)
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


# ═══════════════════════════════════════════════════════════════════
# CASO 1 — TDA (Figuras 5.1, 5.2, 5.3)
# ═══════════════════════════════════════════════════════════════════

def fig_51_calibracion(taus, aciertos, n_nubes: int, tau_adoptado: float = 0.15):
    """Figura 5.1 — Curva de calibración de τ.

    X: τ ∈ [0.05, 0.40]; Y: nº de nubes con lectura β₁ correcta (0..n_nubes).
    La franja verde marca la ventana de acierto máximo.
    """
    taus = np.asarray(taus)
    aciertos = np.asarray(aciertos)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    amax = int(aciertos.max()) if len(aciertos) else 0
    ventana = taus[aciertos == amax] if amax else np.array([])
    if len(ventana):
        ax.axvspan(ventana.min(), ventana.max(), color="green", alpha=0.12,
                   label=f"Ventana válida [{ventana.min():.2f}, {ventana.max():.2f}]")
    ax.plot(taus, aciertos, "-o", color=_AZUL, ms=5, label="Nubes con lectura β₁ correcta")
    ax.axvline(tau_adoptado, color=_ROJO, ls="--", lw=1.5, label=f"τ adoptado = {tau_adoptado}")
    ax.set_xlabel("umbral relativo de persistencia τ")
    ax.set_ylabel(f"nubes con lectura β₁ correcta (de {n_nubes})")
    ax.set_title("Figura 5.1 — Calibración del umbral τ contra el patrón de oro")
    ax.set_xlim(0.05, 0.40)
    ax.set_ylim(0, n_nubes)
    ax.set_yticks(range(0, n_nubes + 1))
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    return fig


# Marcadores por nube (como el PDF: cada curva con su símbolo)
_MARCADORES_52A = {
    "anillo": "o", "circulo": "s", "disco": "^", "dos_circulos": "v",
    "esfera": "D", "gaussiana": "P", "toro": "X",
}


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
    cmap = plt.get_cmap("tab10")
    for i, shape in enumerate(catalogo):
        ax.plot(qs, sweep[shape]["acierto"], "-",
                marker=_MARCADORES_52A.get(shape, "o"), ms=5,
                color=(colores or {}).get(shape, cmap(i % 10)), label=shape)
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


def fig_53_cota(qs, crudo, catalogo, colores=None):
    """Figura 5.3 — dB/dH promediada por nube y nivel vs la cota 2 (teorema).

    Forma del Teorema de Estabilidad: dB ≤ 2·dH. Todas las curvas quedan
    por debajo de la recta dB/dH = 2; la cota 2q (no implicada por el
    teorema) se omite a propósito (ver Correcciones, Item 11).
    """
    qs = np.asarray(qs)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, shape in enumerate(catalogo):
        y, e = [], []
        for q in qs:
            vals = [c["dB"] / c["dH"] for c in crudo
                    if c["shape"] == shape and c["q"] == q and c["dH"] > 0]
            y.append(float(np.mean(vals)) if vals else np.nan)
            e.append(float(np.std(vals)) if vals else 0.0)
        ax.errorbar(qs, y, yerr=e, fmt="-o", ms=4, capsize=3, lw=1.3,
                    color=(colores or {}).get(shape, cmap(i % 10)), label=shape)
    ax.axhline(2.0, ls="--", color=_ROJO, lw=1.5, label="cota dB = 2 dH")
    ax.set_xlabel("nivel de ruido q")
    ax.set_ylabel("dB / dH")
    ax.set_title("Figura 5.3 — Distancia bottleneck relativa frente a la cota del teorema")
    ax.set_xlim(0.0, float(qs.max()))
    ax.set_ylim(0.0, 2.1)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# CASO 2 — SIMP (Figuras 5.4, 5.5, 5.6, 5.7)
# ═══════════════════════════════════════════════════════════════════

def fig_54a_diseno(rho_2d, binaria_2d, beta0, beta1, nx, ny):
    """Figura 5.4(a) — Densidad continua (izq) y diseño binarizado (der)."""
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
    fig.suptitle("Figura 5.4(a) — Distribución de material del diseño convergido")
    fig.tight_layout()
    return fig


def fig_54b_historial(c_hist, c_uniforme, c_final):
    """Figura 5.4(b) — Historial de compliance en escala logarítmica."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    k = np.arange(1, len(c_hist) + 1)
    ax.semilogy(k, c_hist, "-o", ms=3, color=_AZUL, label="c(k)")
    ax.axhline(c_uniforme, color=_GRIS, ls=":", lw=1.5, label=f"c(fV) = {c_uniforme:.1f}")
    ax.axhline(c_final, color=_VERDE, ls="--", lw=1.5, label=f"c* = {c_final:.1f}")
    ax.set_xlabel("iteración k")
    ax.set_ylabel("compliance c(ρ) (escala log)")
    ax.set_title("Figura 5.4(b) — Historial de compliance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_55_penalizacion(df_p):
    """Figura 5.5 — Compliance (izq) y grisura (der) vs p; anota β₁."""
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()
    p = df_p["p"].to_numpy()
    ax1.plot(p, df_p["c(rho*)"], "-o", color=_AZUL, label="compliance")
    ax2.plot(p, df_p["grisura"], "-s", color=_NARANJA, label="grisura")
    for _, row in df_p.iterrows():
        ax1.annotate(f"β₁={row['beta1']}", (row["p"], row["c(rho*)"]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax1.set_xlabel("factor de penalización p")
    ax1.set_ylabel("compliance c(ρ*)", color=_AZUL)
    ax2.set_ylabel("grisura 4ρ̃(1−ρ̃)", color=_NARANJA)
    ax1.set_xticks(p)
    ax1.set_xlim(0.8, 4.2)
    ax1.set_title("Figura 5.5 — Efecto del factor de penalización")
    # margen superior para que la anotación β₁ no se recorte
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.12 * (y1 - y0))
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def fig_56_malla(df_mesh):
    """Figura 5.6 — Compliance (izq) y β₁ (der) vs malla.

    Eje X categórico (las mallas no son equidistantes en Ne): cada tick
    muestra la malla y su Ne. Márgenes para que las anotaciones rmin
    no se recorten, y leyenda combinada de ambos ejes.
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()
    labels = df_mesh["Malla"].tolist()
    ne = df_mesh["Ne"].tolist()
    x = np.arange(len(labels))
    l1, = ax1.plot(x, df_mesh["c(rho*)"], "-o", color=_AZUL, label="compliance c(ρ*)")
    l2, = ax2.plot(x, df_mesh["beta1"], "-s", color=_VERDE, lw=2.5, label="β₁")
    for i, (_, row) in enumerate(df_mesh.iterrows()):
        ax1.annotate(f"rmin={row['rmin']}", (x[i], row["c(rho*)"]),
                     textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\nNe={n}" for l, n in zip(labels, ne)])
    ax1.set_ylabel("compliance c(ρ*)", color=_AZUL)
    ax2.set_ylabel("β₁", color=_VERDE)
    b1 = df_mesh["beta1"].to_numpy()
    ax2.set_ylim(min(b1 + [0]) - 1.5, max(b1 + [4]) + 1.5)
    ax2.set_yticks(range(int(min(b1 + [0])), int(max(b1 + [4])) + 1))
    # margen vertical para que las anotaciones rmin no se recorten
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0 - 0.15 * (y1 - y0), y1 + 0.20 * (y1 - y0))
    ax1.set_xlim(-0.3, len(labels) - 0.7)
    ax1.set_xlabel("malla (Ne entre paréntesis)")
    ax1.set_title("Figura 5.6 — Independencia de malla (rmin escalado)")
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=8, loc="best")
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
    from tda.simulation.he_protocols import (
        protocolo_he1, calibrar_tau, CATALOGO, Q_LEVELS, TAU_CALIBRADO, SEED,
    )
    taus, aciertos = calibrar_tau(n_points, SEED)
    sweep, crudo = protocolo_he1(n_points, Q_LEVELS, n_rep, TAU_CALIBRADO, SEED)
    # std por nube/nivel para las barras de error de Fig 5.3
    for shape in CATALOGO:
        std = []
        for q in Q_LEVELS:
            vals = [c["dB_diam"] for c in crudo if c["shape"] == shape and c["q"] == q]
            std.append(float(np.std(vals)) if vals else 0.0)
        sweep[shape]["ratio_diam_std"] = std
    return taus, aciertos, sweep, crudo


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
        from tda.simulation.he_protocols import CATALOGO
        print(f"[H.E.1] n={args.n}, R={args.R}...")
        taus, aciertos, sweep, crudo = _run_he1(args.n, args.R)
        qs = [0.0, 0.05, 0.10, 0.15, 0.20]
        fig_51_calibracion(taus, aciertos, len(CATALOGO)).savefig(OUT_DIR / "fig51_calibracion_tau.png", dpi=300)
        fig_52a_tasa_acierto(qs, sweep, CATALOGO).savefig(OUT_DIR / "fig52a_tasa_acierto.png", dpi=300)
        fig_52b_cota_estabilidad(crudo).savefig(OUT_DIR / "fig52b_cota_estabilidad.png", dpi=300)
        fig_53_cota(qs, crudo, CATALOGO).savefig(OUT_DIR / "fig53_bottleneck_2q.png", dpi=300)
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
        fig_54a_diseno(m.rho_tilde_final.reshape(ney, nex), binaria, m.beta0, m.beta1, nex, ney).savefig(OUT_DIR / "fig54a_diseno.png", dpi=300)
        fig_54b_historial(m.c_hist, c_unif, c_fin).savefig(OUT_DIR / "fig54b_historial.png", dpi=300)
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
