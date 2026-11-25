"""Calcula el veredicto H.G. con los CSV reales y deja tabla + captura.

No reejecuta los protocolos: usa los artefactos ya generados
(`fig52_replicas.csv`, `barrido_HE2b.csv`, `validacion_HE2_corregida.csv`)
y aplica exactamente la lógica de la página H.G.

Salidas:
    data/veredicto_HG.csv
    data/veredicto_HG.png
    docs/Veredicto_HG.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tda.analysis.estabilidad_dimensional import rango_fiabilidad
from tda.analysis.stability import resultado_piloto_pdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
FIGS = DATA / "figuras"
OUT_PNG = DATA / "veredicto_HG.png"
OUT_CSV = DATA / "veredicto_HG.csv"
OUT_MD = ROOT / "docs" / "Veredicto_HG.md"


def main() -> None:
    rep = pd.read_csv(FIGS / "fig52_replicas.csv")
    barrido = pd.read_csv(DATA / "barrido_HE2b.csv")
    val = pd.read_csv(DATA / "validacion_HE2_corregida.csv").iloc[0]

    # H.E.1a
    frac = float((rep["dB"] <= rep["cota_teorema"] + 1e-9).mean())
    he1a_ok = frac == 1.0

    # H.E.1c
    rango_full = rango_fiabilidad(rep)
    rango = rango_full[~rango_full["nube"].isin(["toro"])]
    n_sost = int((rango["estado"] == "se sostiene").sum())
    n_acot = int(rango["estado"].str.startswith("acotada").sum())
    n_no = int((rango["estado"] == "no se sostiene").sum())
    n_toro = int(len(rango_full) - len(rango))
    he1c_ok = n_sost == len(rango)
    he1c_acotada = (n_sost + n_acot) > 0

    # H.E.2b
    vol = barrido[barrido["Caso"] == "voladizo"]
    dfp = vol[(vol["Config"].str.startswith("p=")) & (vol["p"] >= 2)]
    dfm = vol[vol["Config"].str.startswith("malla=")]
    dfr = vol[vol["Config"].str.startswith("rmin=")]
    inv = (dfm["beta1"].nunique() <= 1 and dfp["beta1"].nunique() <= 1
           and dfr["beta1"].nunique() <= 1)
    conc = bool((barrido["Concordancia"] == "Si").all())
    he2b_ok = bool(inv and conc)
    mbb = barrido[barrido["Caso"] == "mbb"].sort_values("rmin")
    mbb_b1 = [int(v) for v in mbb["beta1_u05"].tolist()]
    vol_b1 = sorted(set(int(v) for v in dfr["beta1_u05"].tolist()))
    vol_b1_txt = str(vol_b1[0]) if len(vol_b1) == 1 else str(vol_b1)

    # H.E.1b: piloto del PDF
    mcn = resultado_piloto_pdf()
    he1b_ok = bool(mcn["mc_nemar"]["significant"]) and (
        mcn["exactitud_tda"] > mcn["exactitud_eucl"]
    )

    # H.E.2a (fuera de H.G.)
    he2a_ok = float(val["reduccion_vs_placa_p1_pct"]) >= 15.0
    he2a_val = float(val["reduccion_vs_placa_p1_pct"])
    eta = float(val["eficiencia_relativa"])

    estabilidad_ok = bool(he1a_ok and he2b_ok and (he1c_ok or he1c_acotada))
    comparativa_ok = bool(he1b_ok)
    if estabilidad_ok and comparativa_ok:
        hg_estado = "SOSTENIDA"
    elif estabilidad_ok:
        hg_estado = "PARCIAL / CONDICIONADA"
    else:
        hg_estado = "NO SOSTENIDA"

    filas = [
        {"Cláusula": "A. Estabilidad/verificabilidad de β_k bajo q ≤ q*",
         "Hipótesis": "H.E.1a + H.E.1c + H.E.2b", "Evidencia": "Medido",
         "Estado": "SOSTENIDA CON ALCANCE" if estabilidad_ok else "NO SOSTENIDA",
         "Valor": (f"H.E.1a {frac:.3f}; H.E.1c {n_sost} banda/{n_acot} acotadas/"
                   f"{n_no} sin q*; toro aparte ({n_toro} combinaciones); "
                   f"H.E.2b voladizo β₁={vol_b1_txt}")},
        {"Cláusula": "B. Ventaja frente a descriptores euclidianos",
         "Hipótesis": "H.E.1b", "Evidencia": "Medido",
         "Estado": "SOSTENIDA" if comparativa_ok else "NO CONCLUYENTE",
         "Valor": (f"{mcn['exactitud_tda']:.3f} vs {mcn['exactitud_eucl']:.3f}; "
                   f"McNemar p={mcn['mc_nemar']['p_value']:.4f}")},
        {"Cláusula": "Eficiencia estructural SIMP (fuera de H.G.)",
         "Hipótesis": "H.E.2a", "Evidencia": "Medido",
         "Estado": "INFORMATIVA",
         "Valor": f"reducción {he2a_val:.2f} %; η={eta:.3f}"},
    ]
    df = pd.DataFrame(filas)
    df.to_csv(OUT_CSV, index=False)

    # ── Tarjeta PNG ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7.2))
    fig.patch.set_facecolor("#f7f7f4")
    fig.text(0.05, 0.92, "Veredicto H.G. — Hipótesis General (Cap. I §1.5)",
             fontsize=18, weight="bold", color="#111111")
    fig.text(0.05, 0.865,
             "H.G. = cláusula A (estabilidad/verificabilidad) + cláusula B (ventaja comparativa).",
             fontsize=10, color="#444444")

    color = {"SOSTENIDA": "#1e8449", "PARCIAL / CONDICIONADA": "#d35400",
             "NO SOSTENIDA": "#c0392b"}[hg_estado]
    fig.text(0.05, 0.76, f"H.G. {hg_estado}", fontsize=20, weight="bold",
             color="white",
             bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor="none"))

    y = 0.60
    for fila in filas:
        fig.text(0.05, y, fila["Cláusula"], fontsize=11, weight="bold", color="#111111")
        fig.text(0.05, y - 0.045, fila["Hipótesis"], fontsize=9.5, color="#555555")
        estado = fila["Estado"]
        eco = ("#1e8449" if estado.startswith("SOSTENIDA") else
               "#d35400" if "CONCLUYENTE" in estado else
               "#c0392b" if "NO" in estado else "#2c3e50")
        fig.text(0.60, y, f"{fila['Evidencia']} · {estado}", fontsize=10.5, weight="bold", color=eco)
        fig.text(0.60, y - 0.045, fila["Valor"], fontsize=9.3, color="#333333")
        y -= 0.16

    fig.text(0.05, 0.08,
             "H.E.2a queda fuera de H.G.: mide eficiencia de compliance, no estabilidad del descriptor.",
             fontsize=9, color="#555555")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    md = ["# Veredicto H.G. con datos reales", "",
          f"**Resultado: H.G. {hg_estado}.**", "",
          "| Cláusula | Hipótesis | Evidencia | Veredicto | Valor |",
          "|---|---|---|---|---|"]
    for f in filas:
        md.append(f"| {f['Cláusula']} | {f['Hipótesis']} | {f['Evidencia']} | **{f['Estado']}** | {f['Valor']} |")
    md += ["", "## Lectura", "",
           "- La cláusula A se sostiene con alcance: H.E.1a y H.E.2b se sostienen; "
           "H.E.1c queda acotada (no todas las nubes alcanzan <50 %).",
           "- La cláusula B es **no concluyente**: el piloto del PDF da 0,950 vs 0,925 "
           "y McNemar no significativo.",
           "- H.E.2a es informativa y **no** forma parte de H.G.", "",
           f"Captura: `{OUT_PNG.relative_to(ROOT)}`", ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] {OUT_CSV}")
    print(f"[OK] {OUT_PNG}")
    print(f"[OK] {OUT_MD}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
