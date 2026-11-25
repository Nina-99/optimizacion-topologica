"""Experimento headless de H.E.2 con los criterios del Cap. V corregido.

H.E.2a (Cap. V §5.5.1):
  - referente físico: placa uniforme p = 1, ρ ≡ f_V;
  - métrica: compliance del diseño BINARIZADO al umbral u_V que conserva
    el volumen;
  - criterio: reducción ≥ 15 % frente a la placa física;
  - además: cota inferior convexa M* (p = 1, sin filtro) y eficiencia η.

H.E.2b (Cap. V §5.5.3–§5.5.5):
  - β₁ invariante en el voladizo frente a malla (60×30, 120×60, 180×90 con
    r_min físico constante), p ∈ {2,3,4}, r_min ∈ {1,5…4,0} y umbral;
  - viga MBB 120×40 como caso de falsación: se mide la meseta de r_min.

Uso:
    python -m tda.simulation.experiment_simp
"""

import os

import numpy as np
import pandas as pd

from tda.simulation.he_protocols import (
    FV, P_DEFECTO, RMIN_DEFECTO, barrido_he2, corrida_simp, metricas_he2a,
)


def run_headless_experiment(nelx=60, nely=30, fV=FV, penal=P_DEFECTO,
                            rmin=RMIN_DEFECTO, max_iter=300):
    """Ejecuta el Caso 2 completo y escribe los CSV de verificación."""
    print("=" * 72)
    print("  VALIDACIÓN H.E.2 — criterios del Cap. V corregido")
    print("=" * 72)
    print(f"  malla base: {nelx}×{nely} | fV={fV} | p={penal} | rmin={rmin}")
    print("  material adimensional: E0 = 1, espesor = 1, elemento unitario")

    # ── 1. Diseño base ─────────────────────────────────────────────────────
    print("\n--- Fase 1: optimización SIMP base ---")
    m = corrida_simp(nelx, nely, penal, rmin, fV=fV, max_iter=max_iter,
                     caso="voladizo")
    print(f"  convergió: {m.converged} en {m.n_iter} iteraciones")
    print(f"  β₀={m.beta0} | β₁={m.beta1} | "
          f"u_V={m.umbral_binarizacion:.3f} | "
          f"volumen={m.fraccion_volumen_binarizada:.3f}")

    # ── 2. H.E.2a con referente físico y cota convexa ──────────────────────
    print("\n--- Fase 2: H.E.2a (placa p=1, binario y cota convexa) ---")
    met = metricas_he2a(m, fV=fV, max_iter=max_iter)
    print(f"  c(uniforme p=1) = {met['c_unif']:.4f}")
    print(f"  c(binarizado)   = {met['c_bin']:.4f}  (u_V={met['u_v']:.4f})")
    print(f"  c(cota M*)      = {met['c_ast']:.4f}  "
          f"({met['n_iter_ast']} iter, convergió={met['converged_ast']})")
    print(f"  reducción       = {100*met['reduccion']:.2f} %")
    print(f"  eficiencia η    = {met['eficiencia']:.4f}")
    from tda.analysis.estados import (
        ACOTADA, CONCLUYENTE, MEDIDO, NO_SE_SOSTIENE, SE_SOSTIENE,
    )
    he2a_cumple = met["reduccion"] >= 0.15
    he2a_veredicto = SE_SOSTIENE if he2a_cumple else NO_SE_SOSTIENE
    print(f"  H.E.2a (≥15 %)  = {MEDIDO} · {CONCLUYENTE} · {he2a_veredicto}")

    # ── 3. H.E.2b: barrido completo (malla, p, rmin, MBB) ─────────────────
    print("\n--- Fase 3: H.E.2b (barrido multi-configuración + MBB) ---")
    barrido, mallas_2d = barrido_he2(
        "Rectangular", nelx, nely, penal, rmin, fV=fV,
        max_iter=max_iter,
    )
    df = pd.DataFrame(barrido)
    print(df.to_string(index=False))

    voladizo = df[df["Caso"] == "voladizo"]
    mbb = df[df["Caso"] == "mbb"]
    malla = voladizo[voladizo["Config"].str.startswith("malla=")]
    pen = voladizo[voladizo["Config"].str.startswith("p=")]
    rmin_v = voladizo[voladizo["Config"].str.startswith("rmin=")]
    rmin_m = mbb[mbb["Config"].str.startswith("mbb_rmin=")]

    he2b_voladizo = (
        malla["beta1"].nunique() == 1
        and pen[pen["p"] >= 2]["beta1"].nunique() == 1
        and rmin_v["beta1"].nunique() == 1
        and (df["Concordancia"] == "Si").all()
    )
    print(f"\n  β₁ voladizo: malla={malla['beta1'].tolist()} | "
          f"p≥2={pen[pen['p'] >= 2]['beta1'].tolist()} | "
          f"rmin={rmin_v['beta1'].tolist()}")
    print(f"  β₁ MBB:     {rmin_m['beta1'].tolist()} "
          f"→ meseta {rmin_m['beta1'].nunique() == 1}")
    print(f"  H.E.2b voladizo = {MEDIDO} · {CONCLUYENTE} · "
          f"{SE_SOSTIENE if he2b_voladizo else NO_SE_SOSTIENE}")
    print(f"  H.E.2b MBB     = {MEDIDO} · ACOTADA · "
          f"{'meseta constante' if rmin_m['beta1'].nunique() == 1 else 'meseta no constante'}")

    # ── 4. Exportación ─────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    fila = {
        "experimento": "H.E.2",
        "malla": f"{nelx}×{nely}",
        "volfrac": fV,
        "penal": penal,
        "rmin": rmin,
        "u_V": met["u_v"],
        "c_uniforme_p1": met["c_unif"],
        "c_binarizada": met["c_bin"],
        "c_cota_convexa": met["c_ast"],
        "reduccion_vs_placa_p1_pct": 100 * met["reduccion"],
        "eficiencia_relativa": met["eficiencia"],
        "beta1_base": m.beta1,
        "converged": bool(m.converged),
        "n_iter": m.n_iter,
        "he2a_estado": f"{MEDIDO} · {CONCLUYENTE} · {he2a_veredicto}",
        "he2b_voladizo_estado": (f"{MEDIDO} · {CONCLUYENTE} · "
                                 f"{SE_SOSTIENE if he2b_voladizo else NO_SE_SOSTIENE}"),
        "he2b_mbb_estado": f"{MEDIDO} · {ACOTADA}",
        "he2a_cumple": he2a_cumple,
        "he2b_voladizo_cumple": he2b_voladizo,
        "he2b_mbb_meseta_constante": bool(rmin_m["beta1"].nunique() == 1),
    }
    pd.DataFrame([fila]).to_csv("data/validacion_HE2_corregida.csv", index=False)
    df.to_csv("data/barrido_HE2b.csv", index=False)
    if mallas_2d:
        pd.DataFrame([
            {"label": x["label"], "Ne": x["Ne"], "rmin": x["rmin"],
             "beta1": x["beta1"]}
            for x in mallas_2d
        ]).to_csv("data/mallas_HE2b.csv", index=False)

    print("\n  Resultados guardados en:")
    print("    data/validacion_HE2_corregida.csv")
    print("    data/barrido_HE2b.csv")
    print("    data/mallas_HE2b.csv")
    return {"he2a": met, "he2a_cumple": he2a_cumple,
            "he2b_voladizo": he2b_voladizo, "barrido": df}


if __name__ == "__main__":
    run_headless_experiment()
