"""Replica y justifica las figuras de `8.Figuras/` (Cap. V).

Genera `data/figuras/replica/fig5_*.png` y un libro de valores
`data/figuras/replica/valores.csv` con la métrica que sostiene cada figura.
No modifica los PNG originales de `8.Figuras/`.

Uso:
    python scripts/replicar_figuras.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tda.analysis.calibracion import TAU_CALIBRADO          # noqa: E402
from tda.core.betti2d import betti_euler                    # noqa: E402
from tda.simulation.he_protocols import (                   # noqa: E402
    FV, MALLAS_HE2, P_DEFECTO, RMIN_DEFECTO, corrida_simp, metricas_he2a,
)
from tda.visualization.figuras import (                     # noqa: E402
    fig_51_calibracion, fig_52_estabilidad, fig_53_cota_corregida,
    fig_54a_diseno, fig_54b_historial, fig_55_penalizacion,
    fig_56_malla, fig_57_umbral, fig_58_rmin_falsacion,
)
from tda.visualization.leyendas import LEYENDAS_FIGURAS       # noqa: E402

DATA = ROOT / "data"
FIGS = DATA / "figuras"
OUT = FIGS / "replica"


def _leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(FIGS / nombre)


def _fila(registros: list, figura: str, metrica: str, valor) -> None:
    registros.append({"figura": figura, "metrica": metrica, "valor": valor})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg: list[dict] = []

    # ── Fig. 5.1 ────────────────────────────────────────────────────────
    cal = _leer("fig51_calibracion_C0.csv")
    mar = _leer("fig51_margenes_C.csv")
    fig_51_calibracion(cal, {"tau_star": TAU_CALIBRADO,
                             "ventana_min": TAU_CALIBRADO,
                             "ventana_max": TAU_CALIBRADO}, mar
                      ).savefig(OUT / "fig5_1_calibracion.png", dpi=200)
    _fila(reg, "5.1", "tau_estrella", TAU_CALIBRADO)
    _fila(reg, "5.1", "nubes_7_7", int(cal[cal.tau == TAU_CALIBRADO].correcto.sum()))

    # ── Fig. 5.2 ────────────────────────────────────────────────────────
    rep = _leer("fig52_replicas.csv")
    q0 = _leer("fig52_q0.csv")
    rep["razon"] = rep["dB"] / rep["cota_teorema"].replace(0, np.nan)
    fig_52_estabilidad(rep, q0).savefig(OUT / "fig5_2_tda_estabilidad.png", dpi=200)
    _fila(reg, "5.2", "replicas", len(rep))
    _fila(reg, "5.2", "violaciones_dB_2dH", int((rep["dB"] > rep["cota_teorema"] + 1e-9).sum()))
    for d, g in rep.groupby("d"):
        _fila(reg, "5.2", f"razon_media_d{d}", round(float(g["razon"].mean()), 3))

    # ── Fig. 5.3 ────────────────────────────────────────────────────────
    trunc = _leer("fig53_truncamiento.csv")
    fig_53_cota_corregida(trunc, rep).savefig(OUT / "fig5_3_cota_corregida.png", dpi=200)
    for modo, g in trunc.groupby("modo", sort=False):
        for q, v in g.groupby("q")["dB_diam"].mean().items():
            _fila(reg, "5.3", f"{modo}_q{q}", round(float(v), 3))
    _fila(reg, "5.3", "razon_cota_prob_max", round(float((rep.dB / rep.cota_prob).max()), 3))

    # ── Fig. 5.4 ────────────────────────────────────────────────────────
    m = corrida_simp(60, 30, P_DEFECTO, RMIN_DEFECTO, fV=FV, max_iter=300)
    met = metricas_he2a(m, fV=FV, max_iter=300)
    binaria = m.binaria(m.umbral_binarizacion).astype(float)
    fig_54a_diseno(m.rho_tilde_final.reshape(30, 60), binaria, m.beta0, m.beta1,
                   60, 30, p=P_DEFECTO, fv=FV, rmin=RMIN_DEFECTO,
                   u_v=m.umbral_binarizacion
                   ).savefig(OUT / "fig5_4_simp_base.png", dpi=200)
    fig_54b_historial(m.c_hist, met["c_unif"], m.c_final, n_iter_max=300,
                      converged=bool(m.converged)
                      ).savefig(OUT / "fig5_4b_convergencia.png", dpi=200)
    _fila(reg, "5.4", "iteraciones", m.n_iter)
    _fila(reg, "5.4", "convergido", bool(m.converged))
    _fila(reg, "5.4", "u_V", round(float(m.umbral_binarizacion), 3))
    _fila(reg, "5.4", "c_uniforme_p1", round(float(met["c_unif"]), 3))
    _fila(reg, "5.4", "c_binarizada", round(float(met["c_bin"]), 3))
    _fila(reg, "5.4", "c_cota_convexa", round(float(met["c_ast"]), 3))
    _fila(reg, "5.4", "reduccion_pct", round(100 * float(met["reduccion"]), 3))
    _fila(reg, "5.4", "eficiencia", round(float(met["eficiencia"]), 3))

    # ── Barrido H.E.2 (Figs. 5.5, 5.6, 5.8) ─────────────────────────────
    barrido = pd.read_csv(DATA / "barrido_HE2b.csv") if (DATA / "barrido_HE2b.csv").exists() else None
    if barrido is None or "beta1_u05" not in barrido.columns:
        raise SystemExit("Falta data/barrido_HE2b.csv con beta1_u05; correr experiment_simp.")

    df_p = barrido[(barrido.Caso == "voladizo") & barrido.Config.str.startswith("p=")]
    fig_55_penalizacion(df_p).savefig(OUT / "fig5_5_penalizacion.png", dpi=200)
    for _, r in df_p.iterrows():
        _fila(reg, "5.5", f"p{int(r.p)}_c_bin", round(float(r["c_binarizada"]), 3))
        _fila(reg, "5.5", f"p{int(r.p)}_grisura", round(float(r.grisura), 3))

    df_mesh = barrido[(barrido.Caso == "voladizo") & barrido.Config.str.startswith("malla=")]
    mallas_2d = []
    for nx, ny in MALLAS_HE2:
        mm = corrida_simp(nx, ny, P_DEFECTO, RMIN_DEFECTO * (nx / 60.0),
                          max_iter=300, Lx=60.0, Ly=30.0)
        mallas_2d.append({"label": f"{nx}×{ny}", "Ne": nx * ny,
                          "rmin": round(RMIN_DEFECTO * (nx / 60.0), 2),
                          "beta1": mm.beta1,
                          "rho": mm.rho_tilde_final.reshape(ny, nx)})
    fig_56_malla(df_mesh, mallas_2d=mallas_2d).savefig(OUT / "fig5_6_malla.png", dpi=200)
    for _, r in df_mesh.iterrows():
        _fila(reg, "5.6", f"{r.Config}_beta1", int(r.beta1))
        _fila(reg, "5.6", f"{r.Config}_c_bin", round(float(r["c_binarizada"]), 3))

    # ── Fig. 5.7 ────────────────────────────────────────────────────────
    u = np.arange(0.025, 0.975 + 1e-9, 0.025)
    b0s, b1s, vfs = [], [], []
    for x in u:
        S = m.rho_tilde_final.reshape(30, 60) >= x
        b0, b1 = betti_euler(S)
        b0s.append(b0); b1s.append(b1); vfs.append(float(S.mean()))
    fig_57_umbral(u, b0s, b1s, vfs, u_umbral=m.umbral_binarizacion,
                  dgm1=m.dgm1).savefig(OUT / "fig5_7_umbral.png", dpi=200)
    _fila(reg, "5.7", "u_V", round(float(m.umbral_binarizacion), 3))
    _fila(reg, "5.7", "beta1_en_uV", int(m.beta1))
    _fila(reg, "5.7", "persistencia_min_H1", round(float(m.persistencia_min_h1), 3))

    # ── Fig. 5.8 ────────────────────────────────────────────────────────
    df_vol = barrido[(barrido.Caso == "voladizo") & barrido.Config.str.startswith("rmin=")]
    df_mbb = barrido[(barrido.Caso == "mbb") & barrido.Config.str.startswith("mbb_rmin=")]
    fig_58_rmin_falsacion(df_vol, df_mbb).savefig(OUT / "fig5_8_rmin_falsacion.png", dpi=200)
    for _, r in df_vol.iterrows():
        _fila(reg, "5.8", f"voladizo_rmin{r.rmin}_beta1_u05", int(r.beta1_u05))
    for _, r in df_mbb.iterrows():
        _fila(reg, "5.8", f"mbb_rmin{r.rmin}_beta1_u05", int(r.beta1_u05))

    libro = pd.DataFrame(reg)
    libro.to_csv(OUT / "valores.csv", index=False)
    pd.DataFrame(
        [{"figura": k, "leyenda": v} for k, v in LEYENDAS_FIGURAS.items()]
    ).to_csv(OUT / "leyendas.csv", index=False)
    print(f"[OK] Figuras replicadas en {OUT}")
    print(f"[OK] Libro de valores: {OUT / 'valores.csv'} ({len(libro)} filas)")


if __name__ == "__main__":
    main()
