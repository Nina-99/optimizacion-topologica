"""Artefacto del truncamiento de la filtración — panel (a) de la Figura 5.3.

El panel explica **por qué la versión anterior del documento concluyó
`dB/diam ≤ 2q`**: si la filtración de Vietoris–Rips se corta en ε ≤ 0,6, el
ciclo principal del círculo (que muere en √3 ≈ 1,73) queda como barra
infinita, se descarta, y la distancia bottleneck solo ve el ruido de corta
vida. Con la filtración completa esa cota se rompe.

Se usa **a propósito la definición de ruido anterior** (`σ = q·diam X`, sin
normalizar por √d): el objetivo es reproducir el artefacto de la figura
original, no medir con el protocolo corregido.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from ripser import ripser

from tda.analysis.estabilidad_dimensional import diagramas_finitos
from tda.core.topology import bottleneck_distance
from tda.processing.sampling import compute_diameter, generar_corregida

N_TRUNC = 400
QS_TRUNC = (0.0, 0.05, 0.10, 0.15, 0.20)
R_TRUNC = 20
EPS_TRUNC = 0.6            # corte de la filtración de la figura original
SEED_TRUNC = 42
NUBE_TRUNC = "circulo"

_MODOS = {f"truncado (ε ≤ {EPS_TRUNC}, barras infinitas descartadas)": EPS_TRUNC,
          "completo (sin truncar)": np.inf}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUTA_TRUNCAMIENTO = _PROJECT_ROOT / "data" / "figuras" / "fig53_truncamiento.csv"


def _dgms(puntos: np.ndarray, thresh: float):
    """Diagramas finitos de una nube con la filtración cortada en `thresh`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dgms = ripser(puntos, maxdim=1, thresh=thresh)["dgms"]
    return diagramas_finitos(dgms)


def analisis_truncamiento(
    n_points: int = N_TRUNC,
    qs: Sequence[float] = QS_TRUNC,
    n_rep: int = R_TRUNC,
    eps: float = EPS_TRUNC,
) -> pd.DataFrame:
    """Repite el círculo en los dos modos de filtración y mide `dB/diam(X)`.

    Returns:
        DataFrame con columnas `modo`, `q`, `replica`, `dB_diam` (200 filas con
        los valores por defecto).
    """
    X = generar_corregida(NUBE_TRUNC, n_points, np.random.default_rng(SEED_TRUNC))
    diam = compute_diameter(X)
    modos = {f"truncado (ε ≤ {eps}, barras infinitas descartadas)": eps,
             "completo (sin truncar)": np.inf}
    finitos_X = {modo: _dgms(X, th) for modo, th in modos.items()}

    filas = []
    for q in qs:
        for r in range(n_rep):
            # Semilla del PDF original: base 7, independiente del catálogo (42).
            rng = np.random.default_rng(7 + 1000 * r + int(1e5 * q))
            # Ruido de la figura original: sigma = q·diam(X), sin 1/sqrt(d).
            Y = X + rng.normal(0.0, q * diam, X.shape)
            for modo, th in modos.items():
                finitos_Y = _dgms(Y, th)
                dB = max(bottleneck_distance(a, b)
                         for a, b in zip(finitos_X[modo], finitos_Y))
                filas.append({"modo": modo, "q": q, "replica": r,
                              "dB_diam": dB / diam})
    return pd.DataFrame(filas)


def guardar_truncamiento(df: pd.DataFrame,
                         ruta: Path = RUTA_TRUNCAMIENTO) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)


def cargar_truncamiento(ruta: Path = RUTA_TRUNCAMIENTO) -> pd.DataFrame | None:
    return pd.read_csv(ruta) if ruta.exists() else None


def resumen_truncamiento(df: pd.DataFrame) -> Dict[str, Dict[float, float]]:
    """Media de `dB/diam` por modo y nivel de ruido (lo que reporta el texto)."""
    return {modo: g.groupby("q")["dB_diam"].mean().round(3).to_dict()
            for modo, g in df.groupby("modo", sort=False)}


__all__ = [
    "N_TRUNC", "QS_TRUNC", "R_TRUNC", "EPS_TRUNC", "NUBE_TRUNC",
    "RUTA_TRUNCAMIENTO", "analisis_truncamiento", "guardar_truncamiento",
    "cargar_truncamiento", "resumen_truncamiento",
]
