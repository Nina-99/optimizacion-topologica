"""Protocolo de la Figura 5.2 — estabilidad del diagrama y fiabilidad de βk.

Replica el experimento del Cap. V corregido (`caso1_tda.py`): para cada nube del
catálogo, cada dimensión ambiente y cada nivel de ruido, se perturba la nube,
se mide (β₀, β₁) a umbral τ·diam y se compara el diagrama completo contra la
cota de estabilidad.

Diferencias deliberadas con la versión anterior del software (que la figura
original no tenía): el ruido se normaliza por la dimensión
(`σ = q·diam(X)/√d`, Item 29), el catálogo es el corregido (Item 30) y τ es el
τ* calibrado (Fig. 5.1).

Convenciones del diseño (idénticas al PDF corregido):

- Embebido por relleno de ceros: `incrustar(X, d)` no cambia las distancias,
  así que en `q = 0` la lectura es la misma para toda `d` (por eso el punto
  `q = 0` de los tres paneles (a) sale de una única medición sin ruido).
- Lectura de βk con el **diámetro de la nube perturbada** `diam(Y)`, que es lo
  que el umbral filtra; el diámetro que se reporta es `diam(X)`.
- `dB` = máximo sobre k ∈ {0, 1} de la distancia bottleneck de los diagramas
  finitos; `dH` = distancia de Hausdorff entre nubes.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from ripser import ripser

from tda.analysis.calibracion import (
    TAU_CALIBRADO, betti_tau, margenes,
)
from tda.analysis.stability import hausdorff_distance
from tda.core.topology import bottleneck_distance
from tda.processing.sampling import (
    CATALOGO_CORREGIDO, ORDEN_CALIBRACION, compute_diameter, generar_corregida,
    incrustar,
)

# ── Parámetros del experimento (Cap. V, Fig. 5.2) ───────────────────
N_FIG52 = 400                     # puntos por nube
R_FIG52 = 20                      # réplicas por (nube, d, q)
DELTA_FIG52 = 0.05                # δ de la cota probabilística
QS_FIG52 = (0.05, 0.10, 0.15, 0.20)
DIMS_EXTRA = (10, 100)            # además de la dimensión nativa de cada nube
SEED_MEDICION = 42                # catálogo de medición C (semillas 42 + i)
UMBRAL_ACEPTABLE = 0.95           # q* = mayor q con acierto ≥ 0.95

# Caché en disco: la corrida completa son ~1 700 réplicas (~3 min en paralelo),
# inviable dentro de Streamlit. La app y el CLI leen estos CSV.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUTA_REPLICAS = _PROJECT_ROOT / "data" / "figuras" / "fig52_replicas.csv"
RUTA_Q0 = _PROJECT_ROOT / "data" / "figuras" / "fig52_q0.csv"

_DIM_NATIVA = {"circulo": 2, "anillo": 2, "dos_circulos": 2, "toro": 3,
               "disco": 2, "esfera": 3, "gaussiana": 2}


def dimensiones(shape: str) -> List[int]:
    """Dimensiones ambiente evaluadas para una nube: nativa, 10 y 100."""
    return [_DIM_NATIVA[shape], *DIMS_EXTRA]


def diagramas_finitos(dgms: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Diagramas (n, 2) sin la barra infinita de H₀.

    La barra esencial se descarta porque su muerte no es finita; en la
    distancia bottleneck se empareja con la de la nube perturbada a costo 0.
    """
    out = []
    for k in range(2):
        d = np.asarray(dgms[k], dtype=float) if k < len(dgms) else np.empty((0, 2))
        d = d.reshape(-1, 2)
        if d.size:
            d = d[np.isfinite(d[:, 1])]
        out.append(d)
    return out


def cota_probabilistica(sigma: float, d: int, n: int, delta: float) -> float:
    """Cota del Cap. V: 2σ(√d + √(2·ln(n/δ))), con δ = 0,05."""
    return 2.0 * sigma * (np.sqrt(d) + np.sqrt(2.0 * np.log(n / delta)))


def _replica(tarea: Tuple[str, int, float, int, int, float]) -> Dict:
    """Una réplica (nube, d, q, r, n, τ). Nivel superior: debe ser picklable."""
    shape, d, q, r, n_points, tau = tarea
    i = ORDEN_CALIBRACION.index(shape)
    b0_teo, b1_teo = CATALOGO_CORREGIDO[shape]

    X = incrustar(generar_corregida(shape, n_points,
                                    np.random.default_rng(SEED_MEDICION + i)), d)
    diam = compute_diameter(X)
    sigma = q * diam / np.sqrt(d)
    rng = np.random.default_rng(
        SEED_MEDICION + 1000 * r + int(1e5 * q) + 10_000_000 * d + 100 * i)
    Y = X + rng.normal(0.0, sigma, X.shape)

    with warnings.catch_warnings():
        # ripser avisa "más columnas que filas" cuando n < d en las corridas de
        # test (n chico, d = 100); las filas ya son los puntos.
        warnings.simplefilter("ignore")
        dgm_X = ripser(X, maxdim=1)["dgms"]
        dgm_Y = ripser(Y, maxdim=1)["dgms"]
    b0, b1 = betti_tau(dgm_Y, tau, compute_diameter(Y))

    fin_X, fin_Y = diagramas_finitos(dgm_X), diagramas_finitos(dgm_Y)
    dB = max(bottleneck_distance(a, b) for a, b in zip(fin_X, fin_Y))
    dH = hausdorff_distance(X, Y)

    return {
        "nube": shape, "d": d, "q": q, "replica": r,
        "diam": diam, "sigma": sigma,
        "dB": dB, "dH": dH,
        "cota_teorema": 2.0 * dH,
        "cota_prob": cota_probabilistica(sigma, d, n_points, DELTA_FIG52),
        "b0": int(b0), "b1": int(b1), "b0_teo": int(b0_teo), "b1_teo": int(b1_teo),
        "acierto": int(b0 == b0_teo and b1 == b1_teo),
    }


def protocolo_fig52(
    n_points: int = N_FIG52,
    qs: Sequence[float] = QS_FIG52,
    n_rep: int = R_FIG52,
    tau: float = TAU_CALIBRADO,
    n_jobs: int | None = None,
) -> Dict:
    """Corre las 1 680 réplicas de la Fig. 5.2 y devuelve datos tabulares.

    Returns:
        dict con `replicas` (DataFrame, una fila por réplica), `q0` (DataFrame,
        lectura sin ruido por nube: el punto q = 0 de los paneles (a)) y `tau`.
    """
    tareas = [(shape, d, q, r, n_points, tau)
              for shape in ORDEN_CALIBRACION
              for d in dimensiones(shape)
              for q in qs
              for r in range(n_rep)]

    trabajadores = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 2) - 2)
    if trabajadores == 1:
        filas = [_replica(t) for t in tareas]
    else:
        with ProcessPoolExecutor(max_workers=trabajadores) as pool:
            filas = list(pool.map(_replica, tareas, chunksize=8))

    replicas = (pd.DataFrame(filas)
                .sort_values(["nube", "d", "q", "replica"])
                .reset_index(drop=True))
    return {
        "replicas": replicas,
        "q0": margenes(tau, n_points=n_points),
        "tau": tau,
    }


def guardar_resultados(resultado: Dict,
                       ruta_replicas: Path = RUTA_REPLICAS,
                       ruta_q0: Path = RUTA_Q0) -> None:
    """Escribe los tableros de la Fig. 5.2 en CSV (caché para la app y el CLI)."""
    ruta_replicas.parent.mkdir(parents=True, exist_ok=True)
    resultado["replicas"].to_csv(ruta_replicas, index=False)
    resultado["q0"].to_csv(ruta_q0, index=False)


def cargar_resultados(ruta_replicas: Path = RUTA_REPLICAS,
                      ruta_q0: Path = RUTA_Q0) -> Dict | None:
    """Lee la corrida cacheada de la Fig. 5.2, o `None` si todavía no existe."""
    if not (ruta_replicas.exists() and ruta_q0.exists()):
        return None
    return {
        "replicas": pd.read_csv(ruta_replicas),
        "q0": pd.read_csv(ruta_q0),
        "tau": TAU_CALIBRADO,
    }


def q_estrella(replicas: pd.DataFrame,
               umbral: float = UMBRAL_ACEPTABLE) -> pd.DataFrame:
    """Mayor q con tasa de acierto ≥ `umbral`, por nube y dimensión.

    Si ni el nivel de ruido más bajo alcanza el umbral, q* = 0 (la lectura ya
    falla sin ruido), que es el caso del toro en dimensión nativa.
    """
    filas = []
    for (nube, d), g in replicas.groupby(["nube", "d"]):
        medias = g.groupby("q")["acierto"].mean()
        validos = [q for q, a in medias.items() if a >= umbral]
        filas.append({"nube": nube, "d": int(d),
                      "q_estrella": float(max(validos)) if validos else 0.0})
    return pd.DataFrame(filas).sort_values(["nube", "d"]).reset_index(drop=True)


def resumen_razones(replicas: pd.DataFrame) -> pd.DataFrame:
    """Razón de ajuste dB/(2·dH) por dimensión: media y máximo.

    Es lo que reporta el texto del Cap. V (0,31 nativa / 0,25 en d=10 /
    0,36 en d=100) y lo que dibuja el panel (c).
    """
    r = replicas.copy()
    r["razon"] = np.where(r["cota_teorema"] > 0,
                          r["dB"] / r["cota_teorema"].replace(0, np.nan), np.nan)
    agrupado = r.groupby("d")["razon"].agg(medio="mean", maximo="max")
    return agrupado.reset_index()


__all__ = [
    "N_FIG52", "R_FIG52", "DELTA_FIG52", "QS_FIG52", "DIMS_EXTRA",
    "UMBRAL_ACEPTABLE", "dimensiones", "diagramas_finitos",
    "cota_probabilistica", "protocolo_fig52", "q_estrella", "resumen_razones",
    "RUTA_REPLICAS", "RUTA_Q0", "guardar_resultados", "cargar_resultados",
]
