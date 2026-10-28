"""Calibración del umbral de significación τ (Figura 5.1, Cap. V).

El umbral τ se fija ANTES de medir, sobre un catálogo independiente C₀, y
después se mide el margen de cada nube en el catálogo de medición C (sin
ruido). Ese orden es lo que acredita H.E.1c: τ no se eligió a posteriori
para hacer pasar la medición.

Fórmulas (Def. 8.1 del Perfil):

    βₖ^(τ)(X) = #{(b, d) ∈ Dgmₖ(X) : d − b > τ · diam(X)}
    β₀^(τ)    = 1 + #{barras finitas de H₀ con persistencia > τ · diam(X)}
                (la componente esencial nunca muere)

    A(τ)      = #{nubes de C₀ con β₀ y β₁ correctos}
    τ*        = min{τ : A(τ) = 7}            # ventana válida, si existe

    margen    = min( min(rasgos reales) − τ* ,  τ* − max(rasgos espurios) )

Rasgo real: barra que DEBE sobrevivir al umbral (β teórico).
Rasgo espurio: barra que DEBE morir bajo el umbral (β₀ o β₁ excedente).

Protocolo:
    C₀ = generar_corregida(nube, 400, default_rng(900 + i))   # sin ruido
    C  = generar_corregida(nube, 400, default_rng(42 + i))    # sin ruido
    τ ∈ {0,050, 0,075, ..., 0,400}  (grilla de 15 puntos, paso 0,025)

Referencias:
  - §5 metodología: calibración del umbral sobre catálogo independiente.
  - `tda.processing.sampling.generar_corregida`: geometría exacta del catálogo.
  - `tda.analysis.stability.betti_persistente`: implementación de la Def. 8.1.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from ripser import ripser

from tda.analysis.stability import betti_persistente
from tda.processing.sampling import (
    CATALOGO_CORREGIDO,
    ETIQUETA_NUBE,
    ORDEN_CALIBRACION,
    compute_diameter,
    generar_corregida,
)

# ── Protocolo (números del documento) ────────────────────────────────
N_CALIBRACION = 400          # puntos por nube
SEED_C0 = 900                # catálogo de calibración C₀ (semillas 900 + i)
SEED_MEDICION = 42           # catálogo de medición C (semillas 42 + i)
TAUS = np.round(np.arange(0.05, 0.401, 0.025), 3)   # 15 puntos, paso 0,025

# τ adoptado por todo el protocolo (Fig. 5.1 → Figs. 5.2/5.3): es el τ* de C₀,
# única ventana con 7/7 aciertos de (β₀, β₁). Vive acá para que la calibración
# y las mediciones no puedan desincronizarse.
TAU_CALIBRADO = 0.175


# ═══════════════════════════════════════════════════════════════════
# Piezas puras
# ═══════════════════════════════════════════════════════════════════

def betti_tau(dgm, tau: float, diam: float) -> tuple:
    """(β₀, β₁) a umbral τ · diam(X) — Def. 8.1.

    `dgm` es la salida de ripser (dict o lista de diagramas). Delega en
    `betti_persistente` para que la fórmula viva en un solo lugar.
    """
    return betti_persistente(dgm, tau, diam)


def persistencias_relativas(dgm, diam: float):
    """Persistencias / diam(X) de H₀ y H₁, ordenadas de mayor a menor.

    Excluye la barra infinita de H₀: su persistencia no es finita y
    arrastraría los márgenes del panel (c) a ±inf (su +1 en β₀ lo aporta
    `betti_persistente`, no este cálculo).
    """
    if isinstance(dgm, dict) and "dgms" in dgm:
        dgms = dgm["dgms"]
    else:
        dgms = dgm
    h0 = np.asarray(dgms[0]) if len(dgms) > 0 else np.empty((0, 2))
    if h0.size and h0.ndim >= 2:
        h0 = h0[np.isfinite(h0[:, 1])]
    h1 = np.asarray(dgms[1]) if len(dgms) > 1 else np.empty((0, 2))
    if h1.size and h1.ndim >= 2:
        h1 = h1[np.isfinite(h1[:, 1])]

    def _orden(h):
        if h.size == 0:
            return np.array([], dtype=float)
        return np.sort(h[:, 1] - h[:, 0])[::-1] / diam

    return _orden(h0), _orden(h1)


# ═══════════════════════════════════════════════════════════════════
# Catálogo de calibración C₀
# ═══════════════════════════════════════════════════════════════════

def calibrar_c0(
    n_points: int = N_CALIBRACION,
    seed: int = SEED_C0,
    taus: Sequence[float] = TAUS,
    catalogo: Sequence[str] = ORDEN_CALIBRACION,
) -> pd.DataFrame:
    """Aciertos de (β₀, β₁) en C₀ para cada τ de la grilla.

    Cada nube se muestrea una sola vez por nube (semilla `seed + i`) y se
    relee en los 15 umbrales: el diagrama no depende de τ, solo el conteo.
    """
    filas = []
    for i, shape in enumerate(catalogo):
        b0_teo, b1_teo = CATALOGO_CORREGIDO[shape]
        X = generar_corregida(shape, n_points, np.random.default_rng(seed + i))
        diam = compute_diameter(X)
        # Salida CRUDA de ripser: la barra infinita de H₀ es la que aporta el
        # +1 de β₀ en betti_persistente. Filtrarla antes rompería la lectura.
        dgm = ripser(X, maxdim=1)
        for tau in taus:
            b0, b1 = betti_tau(dgm, tau, diam)
            filas.append(dict(
                nube=shape, tau=float(tau), b0=int(b0), b1=int(b1),
                b0_teo=int(b0_teo), b1_teo=int(b1_teo),
                correcto=int(b0 == b0_teo and b1 == b1_teo),
            ))
    return pd.DataFrame(filas)


def curva_calibracion(cal: pd.DataFrame) -> pd.Series:
    """A(τ): nº de nubes de C₀ con (β₀, β₁) correctos, indexado por τ."""
    return cal.groupby("tau").correcto.sum().sort_index()


def tau_estrella(cal: pd.DataFrame) -> Dict[str, float]:
    """τ* = primer τ donde C₀ acierta en las 7 nubes (ventana única).

    Raises:
        ValueError: si no existe ningún τ con lectura consistente. La
            calibración falló: hay que revisar el catálogo, no elegir otro τ.
    """
    A = curva_calibracion(cal)
    ventana = A[A == len(cal.nube.unique())].index.values
    if len(ventana) == 0:
        raise ValueError(
            "ABORTAR: el catálogo C₀ no admite lectura consistente de (β₀, β₁); "
            "no existe τ con las 7 nubes correctas."
        )
    return {
        "tau_star": float(ventana.min()),
        "ventana_min": float(ventana.min()),
        "ventana_max": float(ventana.max()),
    }


# ═══════════════════════════════════════════════════════════════════
# Catálogo de medición C — márgenes respecto de τ*
# ═══════════════════════════════════════════════════════════════════

def margenes(
    tau_star: float,
    n_points: int = N_CALIBRACION,
    seed: int = SEED_MEDICION,
    catalogo: Sequence[str] = ORDEN_CALIBRACION,
) -> pd.DataFrame:
    """Margen de cada nube de C (sin ruido) respecto de τ*.

    margen = min( min(rasgos reales) − τ*, τ* − max(rasgos espurios) ).

    margen > 0: la lectura es robusta a ruido. margen < 0: la nube se lee mal
    ya sin ruido (p. ej. el toro, cuyo meridiano queda justo bajo τ*).
    """
    filas = []
    for i, shape in enumerate(catalogo):
        b0_teo, b1_teo = CATALOGO_CORREGIDO[shape]
        X = generar_corregida(shape, n_points, np.random.default_rng(seed + i))
        diam = compute_diameter(X)
        dgm = ripser(X, maxdim=1)
        b0, b1 = betti_tau(dgm, tau_star, diam)
        # Para los márgenes sí se descarta la barra infinita: su persistencia
        # no es finita y arrastraría el margen a ±inf.
        p0, p1 = persistencias_relativas(dgm, diam)

        # Rasgo real más débil: el último que debe sobrevivir a τ*.
        reales = list(p0[: b0_teo - 1]) + list(p1[:b1_teo])
        # Rasgo espurio más fuerte: el primero que debe morir bajo τ*.
        espurios = [
            p0[b0_teo - 1] if len(p0) >= b0_teo else 0.0,
            p1[b1_teo] if len(p1) > b1_teo else 0.0,
        ]
        margen_inferior = (min(reales) - tau_star) if reales else np.nan
        margen_superior = tau_star - max(espurios)
        filas.append(dict(
            nube=shape, diam=diam, b0=int(b0), b1=int(b1),
            b0_teo=int(b0_teo), b1_teo=int(b1_teo),
            acierto=int(b0 == b0_teo and b1 == b1_teo),
            rasgo_real_mas_debil=(min(reales) if reales else np.nan),
            espurio_mas_fuerte=max(espurios),
            margen_inferior=margen_inferior,
            margen_superior=margen_superior,
            margen=float(np.fmin(np.inf if np.isnan(margen_inferior) else margen_inferior,
                                 margen_superior)),
        ))
    return pd.DataFrame(filas)


# ═══════════════════════════════════════════════════════════════════
# Pipeline completo (lo que consume la Figura 5.1)
# ═══════════════════════════════════════════════════════════════════

def resultado_calibracion(
    n_points: int = N_CALIBRACION,
    seed_c0: int = SEED_C0,
    seed_medicion: int = SEED_MEDICION,
    taus: Sequence[float] = TAUS,
) -> Dict:
    """Ejecuta C₀ → τ* → C y devuelve todo lo que necesita la Figura 5.1.

    Returns:
        dict con `cal` (DataFrame de aciertos), `tau` (dict con tau_star y
        ventana), `curva` (Serie A(τ)) y `margenes` (DataFrame por nube).
    """
    cal = calibrar_c0(n_points=n_points, seed=seed_c0, taus=taus)
    tau = tau_estrella(cal)
    return {
        "cal": cal,
        "tau": tau,
        "curva": curva_calibracion(cal),
        "margenes": margenes(tau["tau_star"], n_points=n_points, seed=seed_medicion),
    }


__all__ = [
    "N_CALIBRACION", "SEED_C0", "SEED_MEDICION", "TAUS",
    "CATALOGO_CORREGIDO", "ORDEN_CALIBRACION", "ETIQUETA_NUBE",
    "betti_tau", "persistencias_relativas",
    "calibrar_c0", "curva_calibracion", "tau_estrella", "margenes",
    "resultado_calibracion",
]
