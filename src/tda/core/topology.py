"""Módulo de funciones topológicas para análisis TDA-SIMP.

Provee funciones para:
- Distancias entre diagramas de persistencia (Wasserstein, Bottleneck)
- Extracción de números de Betti (β₀, β₁)
- Binarización de diseños SIMP y extracción de nubes de puntos
- Escala adaptativa ε* para filtración Vietoris-Rips
- Homología persistente H₁ con Ripser

Bloque 3 del Algoritmo 1 (Métrica Compuesta TDA-SIMP).
"""

import numpy as np
from typing import Tuple

try:
    from persim import wasserstein_distance as persim_wasserstein
    from persim import bottleneck_distance as persim_bottleneck
except ImportError:
    persim_wasserstein = None
    persim_bottleneck = None


# =============================================================================
# FUNCIONES EXISTENTES (Distancias entre diagramas)
# =============================================================================

def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Wasserstein entre dos diagramas de persistencia."""
    if persim_wasserstein is None:
        raise ImportError("persim library is required for wasserstein_distance")
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")
    return float(persim_wasserstein(dgm1, dgm2))


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Bottleneck entre dos diagramas de persistencia."""
    if persim_bottleneck is None:
        raise ImportError("persim library is required for bottleneck_distance")
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")
    return float(persim_bottleneck(dgm1, dgm2))


def betti_numbers(persistence_diagram: np.ndarray) -> Tuple[int, int]:
    """Extrae los números de Betti (β₀, β₁) de un diagrama de persistencia.

    Args:
        persistence_diagram: Diagrama con forma (n, 3) donde cada fila es
            [nacimiento, muerte, dimensión].

    Returns:
        Tuple[int, int]: (beta_0, beta_1)
    """
    if persistence_diagram.ndim != 2 or persistence_diagram.shape[1] != 3:
        raise ValueError(
            "persistence_diagram must be of shape (n, 3) "
            "with [birth, death, dimension]"
        )
    if not np.all(np.isin(persistence_diagram[:, 2], [0, 1])):
        raise ValueError("Dimension column must contain only 0 (H_0) or 1 (H_1)")

    h0 = persistence_diagram[persistence_diagram[:, 2] == 0]
    h1 = persistence_diagram[persistence_diagram[:, 2] == 1]
    beta_0 = int(h0.shape[0])
    beta_1 = int(h1.shape[0])
    return (beta_0, beta_1)


# =============================================================================
# NUEVAS FUNCIONES TDA (Binarización, escala, homología)
# =============================================================================

def binarizar_y_extraer_nube(rho, nex, ney, umbral=0.5):
    """
    Binariza la solución SIMP y extrae la nube de puntos del diseño.

    La nube X(ρ*) son los centroides de los elementos sólidos:
        X(ρ*) = { c_e ∈ R²  :  ρ_e > 0.5 }

    Parámetros
    ──────────
    rho    : ndarray (N_e,)  Densidades continuas en [0, 1]
    nex    : int             Número de elementos en x
    ney    : int             Número de elementos en y
    umbral : float           Umbral de binarización (defecto 0.5)

    Retorna
    ───────
    nube : ndarray (n_s, 2)  Centroides de los elementos sólidos [x_e, y_e]
    """
    N_e = nex * ney
    idx = np.arange(N_e)
    cx = (idx % nex) + 0.5
    cy = (idx // nex) + 0.5

    mask = rho > umbral
    nube = np.stack([cx[mask], cy[mask]], axis=1)
    return nube


def escala_adaptativa(nube, N_e):
    """
    Calcula la escala adaptativa ε* para la filtración Vietoris-Rips:
        ε* = diam(X) / √N_e

    Parámetros
    ──────────
    nube : ndarray (n_s, 2)  Centroides de elementos sólidos
    N_e  : int               Número total de elementos de la malla

    Retorna
    ───────
    eps_star : float  Escala de filtración
    """
    if len(nube) < 2:
        return 1.0

    xmin, ymin = nube.min(axis=0)
    xmax, ymax = nube.max(axis=0)
    diam = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)

    return max(diam / np.sqrt(float(N_e)), 0.1)


def calcular_homologia_betti(nube, eps_star):
    """
    Calcula la homología persistente H_1 con Ripser sobre la nube del diseño
    y extrae el número de Betti β₁ (agujeros 1-dimensionales significativos).

    Un punto (b_i, d_i) en Dgm_1 es significativo si:
        d_i - b_i > umbral_pers = ε*/2

    Parámetros
    ──────────
    nube     : ndarray (n_s, 2)  Centroides de elementos sólidos
    eps_star : float             Escala adaptativa de la filtración

    Retorna
    ───────
    beta1 : int              Número de Betti β₁
    dgm1  : ndarray (k, 2)   Diagrama de persistencia H_1 (puntos (b_i, d_i))
    """
    if len(nube) < 3:
        return 0, np.zeros((0, 2))

    from ripser import ripser
    resultado = ripser(nube, maxdim=1, thresh=4.0 * eps_star)
    dgm1 = resultado["dgms"][1]

    if len(dgm1) == 0:
        return 0, dgm1

    umbral_pers = 0.5 * eps_star

    beta1 = 0
    for b_i, d_i in dgm1:
        if np.isinf(d_i):
            beta1 += 1
        elif (d_i - b_i) > umbral_pers:
            beta1 += 1

    return int(beta1), dgm1
