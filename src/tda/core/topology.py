"""Modulo de funciones topologicas para analisis TDA-SIMP.

Provee funciones para:
- Distancias entre diagramas de persistencia (Wasserstein, Bottleneck)
- Extraccion de numeros de Betti (beta_0, beta_1)
- Binarizacion de disenos SIMP y extraccion de nubes de puntos
- Escala adaptativa epsilon* para filtracion Vietoris-Rips
- Homologia persistente H_1 con Ripser

Bloque 3 del Algoritmo 1 (Metrica Compuesta TDA-SIMP).

Implementaciones puras en numpy (sin dependencia de persim).
"""

import numpy as np
from typing import Tuple


# =============================================================================
# DISTANCIAS ENTRE DIAGRAMAS (numpy puro, sin persim)
# =============================================================================

def _points_to_diagonal(points: np.ndarray) -> np.ndarray:
    """Project points onto the diagonal y=x (birth=death line).

    For a point (b, d), the closest point on the diagonal is ((b+d)/2, (b+d)/2).
    """
    mid = (points[:, 0] + points[:, 1]) / 2.0
    return np.column_stack([mid, mid])


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between sets a (n,d) and b (m,d)."""
    # ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i . b_j
    a_sq = np.sum(a ** 2, axis=1)[:, np.newaxis]
    b_sq = np.sum(b ** 2, axis=1)[np.newaxis, :]
    dist_sq = a_sq + b_sq - 2.0 * np.dot(a, b.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Wasserstein (q=1) entre dos diagramas de persistencia.

    Implementacion pura numpy: empareja puntos del diagrama mas corto con
    proyecciones sobre la diagonal del mas largo, ordenando por persistencia.

    Args:
        dgm1: Primer diagrama de forma (n, 2) con [birth, death].
        dgm2: Segundo diagrama de forma (m, 2) con [birth, death].

    Returns:
        Distancia de Wasserstein W_1(dgm1, dgm2).
    """
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")

    # Handle empty diagrams
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0:
        diag2 = _points_to_diagonal(dgm2)
        return float(np.sum(np.linalg.norm(dgm2 - diag2, axis=1)))
    if len(dgm2) == 0:
        diag1 = _points_to_diagonal(dgm1)
        return float(np.sum(np.linalg.norm(dgm1 - diag1, axis=1)))

    # Sort by persistence (death - birth), descending
    pers1 = dgm1[:, 1] - dgm1[:, 0]
    pers2 = dgm2[:, 1] - dgm2[:, 0]
    order1 = np.argsort(-pers1)
    order2 = np.argsort(-pers2)
    dgm1_sorted = dgm1[order1]
    dgm2_sorted = dgm2[order2]

    # Pad shorter diagram with diagonal projections
    n, m = len(dgm1_sorted), len(dgm2_sorted)
    max_len = max(n, m)

    if n < max_len:
        diag2_proj = _points_to_diagonal(dgm2_sorted[n:])
        dgm1_padded = np.vstack([dgm1_sorted, diag2_proj])
    else:
        dgm1_padded = dgm1_sorted

    if m < max_len:
        diag1_proj = _points_to_diagonal(dgm1_sorted[m:])
        dgm2_padded = np.vstack([dgm2_sorted, diag1_proj])
    else:
        dgm2_padded = dgm2_sorted

    # W_1 = sum of pointwise distances
    diff = dgm1_padded[:max_len] - dgm2_padded[:max_len]
    return float(np.sum(np.sqrt(np.sum(diff ** 2, axis=1))))


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Bottleneck entre dos diagramas de persistencia.

    Implementacion pura numpy: para cada punto de ambos diagramas, calcula
    la minima distancia a cualquier punto del otro diagrama o a la diagonal.
    Retorna el maximo de todas esas minimas distancias.

    Args:
        dgm1: Primer diagrama de forma (n, 2) con [birth, death].
        dgm2: Segundo diagrama de forma (m, 2) con [birth, death].

    Returns:
        Distancia de Bottleneck W_inf(dgm1, dgm2).
    """
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")

    # Handle empty diagrams
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0:
        diag2 = _points_to_diagonal(dgm2)
        return float(np.max(np.linalg.norm(dgm2 - diag2, axis=1)))
    if len(dgm2) == 0:
        diag1 = _points_to_diagonal(dgm1)
        return float(np.max(np.linalg.norm(dgm1 - diag1, axis=1)))

    # Distance from each point in dgm1 to nearest in dgm2 or diagonal
    d_1_to_2 = _pairwise_distances(dgm1, dgm2).min(axis=1)
    diag2 = _points_to_diagonal(dgm1)
    d_1_to_diag = np.linalg.norm(dgm1 - diag2, axis=1)
    min_dist_1 = np.minimum(d_1_to_2, d_1_to_diag)

    # Distance from each point in dgm2 to nearest in dgm1 or diagonal
    d_2_to_1 = _pairwise_distances(dgm2, dgm1).min(axis=1)
    diag2_pts = _points_to_diagonal(dgm2)
    d_2_to_diag = np.linalg.norm(dgm2 - diag2_pts, axis=1)
    min_dist_2 = np.minimum(d_2_to_1, d_2_to_diag)

    return float(max(min_dist_1.max(), min_dist_2.max()))


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

    Soporta nubes de puntos en任意 dimensión (2D, 3D, etc.).

    Parámetros
    ──────────
    nube : ndarray (n_s, d)  Centroides de elementos sólidos (d dimensiones)
    N_e  : int               Número total de elementos de la malla

    Retorna
    ───────
    eps_star : float  Escala de filtración
    """
    if len(nube) < 2:
        return 1.0

    # Diagonal del bounding box (funciona para任意 dimensión)
    mins = nube.min(axis=0)
    maxs = nube.max(axis=0)
    diam = np.sqrt(np.sum((maxs - mins)**2))

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
