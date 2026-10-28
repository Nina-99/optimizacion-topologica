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
    """Project points onto the diagonal y=x (nacimiento=muerte line).

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

    Parámetros:
        dgm1: Primer diagrama de forma (n, 2) con [nacimiento, muerte].
        dgm2: Segundo diagrama de forma (m, 2) con [nacimiento, muerte].

    Retorna:
        Distancia de Wasserstein W_1(dgm1, dgm2).
    """
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 debe tener forma (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 debe tener forma (m, 2)")

    # Handle empty diagrams
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0:
        diag2 = _points_to_diagonal(dgm2)
        return float(np.sum(np.linalg.norm(dgm2 - diag2, axis=1)))
    if len(dgm2) == 0:
        diag1 = _points_to_diagonal(dgm1)
        return float(np.sum(np.linalg.norm(dgm1 - diag1, axis=1)))

    # Sort by persistence (muerte - nacimiento), descending
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


def _bottleneck_factible(A: np.ndarray, B: np.ndarray, t: float,
                         D: np.ndarray) -> bool:
    """¿Existe un matching entre A y B con costo máximo ≤ t?

    Grafo bipartito cuadrado de (n + m) × (n + m):

      - Filas    0..n-1 → puntos de A;  filas n..n+m-1 → comodines `e_j` de B.
      - Columnas 0..m-1 → puntos de B;  columnas m..m+n-1 → comodines `f_i` de A.

    Aristas:
      `A_i → B_j` si `D[i,j] ≤ t`   (el par se empareja)
      `A_i → f_i` si `pers(A_i)/2 ≤ t`  (A_i va a la diagonal)
      `e_j → B_j` si `pers(B_j)/2 ≤ t`  (B_j va a la diagonal)
      `e_j → f_i` siempre (los comodines se "apartan" cuando su vértice se
      emparejó con el otro diagrama; sin esta arista el matching prohibiría
      todo emparejamiento real)

    Existe matching perfecto ⟺ el umbral `t` es factible. La factibilidad es
    monótona en `t`, lo que habilita la búsqueda binaria de
    `bottleneck_distance`.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    n, m = D.shape
    filas, columnas = [], []
    if n and m:
        r, c = np.nonzero(D <= t)
        filas.append(r)
        columnas.append(c)
    ia = np.nonzero((A[:, 1] - A[:, 0]) / 2.0 <= t)[0]
    filas.append(ia)
    columnas.append(m + ia)
    ib = np.nonzero((B[:, 1] - B[:, 0]) / 2.0 <= t)[0]
    filas.append(n + ib)
    columnas.append(ib)
    if n and m:
        e_j = np.repeat(np.arange(n, n + m), n)
        f_i = np.tile(np.arange(m, m + n), m)
        filas.append(e_j)
        columnas.append(f_i)
    f = np.concatenate(filas)
    c = np.concatenate(columnas)
    grafo = csr_matrix((np.ones(len(f), dtype=bool), (f, c)),
                       shape=(n + m, n + m))
    return bool((maximum_bipartite_matching(grafo, perm_type="row") >= 0).all())


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Distancia bottleneck exacta entre dos diagramas de persistencia.

    Definición estándar (Cohen-Steiner, Edelsbrunner & Harer 2007): mínimo,
    sobre las biyecciones entre `(A ∪ diagonal) y (B ∪ diagonal)`, del **máximo**
    costo L∞; el costo de mandar un punto a la diagonal es `persistencia / 2`.

    **Cuidado (trampa clásica):** el objetivo es **minimizar el máximo**, no la
    suma. Resolverlo con `scipy.optimize.linear_sum_assignment` (que minimiza
    la suma) devuelve el máximo de un matching *óptimo para la suma*, que es
    **mayor o igual** al bottleneck: sobreestima siempre y, en el 36 % de los
    diagramas de prueba, se desviaba hasta 0,27 (ver `tests/core/test_topology.py`).

    Implementación: búsqueda binaria sobre el conjunto finito de costos
    candidatos (todas las distancias L∞ entre puntos, más `persistencia/2`),
    comprobando cada umbral con un matching bipartito
    (`scipy.sparse.csgraph.maximum_bipartite_matching`). El resultado es el
    primer umbral factible, y como la respuesta pertenece al conjunto de
    candidatos, es **exacto**.

    Parámetros:
        dgm1: Primer diagrama de forma (n, 2) con [nacimiento, muerte].
        dgm2: Segundo diagrama de forma (m, 2) con [nacimiento, muerte].

    Retorna:
        Distancia bottleneck dB(dgm1, dgm2) con norma L∞.
    """
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 debe tener forma (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 debe tener forma (m, 2)")

    A = np.asarray(dgm1, dtype=float).reshape(-1, 2)
    B = np.asarray(dgm2, dtype=float).reshape(-1, 2)
    A = A[np.isfinite(A[:, 1])] if len(A) > 0 else A.reshape(0, 2)
    B = B[np.isfinite(B[:, 1])] if len(B) > 0 else B.reshape(0, 2)

    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(np.max((B[:, 1] - B[:, 0]) / 2.0))
    if m == 0:
        return float(np.max((A[:, 1] - A[:, 0]) / 2.0))

    D = np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    candidatos = np.unique(np.concatenate([
        D.ravel(), (A[:, 1] - A[:, 0]) / 2.0, (B[:, 1] - B[:, 0]) / 2.0,
    ]))
    if _bottleneck_factible(A, B, float(candidatos[0]), D):
        return float(candidatos[0])

    lo, hi = 0, len(candidatos) - 1      # invariante: candidatos[hi] factible
    while lo < hi:
        mid = (lo + hi) // 2
        if _bottleneck_factible(A, B, float(candidatos[mid]), D):
            hi = mid
        else:
            lo = mid + 1
    return float(candidatos[lo])


def betti_numbers(persistence_diagram: np.ndarray) -> Tuple[int, int]:
    """Extrae los números de Betti (β₀, β₁) de un diagrama de persistencia.

    Parámetros:
        persistence_diagram: Diagrama con forma (n, 3) donde cada fila es
            [nacimiento, muerte, dimensión].

    Retorna:
        Tuple[int, int]: (beta_0, beta_1)
    """
    if persistence_diagram.ndim != 2 or persistence_diagram.shape[1] != 3:
        raise ValueError(
            "persistence_diagram debe tener forma (n, 3) "
            "con [nacimiento, muerte, dimension]"
        )
    if not np.all(np.isin(persistence_diagram[:, 2], [0, 1])):
        raise ValueError("La columna de dimensión debe contener solo 0 (H_0) or 1 (H_1)")

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
