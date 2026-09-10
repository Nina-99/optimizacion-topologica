"""Cálculo de β₀ y β₁ del diseño SIMP por dos vías independientes.

Implementa el Instrumento del §8.5 del Perfil y la §2.2 de la metodología:
  Método A — Característica de Euler (conteo celular + etiquetado 4-conexo)
  Método B — Homología cúbica persistente (GUDHI CubicalComplex)

La coincidencia de ambos métodos valida el indicador antes de usarlo
para contrastar H.E.2b.

Referencias:
  - Perfil §8.5: "Instrumento para el cálculo de β₁: homología cúbica"
  - Perfil §7.3: χ = V − E + F, H₂ = 0 en R² → β₁ = β₀ − χ
  - metodologia_implementacion §2.2: Método A y B
  - Listing 1 de la metodología: betti_cubico()
"""

import numpy as np
from scipy import ndimage


def _conteo_celular(S):
    """Cuenta vértices, aristas y caras del complejo cúbico de S.

    Cada celda sólida S[i,j] = True aporta 4 vértices y 4 aristas,
    pero vértices y aristas compartidos entre celdas adyacentes se
    cuentan una sola vez.

    Parámetros
    ──────────
    S : ndarray bool (ny, nx)  Grid binario del diseño (True = sólido)

    Retorna
    ───────
    V : int  Número de vértices únicos
    E : int  Número de aristas únicas
    F : int  Número de caras (celdas sólidas)
    """
    F = int(np.sum(S))
    if F == 0:
        return 0, 0, 0

    # Vértices: cada celda (i,j) tiene vértices en (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    vertices = set()
    ys, xs = np.where(S)
    for y, x in zip(ys, xs):
        vertices.add((y, x))
        vertices.add((y + 1, x))
        vertices.add((y, x + 1))
        vertices.add((y + 1, x + 1))
    V = len(vertices)

    # Aristas: cada celda (i,j) tiene 4 aristas
    aristas = set()
    for y, x in zip(ys, xs):
        # Horizontal superior: (y,x)-(y,x+1)
        aristas.add(((y, x), (y, x + 1)))
        # Horizontal inferior: (y+1,x)-(y+1,x+1)
        aristas.add(((y + 1, x), (y + 1, x + 1)))
        # Vertical izquierda: (y,x)-(y+1,x)
        aristas.add(((y, x), (y + 1, x)))
        # Vertical derecha: (y,x+1)-(y+1,x+1)
        aristas.add(((y, x + 1), (y + 1, x + 1)))
    E = len(aristas)

    return V, E, F


def betti_euler(S):
    """Método A: β₀ y β₁ por característica de Euler.

    Procedimiento (§8.5):
      1. β₀ por etiquetado de componentes 4-conexas (ndimage.label)
      2. χ = V − E + F por conteo celular del complejo cúbico
      3. Como Ω ⊂ R² implica H₂ = 0:  β₁ = β₀ − χ

    Parámetros
    ──────────
    S : ndarray bool (ny, nx)  Grid binario del diseño (True = sólido)

    Retorna
    ───────
    beta0 : int  Número de componentes conexas
    beta1 : int  Número de ciclos 1-dimensionales (agujeros)
    """
    S = S.astype(bool)
    if not S.any():
        return 0, 0

    # β₀: etiquetado de componentes 4-conexas
    estructura = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=bool)
    _, beta0 = ndimage.label(S, structure=estructura)

    # Característica de Euler
    V, E, F = _conteo_celular(S)
    chi = V - E + F

    # H₂ = 0 en R² → β₁ = β₀ − χ
    beta1 = beta0 - chi

    return int(beta0), int(beta1)


def betti_gudhi(S, filtracion=0.5):
    """Método B: β₀ y β₁ por homología cúbica persistente (GUDHI).

    Construye un CubicalComplex con filtración binaria:
      f = 0 en el sólido, f = 1 en el vacío.
    Lee β₀, β₁ del subnivel f < filtracion (= 0.5 por defecto).

    Parámetros
    ──────────
    S          : ndarray bool (ny, nx)  Grid binario (True = sólido)
    filtracion : float  Umbral de subnivel (default 0.5)

    Retorna
    ───────
    beta0 : int  Número de componentes conexas
    beta1 : int  Número de ciclos 1-dimensionales (agujeros)
    """
    try:
        import gudhi
    except ImportError:
        raise ImportError(
            "gudhi >= 3.13 es requerido para betti_gudhi(). "
            "Instalar con: pip install gudhi"
        )

    S = S.astype(bool)
    if not S.any():
        return 0, 0

    # Filtración: 0 = sólido, 1 = vacío
    grid = np.where(S, 0.0, 1.0)

    cc = gudhi.CubicalComplex(top_dimensional_cells=grid)
    cc.compute_persistence()

    # β₀ y β₁ en el subnivel [0, filtracion)
    betti = cc.persistent_betti_numbers(0, filtracion)
    beta0 = int(betti[0]) if len(betti) > 0 else 0
    beta1 = int(betti[1]) if len(betti) > 1 else 0

    return beta0, beta1


def diagramas_gudhi(S):
    """Extrae los diagramas de persistencia H₀ y H₁ de GUDHI.

    Retorna los diagramas como arrays numpy para visualización.

    Parámetros
    ──────────
    S : ndarray bool (ny, nx)  Grid binario (True = sólido)

    Retorna
    ───────
    dgm0 : ndarray (n0, 2)  Diagrama H₀  [birth, death]
    dgm1 : ndarray (n1, 2)  Diagrama H₁  [birth, death]
    """
    try:
        import gudhi
    except ImportError:
        raise ImportError("gudhi >= 3.13 requerido")

    S = S.astype(bool)
    if not S.any():
        return np.empty((0, 2)), np.empty((0, 2))

    grid = np.where(S, 0.0, 1.0)
    cc = gudhi.CubicalComplex(top_dimensional_cells=grid)
    cc.compute_persistence()

    # gudhi retorna (dimension, birth, death)
    persistence = cc.persistence()
    dgm0_list = []
    dgm1_list = []
    for dim, (b, d) in persistence:
        if dim == 0:
            dgm0_list.append([b, d])
        elif dim == 1:
            dgm1_list.append([b, d])

    dgm0 = np.array(dgm0_list) if dgm0_list else np.empty((0, 2))
    dgm1 = np.array(dgm1_list) if dgm1_list else np.empty((0, 2))
    return dgm0, dgm1


def betti_doble_computo(S, verbose=False):
    """Calcula β₀, β₁ por ambos métodos y verifica concordancia.

    Parámetros
    ──────────
    S       : ndarray bool (ny, nx)  Grid binario (True = sólido)
    verbose : bool  Mostrar diagnóstico en consola

    Retorna
    ───────
    dict con:
        beta0_euler : int   β₀ por Método A
        beta1_euler : int   β₁ por Método A
        beta0_gudhi : int   β₀ por Método B
        beta1_gudhi : int   β₁ por Método B
        concordancia: bool  True si ambos métodos coinciden
        V, E, F    : int   Conteo celular (para auditoría)
        chi         : int   Característica de Euler
    """
    S = S.astype(bool)

    # Método A: Euler
    beta0_a, beta1_a = betti_euler(S)
    V, E, F = _conteo_celular(S)
    chi = V - E + F

    # Método B: GUDHI
    beta0_b, beta1_b = betti_gudhi(S)

    concordancia = (beta0_a == beta0_b) and (beta1_a == beta1_b)

    if verbose:
        status = "✓ CONCORDANCIA" if concordancia else "✗ DISCREPANCIA"
        print(f"  Método A (Euler):  β₀={beta0_a}, β₁={beta1_a}  "
              f"[V={V}, E={E}, F={F}, χ={chi}]")
        print(f"  Método B (GUDHI):  β₀={beta0_b}, β₁={beta1_b}")
        print(f"  {status}")

    return {
        "beta0_euler": beta0_a,
        "beta1_euler": beta1_a,
        "beta0_gudhi": beta0_b,
        "beta1_gudhi": beta1_b,
        "concordancia": concordancia,
        "V": V, "E": E, "F": F,
        "chi": chi,
    }
