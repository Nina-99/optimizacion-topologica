"""Fixtures compartidas para tests del módulo core — malla 2×2."""

import numpy as np
import pytest

from tda.core.fem import calcular_K_elemental


@pytest.fixture(scope="module")
def nex():
    """Número de elementos en dirección x."""
    return 2


@pytest.fixture(scope="module")
def ney():
    """Número de elementos en dirección y."""
    return 2


@pytest.fixture(scope="module")
def N_e(nex, ney):
    """Número total de elementos."""
    return nex * ney


@pytest.fixture(scope="module")
def nnx(nex):
    """Número de nodos en dirección x."""
    return nex + 1


@pytest.fixture(scope="module")
def nny(ney):
    """Número de nodos en dirección y."""
    return ney + 1


@pytest.fixture(scope="module")
def n_dof(nnx, nny):
    """Número total de grados de libertad."""
    return 2 * nnx * nny


@pytest.fixture(scope="module")
def K0():
    """Matriz de rigidez elemental K_0 (8×8) con E=1.0, nu=0.3."""
    return calcular_K_elemental(E=1.0, nu=0.3)


@pytest.fixture(scope="module")
def DOFS(nex, ney):
    """Índices de DOF globales para cada elemento de la malla 2×2.

    Cada fila contiene los 8 DOF globales del elemento (u1, v1, u2, v2, ..., u4, v4).
    """
    N_e = nex * ney
    idx = np.arange(N_e)
    ey = idx // nex
    ex = idx % nex
    nnx = nex + 1
    n1 = ey * nnx + ex
    n2 = n1 + 1
    n3 = (ey + 1) * nnx + ex + 1
    n4 = (ey + 1) * nnx + ex
    DOFS = np.stack(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
         2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1], axis=1
    )
    return DOFS


@pytest.fixture(scope="module")
def F(n_dof):
    """Vector de fuerzas externas: fuerza vertical hacia abajo en nodo superior derecho.

    Malla 2×2 → nodos 0..8. Nodo 8 (top-right): DOF 16 (ux), DOF 17 (uy).
    Aplicamos F[17] = -1.0 (carga vertical descendente).
    """
    F = np.zeros(n_dof)
    F[17] = -1.0
    return F


@pytest.fixture(scope="module")
def fixed(nnx, nny):
    """Índices de DOF fijos: borde izquierdo (nodos 0, 3, 6).

    Cada nodo tiene DOF x (2*n) e y (2*n+1).
    Empotramiento: ux = uy = 0 en nodos 0, 3, 6.
    """
    # Nodos del borde izquierdo: (ex=0, ey=0..2)
    fixed_nodes = np.array([0, 3, 6], dtype=int)
    fixed_dofs = np.column_stack([2 * fixed_nodes, 2 * fixed_nodes + 1]).ravel()
    return fixed_dofs


@pytest.fixture(scope="module")
def p():
    """Exponente de penalización SIMP (típico p ≥ 3)."""
    return 3.0


@pytest.fixture(scope="module")
def rho0(N_e, f_V=0.5):
    """Densidades iniciales uniformes."""
    return np.full(N_e, f_V)


@pytest.fixture(scope="module")
def H(nex, ney):
    """Matriz de filtro de sensibilidad para malla 2×2 con r_min=2.4."""
    r_min = 2.4
    N_e = nex * ney
    cx = np.arange(N_e) % nex + 0.5
    cy = np.arange(N_e) // nex + 0.5
    dist = np.sqrt((cx[:, None] - cx)**2 + (cy[:, None] - cy)**2)
    return np.maximum(0.0, r_min - dist)


@pytest.fixture(scope="module")
def eps_star():
    """Fracción de volumen objetivo."""
    return 0.5
