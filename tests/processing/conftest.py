"""Fixtures compartidas para tests del módulo processing.

Proporciona puntos 3D de prueba y diagramas de persistencia sintéticos
para tests de sampling y preprocessing.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def puntos_3d():
    """Puntos 3D de prueba con geometría conocida.

    Cuatro puntos en las esquinas de un cuadrado en el plano XY (z=0):
      (0,0,0), (1,0,0), (1,1,0), (0,1,0)

    Diámetro = sqrt(2) ~ 1.414.
    """
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)


@pytest.fixture(scope="module")
def diagrama_sintetico():
    """Diagrama de persistencia sintético 2-dimensiones.

    Formato: lista de dimensiones, cada una con lista de [birth, death].
    - H0: 1 componente larga + 1 corta
    - H1: 1 agujero persistente
    """
    return [
        [[0.0, 3.0], [0.0, 0.5]],   # H0: (birth, death)
        [[1.0, 2.5]],                 # H1: 1 agujero
    ]


@pytest.fixture(scope="module")
def diagrama_vacio():
    """Diagrama de persistencia vacío (sin puntos en ninguna dimensión)."""
    return [[], []]


@pytest.fixture(scope="module")
def diagrama_histograma():
    """Diagrama 1D para tests de histograma: array (n_pairs, 2).

    Formato [birth, death]:
      (0.0, 3.0) persistencia 3.0
      (1.0, 2.0) persistencia 1.0
      (0.5, 0.7) persistencia 0.2
    """
    return np.array([
        [0.0, 3.0],
        [1.0, 2.0],
        [0.5, 0.7],
    ])
