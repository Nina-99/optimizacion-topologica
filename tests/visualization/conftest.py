"""Fixtures compartidas para tests del módulo visualization — backend Agg headless.

Configura matplotlib en modo Agg (sin GUI) para entornos headless/CI.
Proporciona nubes de puntos 2D y 3D sintéticas con seed fija 42.
"""

import matplotlib

matplotlib.use("Agg")  # Forzar backend headless antes de importar pyplot

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _agg_backend():
    """Asegura backend Agg antes de cada test de visualización.

    Previene errores de display en entornos headless (CI, servidores, .exe).
    """
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    yield


@pytest.fixture(scope="module")
def points_2d():
    """Nube de 10 puntos 2D aleatorios con seed fija 42.

    Retorna array (10, 2) para tests de visualización 2D.
    """
    return np.random.rand(10, 2)


@pytest.fixture(scope="module")
def points_3d():
    """Nube de 15 puntos 3D aleatorios con seed fija 42.

    Retorna array (15, 3) para tests de visualización 3D.
    """
    return np.random.rand(15, 3)
