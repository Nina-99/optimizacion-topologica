"""Fixtures compartidas para tests del módulo simulation — config pipeline.

Proporciona parámetros mínimos para ejecutar run_tda_experiment
con repetición única y tamaño de nube pequeño.
"""

import pytest


@pytest.fixture(scope="module")
def n_points_small():
    """Número pequeño de puntos para tests rápidos de pipeline."""
    return 10


@pytest.fixture(scope="module")
def noise_levels_small():
    """Dos niveles de ruido para verificar estructura multi-key."""
    return [0.05, 0.10]
