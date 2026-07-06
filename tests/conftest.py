"""Configuración global de tests — seed fija para reproducibilidad."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_random():
    """Fija la semilla aleatoria antes de cada test para resultados deterministas."""
    np.random.seed(42)
