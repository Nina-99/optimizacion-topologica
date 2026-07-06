"""Fixtures compartidas para tests del módulo analysis — nubes sintéticas.

Proporciona nubes de puntos 3D generadas con generate_cloud para
esfera y toro, con seed fija 42 (global conftest).
"""

import numpy as np
import pytest

from tda.processing.sampling import generate_cloud


@pytest.fixture(scope="module")
def n_points_small():
    """Número pequeño de puntos para tests rápidos."""
    return 50


@pytest.fixture(scope="module")
def nube_sphere(n_points_small):
    """Nube de puntos sintética de una esfera unitaria S².

    Retorna array (n_points_small, 3) con puntos en la superficie
    de la esfera de radio 1. Seed fija 42 via global conftest.
    """
    return generate_cloud("sphere", n_points_small)


@pytest.fixture(scope="module")
def nube_torus(n_points_small):
    """Nube de puntos sintética de un toro T² con R=2.0, r=1.0.

    Retorna array (n_points_small, 3) con puntos en la superficie
    del toro. Seed fija 42 via global conftest.
    """
    return generate_cloud("torus", n_points_small)


@pytest.fixture(scope="module")
def nube_sphere_large():
    """Nube más grande para tests de barrido de ruido."""
    return generate_cloud("sphere", 100)
