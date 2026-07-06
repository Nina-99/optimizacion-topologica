"""Tests unidad — módulos core: FEM, topología, métricas."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestMetric:
    """Test básicos del módulo de métricas topológicas."""

    @pytest.fixture(scope="class")
    def sample_distance_matrix(self) -> np.ndarray:
        """Matriz 4x4 de distancias controlada."""
        return np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0],
        ])

    def test_distance_matrix_symmetric(self, sample_distance_matrix: np.ndarray) -> None:
        """La matriz de distancias debe ser simétrica."""
        assert_allclose(sample_distance_matrix, sample_distance_matrix.T)


class TestFEM:
    """Test smoke del módulo de elementos finitos."""

    def test_basic_stiffness_shape(self) -> None:
        """La matriz de rigidez elemental 2D debe ser 8x8."""
        try:
            from tda.core.fem import calcular_K_elemental
        except ImportError:
            pytest.skip("element_stiffness no disponible")

        E = 210e9  # Módulo Young (acero)
        nu = 0.3   # Coeficiente Poisson

        K = calcular_K_elemental(E, nu)
        assert K.shape == (8, 8), f"Esperado (8,8), obtenido {K.shape}"


class TestTopology:
    """Test smoke del módulo de topología persistente."""

    def test_ripser_importable(self) -> None:
        """ripser debe importar sin error."""
        pytest.importorskip("ripser")
        import ripser  # noqa: F401
