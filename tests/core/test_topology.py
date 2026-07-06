"""Tests del módulo de topología — Betti, binarización, escala, homología.

Verifica invariantes del análisis topológico de datos (TDA):
  - betti_numbers extrae β₀ y β₁ de diagramas sintéticos
  - binarizar_y_extraer_nube produce coordenadas de centroides sólidos
  - escala_adaptativa devuelve ε* > 0 acotado
  - calcular_homologia_betti skip graceful sin ripser
  - wasserstein/bottleneck_distance validan shape de inputs
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.core.topology import (
    betti_numbers,
    binarizar_y_extraer_nube,
    bottleneck_distance,
    calcular_homologia_betti,
    escala_adaptativa,
    wasserstein_distance,
)


class TestBettiNumbers:
    """Extracción de números de Betti de diagramas de persistencia."""

    def test_betti_no_holes(self):
        """Diagrama sin puntos debe dar (0, 0)."""
        dgm = np.zeros((0, 3))
        beta0, beta1 = betti_numbers(dgm)
        assert beta0 == 0
        assert beta1 == 0

    def test_betti_two_components_one_hole(self):
        """Diagrama con 2 componentes H_0 y 1 agujero H_1."""
        dgm = np.array([
            [0.0, 1.0, 0],   # H_0: nace 0, muere 1
            [0.0, 2.0, 0],   # H_0: nace 0, muere 2
            [0.5, 2.0, 1],   # H_1: nace 0.5, muere 2
        ])
        beta0, beta1 = betti_numbers(dgm)
        assert beta0 == 2, f"Esperado β₀=2, obtenido {beta0}"
        assert beta1 == 1, f"Esperado β₁=1, obtenido {beta1}"

    def test_betti_only_h0(self):
        """Diagrama solo con puntos H_0 debe dar β₁=0."""
        dgm = np.array([
            [0.0, 0.5, 0],
            [0.0, 1.0, 0],
        ])
        beta0, beta1 = betti_numbers(dgm)
        assert beta0 == 2
        assert beta1 == 0

    def test_betti_invalid_shape_raises(self):
        """Shape incorrecto debe lanzar ValueError."""
        with pytest.raises(ValueError, match="shape"):
            betti_numbers(np.array([[0.0, 1.0]]))  # shape (1,2) en vez de (n,3)

    def test_betti_invalid_dimension_raises(self):
        """Dimensión fuera de {0,1} debe lanzar ValueError."""
        dgm = np.array([[0.0, 1.0, 2]])  # dim 2 no es válida
        with pytest.raises(ValueError, match="Dimension"):
            betti_numbers(dgm)

    def test_betti_inf_death_counts_as_feature(self):
        """Puntos H_0 con muerte Inf cuentan como componente."""
        dgm = np.array([
            [0.0, np.inf, 0],
            [0.0, 0.5, 1],
        ])
        beta0, beta1 = betti_numbers(dgm)
        assert beta0 == 1, f"Esperado β₀=1, obtenido {beta0}"
        assert beta1 == 1


class TestBinarizar:
    """Binarización de diseño SIMP y extracción de nube de puntos."""

    def test_binarize_all_solid(self):
        """Todos los elementos sólidos (>0.5) deben aparecer en la nube."""
        rho = np.ones(4)
        nube = binarizar_y_extraer_nube(rho, 2, 2)
        assert len(nube) == 4, (
            f"Esperados 4 puntos sólidos, obtenidos {len(nube)}"
        )

    def test_binarize_none_solid(self):
        """Sin elementos sólidos (ρ ≤ 0.5) la nube debe estar vacía."""
        rho = np.full(4, 0.3)
        nube = binarizar_y_extraer_nube(rho, 2, 2)
        assert len(nube) == 0, (
            f"Esperada nube vacía, obtenidos {len(nube)} puntos"
        )

    def test_binarize_coordinates(self):
        """Los centroides deben estar en [0.5, 1.5) para malla 2×2."""
        rho = np.array([1.0, 0.0, 0.0, 0.0])
        nube = binarizar_y_extraer_nube(rho, 2, 2)
        assert len(nube) == 1
        assert_allclose(nube[0], [0.5, 0.5], atol=1e-10)

    def test_binarize_custom_threshold(self):
        """Umbral personalizado debe cambiar qué elementos se consideran sólidos."""
        rho = np.array([0.6, 0.4])
        nube_alto = binarizar_y_extraer_nube(rho, 2, 1, umbral=0.55)
        nube_bajo = binarizar_y_extraer_nube(rho, 2, 1, umbral=0.35)
        assert len(nube_alto) == 1
        assert len(nube_bajo) == 2


class TestEscala:
    """Escala adaptativa ε* para filtración Vietoris-Rips."""

    def test_scale_single_point(self):
        """Un solo punto debe devolver 1.0 (default mínimo)."""
        eps = escala_adaptativa(np.array([[0.5, 0.5]]), 4)
        assert eps == 1.0

    def test_scale_two_points(self):
        """Dos puntos separados: ε* = distancia / √N_e."""
        nube = np.array([[0.0, 0.0], [1.0, 0.0]])
        eps = escala_adaptativa(nube, 4)
        expected = max(np.sqrt(1.0) / np.sqrt(4.0), 0.1)
        assert_allclose(eps, expected, atol=1e-10)

    def test_scale_lower_bound(self):
        """ε* no debe ser menor que 0.1."""
        nube = np.array([[0.0, 0.0], [0.01, 0.0]])
        eps = escala_adaptativa(nube, 1000)
        assert eps >= 0.1 - 1e-12, f"ε* = {eps} < 0.1"

    def test_scale_larger_domain(self):
        """Nube más dispersa debe dar ε* mayor."""
        nube_chica = np.array([[0.0, 0.0], [1.0, 0.0]])
        nube_grande = np.array([[0.0, 0.0], [5.0, 0.0]])
        eps_chico = escala_adaptativa(nube_chica, 4)
        eps_grande = escala_adaptativa(nube_grande, 4)
        assert eps_grande > eps_chico, (
            "Nube más dispersa debe producir ε* mayor"
        )


class TestHomologia:
    """Homología persistente H₁ con Ripser (opcional)."""

    def test_homologia_graceful_empty(self):
        """Menos de 3 puntos no necesita ripser y retorna (0, vacío)."""
        nube = np.array([[0.5, 0.5], [1.5, 0.5]])
        beta1, dgm1 = calcular_homologia_betti(nube, 0.5)
        assert beta1 == 0
        assert dgm1.shape == (0, 2)

    def test_homologia_few_points(self):
        """Menos de 3 puntos debe dar (0, dgm vacío) sin llamar a ripser."""
        nube = np.array([[0.5, 0.5], [1.5, 0.5]])
        beta1, dgm1 = calcular_homologia_betti(nube, 0.5)
        assert beta1 == 0
        assert dgm1.shape == (0, 2)

    def test_homologia_ripser_importable(self):
        """ripser debe ser importable (dependencia opcional)."""
        pytest.importorskip("ripser")
        import ripser  # noqa: F401


class TestDistanceErrors:
    """Validación de errores en funciones de distancia."""

    def test_1d_array_raises(self):
        """Input 1D debe lanzar ValueError o ImportError (si persim ausente)."""
        with pytest.raises((ValueError, ImportError)):
            wasserstein_distance(np.array([0.0, 1.0]),
                                 np.array([[0.0, 1.0]]))

    def test_wrong_shape_raises(self):
        """Shape incorrecto debe lanzar ValueError (o ImportError sin persim)."""
        with pytest.raises((ValueError, ImportError)):
            wasserstein_distance(
                np.array([[0.0, 1.0, 2.0]]),
                np.array([[0.0, 1.0]]),
            )

    def test_bottleneck_wrong_shape_raises(self):
        """Shape 1D debe lanzar ValueError (o ImportError sin persim)."""
        with pytest.raises((ValueError, ImportError)):
            bottleneck_distance(
                np.array([[0.0, 1.0], [0.5, 1.5]]),
                np.array([0.0, 1.0]),  # 1D
            )
