"""Tests del módulo de preprocesamiento — filter, normalize, histograma.

Verifica que:
  - filter_persistence_diagram elimina intervalos bajo threshold
  - normalize_diagram escala coordenadas por diámetro
  - get_persistence_histogram cuenta vidas en bins correctos
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.processing.preprocessing import (
    filter_persistence_diagram,
    get_persistence_histogram,
    normalize_diagram,
)


class TestFilterPersistenceDiagram:
    """Filtrado de intervalos de persistencia cortos."""

    def test_filter_keeps_long_intervals(self):
        """Intervalos con persistencia ≥ threshold deben conservarse."""
        dgm = [[[0.0, 3.0], [0.0, 0.5]], [[1.0, 2.5]]]
        filtered = filter_persistence_diagram(dgm, 1.0)
        # 0.0→3.0 (pers=3.0 ≥ 1.0) ✓
        # 0.0→0.5 (pers=0.5 < 1.0) ✗
        # 1.0→2.5 (pers=1.5 ≥ 1.0) ✓
        assert len(filtered[0]) == 1, "Solo el intervalo largo debe conservarse en H0"
        assert filtered[0][0] == [0.0, 3.0]
        assert len(filtered[1]) == 1, "El agujero debe conservarse en H1"

    def test_filter_threshold_zero(self):
        """Threshold 0 debe conservar todos los intervalos."""
        dgm = [[[0.0, 3.0], [0.0, 0.5]], [[1.0, 2.5]]]
        filtered = filter_persistence_diagram(dgm, 0.0)
        assert len(filtered[0]) == 2
        assert len(filtered[1]) == 1

    def test_filter_very_high_threshold(self):
        """Threshold mayor que toda persistencia debe eliminar todo."""
        dgm = [[[0.0, 3.0], [0.0, 0.5]], [[1.0, 2.5]]]
        filtered = filter_persistence_diagram(dgm, 10.0)
        assert filtered == [[], []]

    def test_filter_empty_diagram(self):
        """Diagrama vacío debe retornar lista vacía por dimensión."""
        filtered = filter_persistence_diagram([[], []], 0.5)
        assert filtered == [[], []]

    def test_filter_negative_threshold(self):
        """Threshold negativo debe conservar todos."""
        dgm = [[[0.0, 0.3]]]
        filtered = filter_persistence_diagram(dgm, -1.0)
        assert len(filtered[0]) == 1

    def test_filter_exact_threshold(self):
        """Intervalo con persistencia exactamente igual al threshold debe conservarse."""
        dgm = [[[0.0, 1.0], [0.0, 0.999]]]
        filtered = filter_persistence_diagram(dgm, 1.0)
        assert len(filtered[0]) == 1, "Persistencia exacta debe conservarse"
        assert filtered[0][0] == [0.0, 1.0]


class TestNormalizeDiagram:
    """Normalización de diagramas de persistencia."""

    def test_normalize_by_diameter(self):
        """Coordenadas deben escalarse por 1/diámetro."""
        dgm = [[[0.0, 2.0], [0.0, 0.5]], [[1.0, 1.5]]]
        normalized = normalize_diagram(dgm, 2.0)
        # 0.0/2=0.0, 2.0/2=1.0; 0.0/2=0.0, 0.5/2=0.25; 1.0/2=0.5, 1.5/2=0.75
        assert_allclose(normalized[0][0], [0.0, 1.0], atol=1e-10)
        assert_allclose(normalized[0][1], [0.0, 0.25], atol=1e-10)
        assert_allclose(normalized[1][0], [0.5, 0.75], atol=1e-10)

    def test_normalize_empty_diagram(self):
        """Diagrama vacío debe retornar lista vacía por dimensión."""
        normalized = normalize_diagram([[], []], 2.0)
        assert normalized == [[], []]

    def test_normalize_zero_diameter(self):
        """Diámetro cero debe usar factor de seguridad (1e-8) para evitar división por cero."""
        dgm = [[[1.0, 2.0]]]
        normalized = normalize_diagram(dgm, 0.0)
        # factor = 1.0 / 1e-8 = 1e8
        assert_allclose(normalized[0][0][0], 1.0e8, atol=1e-5)
        assert_allclose(normalized[0][0][1], 2.0e8, atol=1e-5)

    def test_normalize_preserves_proportions(self):
        """La relación entre valores debe preservarse tras normalizar."""
        dgm = [[[1.0, 3.0], [2.0, 6.0]]]
        normalized = normalize_diagram(dgm, 2.0)
        # Valores normalizados: [0.5, 1.5] y [1.0, 3.0]
        # Relación: antes 1/3 = 0.333, después 0.5/1.5 = 0.333 ✓
        ratio_before = 1.0 / 3.0
        ratio_after = normalized[0][0][0] / normalized[0][0][1]
        assert_allclose(ratio_before, ratio_after, atol=1e-10)

    def test_normalize_multi_dimension(self):
        """Múltiples dimensiones deben normalizarse con el mismo factor."""
        dgm = [[[0.0, 4.0]], [[2.0, 6.0]], [[1.0, 3.0]]]
        normalized = normalize_diagram(dgm, 2.0)
        # H0: [0.0, 2.0]
        # H1: [1.0, 3.0]
        # H2: [0.5, 1.5]
        assert_allclose(normalized[0][0], [0.0, 2.0], atol=1e-10)
        assert_allclose(normalized[1][0], [1.0, 3.0], atol=1e-10)
        assert_allclose(normalized[2][0], [0.5, 1.5], atol=1e-10)


class TestGetPersistenceHistogram:
    """Histograma de vidas de persistencia."""

    def test_histogram_known_values(self):
        """Histograma con vidas conocidas debe dar bins correctos."""
        dgm = np.array([[3.0, 0.0], [2.0, 1.0], [0.7, 0.5]])
        # lifetimes: 3.0-0.0=3.0, 2.0-1.0=1.0, 0.7-0.5=0.2
        # max_life = 3.0, rango (0, 3.0), 3 bins: [0,1), [1,2), [2,3]
        # 3.0 → bin 2, 1.0 → bin 1, 0.2 → bin 0
        hist = get_persistence_histogram(dgm, 3)
        assert hist[0] == 1, "0.2 debe caer en bin 0"
        assert hist[1] == 1, "1.0 debe caer en bin 1"
        assert hist[2] == 1, "3.0 debe caer en bin 2"

    def test_histogram_empty(self):
        """Diagrama vacío debe retornar arreglo de ceros."""
        dgm = np.array([]).reshape(0, 2)
        hist = get_persistence_histogram(dgm, 5)
        assert hist.shape == (5,)
        assert np.all(hist == 0)

    def test_histogram_custom_bins(self):
        """Número de bins personalizado debe respetarse."""
        dgm = np.array([[1.0, 0.0]])
        hist = get_persistence_histogram(dgm, 10)
        assert hist.shape == (10,)

    def test_histogram_default_bins(self):
        """Sin especificar bins, debe usar 10."""
        dgm = np.array([[1.0, 0.0]])
        hist = get_persistence_histogram(dgm)
        assert hist.shape == (10,)

    def test_histogram_single_lifetime(self):
        """Una sola vida debe caer en el último bin (max_life = lifetime)."""
        dgm = np.array([[5.0, 0.0]])
        hist = get_persistence_histogram(dgm, 5)
        # lifetime = 5.0, max_life = 5.0, bins 5, range (0, 5)
        # 5.0 cae en el último bin (índice 4)
        assert hist[-1] == 1, "La única vida debe estar en el último bin"
        assert np.sum(hist) == 1, "Solo debe haber 1 cuenta"

    def test_histogram_returns_ndarray(self):
        """El retorno debe ser un ndarray."""
        dgm = np.array([[1.0, 0.0]])
        hist = get_persistence_histogram(dgm)
        assert isinstance(hist, np.ndarray)
