"""Tests del módulo de métricas — kmeans_accuracy y verify_betti.

Verifica que:
  - compute_kmeans_accuracy maneja etiquetas intercambiadas
  - verify_betti_numbers detecta correcto/incorrecto
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.analysis.metrics import compute_kmeans_accuracy, verify_betti_numbers


class TestKMeansAccuracy:
    """Accuracy de K-Means con manejo de etiquetas intercambiadas."""

    def test_perfect_match(self):
        """Labels idénticos deben dar accuracy=1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert acc == 1.0

    def test_perfect_swap(self):
        """Labels intercambiados (0↔1) deben dar accuracy=1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert acc == 1.0

    def test_partial_correct(self):
        """50% correcto sin swap debe dar accuracy=0.5."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert acc == 0.5

    def test_partial_with_swap(self):
        """50% correcto que mejora con swap debe dar el max."""
        # Sin swap: 2/4 = 0.5; con swap: 2/4 = 0.5 (igual)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 0])
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert acc == 0.5

    def test_all_wrong(self):
        """100% incorrecto sin swap da accuracy=0 con swap=1."""
        # Con swap: 1 - [0,0,1,1] = [1,1,0,0] vs [1,1,0,0] → perfect!
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert acc == 1.0

    def test_single_cluster(self):
        """Un solo clúster (todos iguales) debe dar accuracy=1.0."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert acc == 1.0

    def test_returns_float(self):
        """El retorno debe ser un float."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert isinstance(acc, float)

    def test_larger_arrays(self):
        """Arrays más grandes deben funcionar."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_pred = rng.randint(0, 2, 100)
        acc = compute_kmeans_accuracy(y_true, y_pred)
        assert 0.0 <= acc <= 1.0


class TestVerifyBettiNumbers:
    """Verificación de números de Betti esperados."""

    def test_correct_betti(self):
        """Esfera β₀=1,β₁=0 y toro β₀=1,β₁=2 debe dar tda_correct=True."""
        result = verify_betti_numbers(b0_s=1, b1_s=0, b0_t=1, b1_t=2)
        assert result["tda_correct"] is True
        assert "✓" in result["message"]

    def test_incorrect_sphere_beta1(self):
        """β₁ de esfera ≠ 0 debe dar tda_correct=False."""
        result = verify_betti_numbers(b0_s=1, b1_s=1, b0_t=1, b1_t=2)
        assert result["tda_correct"] is False
        assert result["b1_esfera_ok"] is False
        assert "✗" in result["message"]

    def test_incorrect_torus_beta1(self):
        """β₁ de toro ≠ 2 debe dar tda_correct=False."""
        result = verify_betti_numbers(b0_s=1, b1_s=0, b0_t=1, b1_t=1)
        assert result["tda_correct"] is False
        assert result["b1_toro_ok"] is False

    def test_both_betti_wrong(self):
        """Ambos β₁ incorrectos debe dar tda_correct=False."""
        result = verify_betti_numbers(b0_s=1, b1_s=3, b0_t=1, b1_t=5)
        assert result["tda_correct"] is False
        assert result["b1_esfera_ok"] is False
        assert result["b1_toro_ok"] is False

    def test_b0_stable(self):
        """β₀=1 para ambas formas debe dar b0_stable=True."""
        result = verify_betti_numbers(b0_s=1, b1_s=0, b0_t=1, b1_t=2)
        assert result["b0_stable"] is True

    def test_b0_unstable(self):
        """β₀ diferente debe dar b0_stable=False."""
        result = verify_betti_numbers(b0_s=2, b1_s=0, b0_t=1, b1_t=2)
        assert result["b0_stable"] is False

    def test_returns_dict_with_all_keys(self):
        """El dict debe contener las 5 claves esperadas."""
        result = verify_betti_numbers(b0_s=1, b1_s=0, b0_t=1, b1_t=2)
        expected_keys = {"tda_correct", "b0_stable", "b1_toro_ok",
                         "b1_esfera_ok", "message"}
        assert result.keys() == expected_keys
