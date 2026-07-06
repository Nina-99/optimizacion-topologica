"""Tests del pipeline TDA — run_tda_experiment, estructura y tipos de resultados.

Verifica que run_tda_experiment con n_rep=1:
  - Retorna dict con un key por nivel de ruido
  - Cada entry contiene las 8 estadísticas esperadas
  - Todos los valores son floats finitos
  - Con n_rep=1, std debe ser 0.0
  - La función maneja shapes conocidas (sphere, torus, cube)
"""

import numpy as np
import pytest

from tda.simulation.pipeline import run_tda_experiment, _safe_stack


class TestRunTdaExperiment:
    """Estructura y tipos de retorno de run_tda_experiment.

    Nota: torus solía fallar por un bug de np.vstack con dimensiones mixtas
    (H0 con puntos, H1 vacío). Se corrigió con _safe_stack en pipeline.py.
    """

    # -- Constantes de test --
    N_POINTS = 20
    N_REP = 1
    NOISE = [0.10]

    def test_returns_dict(self):
        """El retorno debe ser un dict."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        assert isinstance(result, dict)

    def test_dict_keys_match_noise_levels(self):
        """Las keys del dict deben coincidir con noise_levels."""
        levels = [0.05, 0.10, 0.15]
        result = run_tda_experiment("sphere", noise_levels=levels,
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        assert set(result.keys()) == {0.05, 0.10, 0.15}

    def test_single_noise_level(self):
        """Un solo nivel de ruido debe producir un solo key."""
        result = run_tda_experiment("sphere", noise_levels=[0.10],
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        assert list(result.keys()) == [0.10]

    def test_expected_stat_keys(self):
        """Cada noise level debe tener las 8 estadísticas esperadas."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        expected = {
            "betti0_mean", "betti0_std",
            "betti1_mean", "betti1_std",
            "wasserstein_mean", "wasserstein_std",
            "bottleneck_mean", "bottleneck_std",
        }
        assert expected == set(result[0.10].keys()), (
            f"Keys mismatch. Missing: {expected - set(result[0.10].keys())}"
        )

    def test_all_values_are_floats(self):
        """Todos los valores estadísticos deben ser float."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        for key, val in result[0.10].items():
            assert isinstance(val, float), (
                f"{key} debe ser float, got {type(val)}"
            )

    def test_all_values_finite(self):
        """Todos los valores deben ser finitos (no NaN, no Inf)."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        for key, val in result[0.10].items():
            assert np.isfinite(val), f"{key} no es finito: {val}"

    def test_std_zero_with_one_rep(self):
        """Con n_rep=1, todas las std deben ser 0.0."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=1, n_points=self.N_POINTS)
        for noise_key in result:
            assert result[noise_key]["betti0_std"] == 0.0
            assert result[noise_key]["betti1_std"] == 0.0
            assert result[noise_key]["wasserstein_std"] == 0.0
            assert result[noise_key]["bottleneck_std"] == 0.0

    def test_betti_counts_nonnegative(self):
        """Los valores de Betti deben ser ≥ 0."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        for noise_key in result:
            stats = result[noise_key]
            assert stats["betti0_mean"] >= 0.0
            assert stats["betti1_mean"] >= 0.0

    def test_wasserstein_and_bottleneck_nonnegative(self):
        """Las distancias Wasserstein y Bottleneck deben ser ≥ 0."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=self.N_REP, n_points=self.N_POINTS)
        for noise_key in result:
            stats = result[noise_key]
            assert stats["wasserstein_mean"] >= 0.0
            assert stats["wasserstein_std"] >= 0.0
            assert stats["bottleneck_mean"] >= 0.0
            assert stats["bottleneck_std"] >= 0.0

    def test_multiple_shapes(self):
        """Debe funcionar con sphere y cube (torus tiene bug pre-existente)."""
        for shape in ["sphere", "cube"]:
            result = run_tda_experiment(shape, noise_levels=[0.10],
                                        n_rep=self.N_REP,
                                        n_points=self.N_POINTS)
            assert isinstance(result, dict)
            assert 0.10 in result

    def test_custom_seed(self):
        """Seed personalizada debe producir resultados sin error."""
        result = run_tda_experiment("sphere", noise_levels=self.NOISE,
                                    n_rep=self.N_REP, n_points=self.N_POINTS,
                                    seed=123)
        assert isinstance(result[0.10], dict)

    def test_torus_mixed_dimensions(self):
        """Torus no debe fallar con diagramas de dimensión mixta.

        Con threshold=1.5 y 20 puntos, es probable que H1 quede vacío
        mientras H0 retiene puntos, lo que disparaba el bug de np.vstack
        con dimensión mixta (lista con elementos 2D y []).
        """
        result = run_tda_experiment("torus", noise_levels=[0.10],
                                    n_rep=1, n_points=20)
        assert isinstance(result, dict)
        assert 0.10 in result
        expected = {
            "betti0_mean", "betti0_std",
            "betti1_mean", "betti1_std",
            "wasserstein_mean", "wasserstein_std",
            "bottleneck_mean", "bottleneck_std",
        }
        assert expected == set(result[0.10].keys())
        for val in result[0.10].values():
            assert np.isfinite(val)


class TestSafeStack:
    """Tests unitarios para _safe_stack — helper que filtra dimensiones vacías."""

    def test_all_non_empty_returns_vstack(self):
        """Todas las dimensiones con puntos devuelve np.vstack normal."""
        dims = [[[0.1, 0.5], [0.2, 0.6]], [[0.3, 0.7]]]
        result = _safe_stack(dims)
        expected = np.vstack([np.array([[0.1, 0.5], [0.2, 0.6]]),
                              np.array([[0.3, 0.7]])])
        np.testing.assert_array_equal(result, expected)

    def test_all_empty_returns_empty(self):
        """Todas las dimensiones vacías devuelve np.empty((0, 2))."""
        result = _safe_stack([[], []])
        assert result.shape == (0, 2)

    def test_mixed_empty_and_non_empty(self):
        """Dimensión mixta (H0 con puntos, H1 vacío) debe funcionar."""
        dims = [[[0.1, 0.5], [0.2, 0.6]], []]
        result = _safe_stack(dims)
        expected = np.array([[0.1, 0.5], [0.2, 0.6]])
        np.testing.assert_array_equal(result, expected)

    def test_single_non_empty_dim(self):
        """Una sola dimensión no vacía debe funcionar."""
        dims = [[], [[0.1, 0.5]]]
        result = _safe_stack(dims)
        expected = np.array([[0.1, 0.5]])
        np.testing.assert_array_equal(result, expected)

    def test_empty_input_list(self):
        """Lista vacía (sin dimensiones) debe devolver empty((0, 2))."""
        result = _safe_stack([])
        assert result.shape == (0, 2)

    def test_preserves_order(self):
        """El orden de las dimensiones debe preservarse."""
        dims = [[[0.3, 0.7]], [], [[0.1, 0.5], [0.2, 0.6]]]
        result = _safe_stack(dims)
        expected = np.array([[0.3, 0.7], [0.1, 0.5], [0.2, 0.6]])
        np.testing.assert_array_equal(result, expected)

    def test_single_point_in_dim(self):
        """Una dimensión con un solo punto (1, 2) debe funcionar."""
        dims = [[[0.5, 1.5]]]
        result = _safe_stack(dims)
        expected = np.array([[0.5, 1.5]])
        np.testing.assert_array_equal(result, expected)
