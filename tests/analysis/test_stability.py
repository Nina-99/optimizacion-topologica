"""Tests del módulo de estabilidad — noise_sweep retorno y callback progreso.

Verifica que compute_noise_sweep:
  - Retorna un dict con las claves esperadas y tipos correctos
  - Llama al callback de progreso con (paso, total) después de cada iteración
  - Produce noise_vals correctamente espaciados
  - Los números de Betti son tuplas de enteros
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("ripser")

from tda.analysis.stability import compute_noise_sweep


class TestNoiseSweepReturn:
    """Estructura de retorno de compute_noise_sweep."""

    def test_return_is_dict(self):
        """compute_noise_sweep debe retornar un dict."""
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.2,
                                     n_steps=3)
        assert isinstance(result, dict)

    def test_return_has_expected_keys(self):
        """El dict debe contener las 6 claves esperadas."""
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.2,
                                     n_steps=3)
        expected_keys = {"noise_vals", "acc", "betti_s",
                         "betti_t", "diagrams_s", "diagrams_t"}
        assert expected_keys.issubset(result.keys()), (
            f"Faltan claves: {expected_keys - result.keys()}"
        )

    def test_noise_vals_shape(self):
        """noise_vals debe tener shape (n_steps,) con valores en [noise_min, noise_max]."""
        n_steps = 5
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.3,
                                     n_steps=n_steps)
        assert result["noise_vals"].shape == (n_steps,)
        assert result["noise_vals"][0] == 0.0
        assert_allclose(result["noise_vals"][-1], 0.3, atol=1e-10)

    def test_noise_vals_linspace(self):
        """noise_vals debe espaciarse linealmente (linspace)."""
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.4,
                                     n_steps=4)
        expected = np.linspace(0.0, 0.4, 4)
        assert_allclose(result["noise_vals"], expected, atol=1e-10)

    def test_acc_is_list_of_floats(self):
        """acc debe ser una lista de floats de longitud n_steps."""
        n_steps = 3
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.2,
                                     n_steps=n_steps)
        assert isinstance(result["acc"], list)
        assert len(result["acc"]) == n_steps
        for val in result["acc"]:
            assert isinstance(val, float), "acc debe contener floats"

    def test_acc_in_range(self):
        """acc debe estar en [0, 1] (accuracy)."""
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.2,
                                     n_steps=3)
        for val in result["acc"]:
            assert 0.0 <= val <= 1.0, f"acc={val} fuera de [0, 1]"

    def test_betti_are_lists_of_tuples(self):
        """betti_s y betti_t deben ser listas de tuplas (int, int)."""
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.2,
                                     n_steps=3)
        assert isinstance(result["betti_s"], list)
        assert isinstance(result["betti_t"], list)
        assert len(result["betti_s"]) == 3
        assert len(result["betti_t"]) == 3
        for b_s, b_t in zip(result["betti_s"], result["betti_t"]):
            assert isinstance(b_s, tuple) and len(b_s) == 2
            assert isinstance(b_t, tuple) and len(b_t) == 2
            assert isinstance(b_s[0], (int, np.integer))
            assert isinstance(b_s[1], (int, np.integer))
            assert isinstance(b_t[0], (int, np.integer))
            assert isinstance(b_t[1], (int, np.integer))

    def test_diagrams_are_lists(self):
        """diagrams_s y diagrams_t deben ser listas de longitud n_steps."""
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.2,
                                     n_steps=3)
        assert isinstance(result["diagrams_s"], list)
        assert isinstance(result["diagrams_t"], list)
        assert len(result["diagrams_s"]) == 3
        assert len(result["diagrams_t"]) == 3


class TestNoiseSweepCallback:
    """Callback de progreso en compute_noise_sweep."""

    def test_callback_called_correct_times(self):
        """El callback debe llamarse n_steps veces."""
        calls = []

        def cb(current, total):
            calls.append((current, total))

        compute_noise_sweep(n_points=10, n_clusters=2,
                            noise_min=0.0, noise_max=0.2,
                            n_steps=4, progress_callback=cb)
        assert len(calls) == 4, (
            f"Callback llamado {len(calls)} veces, esperado 4"
        )

    def test_callback_receives_current_and_total(self):
        """Cada llamada debe recibir (paso, total) con paso 1..n_steps."""
        calls = []

        def cb(current, total):
            calls.append((current, total))

        compute_noise_sweep(n_points=10, n_clusters=2,
                            noise_min=0.0, noise_max=0.2,
                            n_steps=4, progress_callback=cb)
        for i, (current, total) in enumerate(calls):
            assert current == i + 1, (
                f"Paso {current} no coincide con posición {i + 1}"
            )
            assert total == 4

    def test_callback_none_does_not_raise(self):
        """progress_callback=None no debe lanzar error."""
        result = compute_noise_sweep(n_points=10, n_clusters=2,
                                     noise_min=0.0, noise_max=0.2,
                                     n_steps=3, progress_callback=None)
        assert result is not None
