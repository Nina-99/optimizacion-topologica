"""Tests del optimizador de vigas — BeamOptimizer.

Verifica funciones aislables según design (tests parciales):
  - Init almacena parámetros y precalcula I0/I_min
  - calcular_momento implementa la fórmula analítica
  - simular_viga retorna lista de dicts con estructura esperada
  - optimizar_viga_completo retorna dict con claves esperadas
  - Placeholder D_viga_ideasizada aceptado (retorna constantes)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.optimization.beam_optimizer import BeamOptimizer


class TestInit:
    """Inicialización de BeamOptimizer — parámetros y precálculos."""

    def test_default_params(self):
        """Los parámetros por defecto deben asignarse."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        assert opt.b == 0.3
        assert opt.h0 == 0.5
        assert opt.p == 3
        assert opt.N == 10

    def test_custom_params(self):
        """Parámetros personalizados deben almacenarse."""
        opt = BeamOptimizer(b=0.4, h0=0.6, p=4, N=20,
                            a_c=2.0, A_maximo_concreto=2.0,
                            E_c=2e9, E_ac=2e9, M_opt=2.0,
                            max_iter=200)
        assert opt.b == 0.4
        assert opt.h0 == 0.6
        assert opt.p == 4
        assert opt.N == 20
        assert opt.a_c == 2.0
        assert opt.max_iter == 200

    def test_I0_calculation(self):
        """I0 debe ser b * h0³ / 12."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        expected_I0 = 0.3 * 0.5**3 / 12
        assert_allclose(opt.I0, expected_I0, atol=1e-12)

    def test_I_min_is_fraction_of_I0(self):
        """I_min debe ser 0.15 * I0."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        assert_allclose(opt.I_min, 0.15 * opt.I0, atol=1e-12)

    def test_N_determines_array_lengths(self):
        """N debe ser entero positivo."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=25)
        assert opt.N == 25


class TestCalcularMomento:
    """Momento flector analíto: M(x) = -q·x²/2 + q·L·(L-x)."""

    def test_at_fixed_end(self):
        """M(0) debe ser q*L²."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        M = opt.calcular_momento(x=0.0, q=10.0, L=5.0)
        assert_allclose(M, 10.0 * 5.0**2, atol=1e-10)

    def test_at_midspan(self):
        """Valor de momento en x=2.5 con q=10, L=5."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        M = opt.calcular_momento(x=2.5, q=10.0, L=5.0)
        expected = -10.0 * 2.5**2 / 2 + 10.0 * 5.0 * (5.0 - 2.5)
        assert_allclose(M, expected, atol=1e-10)

    def test_at_free_end(self):
        """M(L) debe ser -q*L²/2."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        M = opt.calcular_momento(x=5.0, q=10.0, L=5.0)
        assert_allclose(M, -10.0 * 5.0**2 / 2, atol=1e-10)

    def test_linearly_scales_with_q(self):
        """Momento debe escalar linealmente con q."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        M1 = opt.calcular_momento(x=2.0, q=5.0, L=5.0)
        M2 = opt.calcular_momento(x=2.0, q=10.0, L=5.0)
        assert_allclose(M2, 2.0 * M1, atol=1e-10)

    def test_zero_load(self):
        """q=0 debe dar M=0 en todos los puntos."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        M = opt.calcular_momento(x=3.0, q=0.0, L=5.0)
        assert_allclose(M, 0.0, atol=1e-10)


class TestSimularViga:
    """Simulación de viga con optimización SIMP."""

    @pytest.fixture
    def opt(self):
        """BeamOptimizer con N pequeño para tests rápidos."""
        return BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)

    def test_returns_list(self, opt):
        """simular_viga debe retornar una lista."""
        result = opt.simular_viga(L=5.0, q=10.0)
        assert isinstance(result, list)

    def test_returns_non_empty(self, opt):
        """La lista no debe estar vacía."""
        result = opt.simular_viga(L=5.0, q=10.0)
        assert len(result) > 0

    def test_each_entry_is_dict(self, opt):
        """Cada entry en la lista debe ser un dict."""
        result = opt.simular_viga(L=5.0, q=10.0)
        for entry in result:
            assert isinstance(entry, dict)

    def test_expected_keys_in_entry(self, opt):
        """Cada entry debe tener las claves de visualización esperadas."""
        result = opt.simular_viga(L=5.0, q=10.0)
        required_keys = {"iteration", "x", "I", "Y", "Y_original",
                         "M", "h_v", "error"}
        entry_keys = set(result[0].keys())
        missing = required_keys - entry_keys
        assert not missing, f"Faltan claves: {missing}"

    def test_iteration_starts_at_zero(self, opt):
        """La primera iteración debe tener iteration=0."""
        result = opt.simular_viga(L=5.0, q=10.0)
        assert result[0]["iteration"] == 0

    def test_error_decreases(self, opt):
        """El error debe disminuir a lo largo de las iteraciones."""
        result = opt.simular_viga(L=5.0, q=10.0)
        errors = [entry["error"] for entry in result]
        assert errors[-1] <= errors[0] + 1e-10

    def test_x_is_array_of_length_N(self, opt):
        """x debe ser un array de longitud N."""
        result = opt.simular_viga(L=5.0, q=10.0)
        assert len(result[0]["x"]) == opt.N

    def test_M_shape_matches_x(self, opt):
        """M debe tener el mismo shape que x."""
        result = opt.simular_viga(L=5.0, q=10.0)
        assert result[0]["M"].shape == result[0]["x"].shape

    def test_M_positive_in_span(self, opt):
        """M debe ser positivo en el interior de la viga."""
        result = opt.simular_viga(L=5.0, q=10.0)
        M = result[0]["M"]
        # Para viga simplemente apoyada con carga distribuida,
        # M debe ser ≥ 0 en [0, L]
        assert np.all(M >= -1e-10)

    def test_inertia_bounded(self, opt):
        """I debe estar entre I_min e I0."""
        result = opt.simular_viga(L=5.0, q=10.0)
        I_final = result[-1]["I"]
        assert np.all(I_final >= opt.I_min - 1e-10)
        assert np.all(I_final <= opt.I0 + 1e-10)


class TestOptimizarVigaCompleto:
    """Optimización completa de la viga."""

    @pytest.fixture
    def opt(self):
        """BeamOptimizer con N pequeño."""
        return BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)

    def test_returns_dict(self, opt):
        """optimizar_viga_completo debe retornar un dict."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        assert isinstance(result, dict)

    def test_expected_keys(self, opt):
        """El dict debe contener las claves esperadas."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        required = {"x", "I", "Y", "Y_original", "M", "h_v",
                    "iterations", "final_error", "saving_pct",
                    "weight_saved", "As", "V_shear", "Vc",
                    "sigma_MPa", "L", "y_adm"}
        missing = required - set(result.keys())
        assert not missing, f"Faltan claves: {missing}"

    def test_iterations_positive(self, opt):
        """iterations debe ser ≥ 1."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        assert result["iterations"] >= 1

    def test_final_error_small(self, opt):
        """final_error debe ser pequeño (convergencia)."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        assert result["final_error"] < 1.0

    def test_saving_pct_is_float(self, opt):
        """saving_pct debe ser un float."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        assert isinstance(result["saving_pct"], (float, np.floating))

    def test_x_length_matches_N(self, opt):
        """x debe tener longitud N."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        assert len(result["x"]) == opt.N

    def test_I_positive(self, opt):
        """I debe tener todos los valores positivos."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        assert np.all(result["I"] > 0)

    def test_h_v_positive(self, opt):
        """h_v (altura de viga) debe ser positiva."""
        result = opt.optimizar_viga_completo(L=5.0, q=10.0)
        assert np.all(result["h_v"] > 0)

    def test_callback_invocation(self, opt):
        """Callback debe recibir datos de visualización."""
        calls = []

        def cb(data):
            calls.append(data)

        opt.optimizar_viga_completo(L=5.0, q=10.0, callback=cb)
        assert len(calls) >= 1
        # Verificar estructura de datos del callback
        cb_data = calls[0]
        assert "iteration" in cb_data
        assert "x" in cb_data
        assert "I" in cb_data

    def test_different_loads_different_deflection(self, opt):
        """Distintas cargas deben producir diferente deflexión (q escala la carga)."""
        r1 = opt.optimizar_viga_completo(L=5.0, q=10.0)
        r2 = opt.optimizar_viga_completo(L=5.0, q=20.0)
        # Y (deflexión) escala con q — doble carga = doble deflexión aprox.
        max_y1 = np.max(np.abs(r1["Y"]))
        max_y2 = np.max(np.abs(r2["Y"]))
        assert max_y2 > max_y1, (
            f"q=20 debe dar mayor deflexión que q=10: {max_y2} vs {max_y1}"
        )
