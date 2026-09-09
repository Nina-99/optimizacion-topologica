"""Tests del optimizador de vigas — BeamOptimizer.

Verifica funciones para Viga en Voladizo de Acero (Cantilever).
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

    def test_I0_calculation(self):
        """I0 debe ser b * h0³ / 12."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        expected_I0 = 0.3 * 0.5**3 / 12.0
        assert_allclose(opt.I0, expected_I0, atol=1e-12)

    def test_I_min_is_fraction_of_I0(self):
        """I_min debe ser 0.05 * I0 en acero."""
        opt = BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)
        assert_allclose(opt.I_min, 0.05 * opt.I0, atol=1e-12)


class TestOptimizarVigaCompleto:
    """Optimización completa de la viga en voladizo."""

    @pytest.fixture
    def opt(self):
        """BeamOptimizer con N pequeño."""
        return BeamOptimizer(b=0.3, h0=0.5, p=3, N=10)

    def test_returns_dict(self, opt):
        """optimizar_viga_completo debe retornar un dict."""
        result = opt.optimizar_viga_completo(L=5.0, F=1000.0)
        assert isinstance(result, dict)

    def test_expected_keys(self, opt):
        """El dict debe contener las claves esperadas."""
        result = opt.optimizar_viga_completo(L=5.0, F=1000.0)
        required = {"x", "I", "Y", "Y_original", "M", "h_v",
                    "iterations", "final_error", "saving_pct",
                    "weight_saved", "sigma_MPa", "L", "y_adm", "F"}
        missing = required - set(result.keys())
        assert not missing, f"Faltan claves: {missing}"

    def test_momento_cantilever(self, opt):
        """Momento flector en el empotramiento debe ser -F*L."""
        L = 5.0
        F = 1000.0
        result = opt.optimizar_viga_completo(L=L, F=F)
        M_fixed = result["M"][0]  # x=0
        M_free = result["M"][-1]  # x=L
        assert_allclose(M_fixed, -F * L, atol=1e-10)
        assert_allclose(M_free, 0.0, atol=1e-10)

    def test_callback_invocation(self, opt):
        """Callback debe recibir datos de visualización."""
        calls = []

        def cb(data):
            calls.append(data)

        opt.optimizar_viga_completo(L=5.0, F=1000.0, callback=cb)
        assert len(calls) >= 1
        # Verificar estructura de datos del callback
        cb_data = calls[0]
        assert "iteration" in cb_data
        assert "x" in cb_data
        assert "I" in cb_data
