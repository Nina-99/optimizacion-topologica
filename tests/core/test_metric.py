"""Tests de la métrica compuesta TDA-SIMP y calibración de α*.

Verifica la Definición 1.9 y Proposición 1.1:
  - μ_α(ρ*) = c(ρ*) + α · β_1(Ω_sólido)
  - α* = (c̄ − c_min) / (β_{1,max} − β_{1,min})
  - Propiedades: positividad, monotonía topológica
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.core.metric import calibrar_alpha_optimo, metrica_compuesta


class TestMetricaCompuesta:
    """Métrica compuesta μ_α."""

    def test_basic_calculation(self):
        """μ_α = c + α*β₁ con valores enteros conocidos."""
        mu = metrica_compuesta(c=10.0, beta1=2, alpha=0.5)
        assert mu == 11.0, f"Esperado 11.0, obtenido {mu}"

    def test_positive(self):
        """μ_α > 0 para cualquier c>0, β₁≥0, α>0."""
        mu = metrica_compuesta(c=1e-6, beta1=0, alpha=0.1)
        assert mu > 0, f"μ_α debe ser positivo, obtenido {mu}"

    def test_monotonic_topology(self):
        """A menor β₁ (con c fija) → menor μ_α."""
        c_val = 50.0
        alpha_val = 1.0
        mu_0 = metrica_compuesta(c_val, 0, alpha_val)
        mu_1 = metrica_compuesta(c_val, 1, alpha_val)
        mu_2 = metrica_compuesta(c_val, 2, alpha_val)
        assert mu_0 < mu_1 < mu_2, (
            "μ_α debe ser monótona creciente con β₁"
        )

    def test_alpha_scales_linearly(self):
        """μ_α escala linealmente con α (derivada parcial ∂μ/∂α = β₁)."""
        c_val = 30.0
        beta1_val = 3
        mu_a1 = metrica_compuesta(c_val, beta1_val, 0.5)
        mu_a2 = metrica_compuesta(c_val, beta1_val, 1.0)
        assert_allclose(mu_a2 - mu_a1, beta1_val * 0.5, atol=1e-12)

    def test_beta1_as_integer(self):
        """β₁ debe aceptarse como int y como float."""
        mu_int = metrica_compuesta(10.0, 2, 0.5)
        mu_float = metrica_compuesta(10.0, 2.0, 0.5)
        assert_allclose(mu_int, mu_float, atol=1e-12)

    def test_zero_alpha(self):
        """Con α=0, μ_α = c (métrica puramente mecánica)."""
        mu = metrica_compuesta(c=25.0, beta1=5, alpha=0.0)
        assert mu == 25.0


class TestCalibrarAlpha:
    """Calibración de α* óptimo."""

    def test_basic_calibration(self):
        """α* con valores de referencia conocidos."""
        # c = [10, 20] → c̄=15, c_min=10
        # β₁ = [0, 2] → rango=2
        # α* = (15-10)/2 = 2.5
        alpha_star = calibrar_alpha_optimo([10.0, 20.0], [0, 2])
        assert_allclose(alpha_star, 2.5, atol=1e-10)

    def test_single_design(self):
        """Un solo diseño debe dar α* = 0.01 (default por rango cero)."""
        alpha_star = calibrar_alpha_optimo([15.0], [1])
        assert_allclose(alpha_star, 0.01, atol=1e-10)

    def test_zero_beta1_range(self):
        """β₁ igual para todos los diseños debe dar α* = 0.01."""
        alpha_star = calibrar_alpha_optimo([10.0, 20.0, 30.0], [1, 1, 1])
        assert_allclose(alpha_star, 0.01, atol=1e-10)

    def test_alpha_positive(self):
        """α* debe ser siempre > 0."""
        alpha_star = calibrar_alpha_optimo([10.0, 20.0], [5, 10])
        assert alpha_star > 0, f"α* debe ser positivo, obtenido {alpha_star}"

    def test_no_variation_in_compliance(self):
        """c igual para todos debe dar α* = 1e-4 (default mínimo)."""
        alpha_star = calibrar_alpha_optimo([10.0, 10.0, 10.0], [0, 1, 2])
        assert_allclose(alpha_star, 1e-4, atol=1e-10)

    def test_larger_c_range_gives_larger_alpha(self):
        """Mayor rango de compliance debe dar mayor α* (más peso topológico)."""
        a1 = calibrar_alpha_optimo([10.0, 20.0], [0, 2])   # c_range=10
        a2 = calibrar_alpha_optimo([10.0, 30.0], [0, 2])   # c_range=20
        # (15-10)/2 = 2.5  vs  (20-10)/2 = 5.0
        assert a2 > a1, "Mayor rango de c debe dar mayor α*"
