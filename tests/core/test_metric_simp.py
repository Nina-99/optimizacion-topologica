"""Tests de la clase MetricaTDA_SIMP — inicialización, problema, optimización, TDA.

Verifica el Algoritmo 1 completo:
  - FASE 1: init con geometría y parámetros correctos
  - FASE 2: definir_problema asigna cargas y condiciones de contorno
  - FASE 3: optimizar con malla 2×2 y max_iter=2 (iteración corta)
  - FASE 4: fase_tda con doble cómputo (Euler + GUDHI)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.optimization.metric_simp import MetricaTDA_SIMP


class TestInit:
    """Inicialización de MetricaTDA_SIMP."""

    def test_basic_params(self):
        """Los parámetros básicos deben almacenarse correctamente."""
        m = MetricaTDA_SIMP(nex=60, ney=30, f_V=0.5, p=3, alpha=0.012)
        assert m.nex == 60
        assert m.ney == 30
        assert m.N_e == 1800
        assert m.f_V == 0.5
        assert m.p == 3
        assert m.alpha == 0.012

    def test_dof_count(self):
        """n_dof debe ser 2 × (nex+1) × (ney+1)."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        assert m.n_dof == 2 * 3 * 3 == 18

    def test_k0_shape(self):
        """K0 debe ser (8, 8)."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        assert m.K0.shape == (8, 8)

    def test_initial_rho(self):
        """ρ inicial debe ser uniforme = f_V."""
        m = MetricaTDA_SIMP(nex=2, ney=2, f_V=0.5)
        assert_allclose(m.rho, 0.5)
        assert m.rho.shape == (4,)

    def test_dofs_shape(self):
        """DOFS debe tener shape (N_e, 8)."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        assert m.DOFS.shape == (4, 8)

    def test_filter_matrix(self):
        """H debe ser (N_e, N_e) con entradas ≥ 0."""
        m = MetricaTDA_SIMP(nex=2, ney=2, r_min=2.4)
        assert m.H.shape == (4, 4)
        assert np.all(m.H >= 0)
        assert np.all(m.H <= 2.4)

    def test_default_tolerance(self):
        """tol_c y tol_rho por defecto deben ser 1e-4 y 1e-2 (Perfil §9.4.4)."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        assert m.tol_c == 1e-4
        assert m.tol_rho == 1e-2

    def test_default_max_iter(self):
        """max_iter por defecto debe ser 200."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        assert m.max_iter == 200

    def test_cx_cy_shape(self):
        """cx y cy deben tener (N_e,)."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        assert m.cx.shape == (4,)
        assert m.cy.shape == (4,)


class TestDefinirProblema:
    """Asignación de cargas y condiciones de contorno."""

    def test_definir_problema_sets_F(self):
        """F debe asignarse como array de float con shape (n_dof,)."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        F = np.zeros(m.n_dof)
        fixed = np.array([0, 1, 6, 7], dtype=int)
        m.definir_problema(F, fixed)
        assert m.F.shape == (m.n_dof,)
        assert m.F.dtype == float

    def test_definir_problema_sets_fixed(self):
        """fixed debe almacenar los índices de DOF fijos."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        F = np.zeros(m.n_dof)
        fixed = np.array([0, 1, 6, 7], dtype=int)
        m.definir_problema(F, fixed)
        assert_allclose(m.fixed, fixed)

    def test_definir_problema_converts_to_array(self):
        """Listas como input deben convertirse a ndarray."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        m.definir_problema([0.0] * m.n_dof, [0, 1])
        assert isinstance(m.F, np.ndarray)
        assert isinstance(m.fixed, np.ndarray)

    def test_optimizar_asserts_without_problem(self):
        """optimizar() sin definir_problema() debe lanzar AssertionError."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        with pytest.raises(AssertionError, match="definir_problema"):
            m.optimizar()


class TestOptimizar:
    """Bucle SIMP con malla 2×2 y max_iter=2."""

    @pytest.fixture
    def m_2x2(self):
        """Instancia de MetricaTDA_SIMP con malla 2×2 y max_iter=2."""
        m = MetricaTDA_SIMP(nex=2, ney=2, f_V=0.5, p=3,
                            r_min=2.4, max_iter=2)
        F = np.zeros(m.n_dof)
        # Nodo 8 (top-right): DOF 17 = uy, carga vertical
        F[17] = -1.0
        # Borde izquierdo fijo: nodos 0, 3, 6
        fixed = np.array([0, 1, 6, 7, 12, 13], dtype=int)
        m.definir_problema(F, fixed)
        return m

    def test_optimizar_returns_self(self, m_2x2):
        """optimizar() debe retornar self (para encadenamiento)."""
        result = m_2x2.optimizar()
        assert result is m_2x2

    def test_optimizar_sets_c_hist(self, m_2x2):
        """c_hist debe tener al menos 1 entrada después de optimizar."""
        m_2x2.optimizar()
        assert len(m_2x2.c_hist) >= 1
        assert all(np.isfinite(c) for c in m_2x2.c_hist)

    def test_optimizar_sets_rho_final(self, m_2x2):
        """rho_final debe tener shape (N_e,) y valores en [0, 1]."""
        m_2x2.optimizar()
        assert m_2x2.rho_final is not None
        assert m_2x2.rho_final.shape == (4,)
        assert np.all(m_2x2.rho_final >= 0)
        assert np.all(m_2x2.rho_final <= 1)

    def test_optimizar_sets_c_final(self, m_2x2):
        """c_final debe ser un float > 0."""
        m_2x2.optimizar()
        assert m_2x2.c_final is not None
        assert m_2x2.c_final > 0
        assert np.isfinite(m_2x2.c_final)

    def test_optimizar_n_iter_positive(self, m_2x2):
        """n_iter debe ser ≥ 1 después de optimizar."""
        m_2x2.optimizar()
        assert m_2x2.n_iter >= 1

    def test_optimizar_converged_flag(self, m_2x2):
        """Con max_iter=2 puede no converger, pero no debe fallar."""
        m_2x2.optimizar()
        # Con max_iter=2 es probable que no haya convergencia aún
        # Solo verificar que el flag existe
        assert hasattr(m_2x2, "converged")

    def test_optimizar_callback(self, m_2x2):
        """Callback debe recibir (k, c, delta_c, delta_rho, rho)."""
        calls = []

        def cb(k, c, dc, dr, rho):
            calls.append((k, c, dc, dr, rho.shape))

        m_2x2.optimizar(callback=cb)
        assert len(calls) >= 1
        k, c, dc, dr, rshape = calls[0]
        assert k >= 1
        assert isinstance(c, float)
        assert rshape == (4,)


class TestFaseTDA:
    """Fase de análisis topológico (doble cómputo Euler + GUDHI)."""

    @pytest.fixture
    def m_2x2(self):
        """Instancia con optimización completada."""
        m = MetricaTDA_SIMP(nex=2, ney=2, f_V=0.5, p=3,
                            r_min=2.4, max_iter=2)
        F = np.zeros(m.n_dof)
        F[17] = -1.0
        fixed = np.array([0, 1, 6, 7, 12, 13], dtype=int)
        m.definir_problema(F, fixed)
        m.optimizar(verbose=False)
        return m

    def test_fase_tda_asserts_without_optimize(self):
        """fase_tda() sin optimizar() debe lanzar AssertionError."""
        m = MetricaTDA_SIMP(nex=2, ney=2)
        with pytest.raises(AssertionError, match="optimizar"):
            m.fase_tda()

    def test_fase_tda_sets_beta1(self, m_2x2):
        """fase_tda debe setear beta1 (0 si no hay puntos)."""
        mu = m_2x2.fase_tda()
        assert m_2x2.beta1 is not None
        assert m_2x2.beta1 >= 0
        assert isinstance(m_2x2.beta1, int)
        assert isinstance(mu, float)

    def test_fase_tda_sets_concordancia(self, m_2x2):
        """fase_tda debe setear betti_concordancia (bool)."""
        m_2x2.fase_tda()
        assert m_2x2._betti_concordancia is not None
        assert isinstance(m_2x2._betti_concordancia, bool)

    def test_fase_tda_sets_mu(self, m_2x2):
        """fase_tda debe setear mu (métrica compuesta)."""
        mu = m_2x2.fase_tda()
        assert m_2x2.mu is not None
        assert_allclose(m_2x2.mu, mu, atol=1e-12)
        assert m_2x2.mu > 0

    def test_fase_tda_sets_t_tda(self, m_2x2):
        """fase_tda debe setear t_tda > 0."""
        m_2x2.fase_tda()
        assert m_2x2.t_tda > 0

    def test_obtener_resultados(self, m_2x2):
        """obtener_resultados debe incluir campos clave."""
        m_2x2.fase_tda()
        res = m_2x2.obtener_resultados()
        for key in ["rho_final", "rho_tilde_final", "c_final", "c_hist", "beta1",
                     "betti_concordancia", "mu", "nex", "ney",
                     "tol_c", "tol_rho"]:
            assert key in res, f"Falta clave '{key}' en resultados"


class TestFiltrarDensidad:
    """Tests del filtro de densidad ρ̃ (§8.4 del Perfil)."""

    def test_filtrar_densidad_uniforme(self):
        """Densidad uniforme ρ ≡ fV debe dar ρ̃ ≈ fV."""
        m = MetricaTDA_SIMP(nex=4, ney=4, f_V=0.5, r_min=2.4)
        rho = np.full(16, 0.5)
        rho_tilde = m.filtrar_densidad(rho)
        # Para densidad uniforme, ρ̃ = ρ (el filtro preserva constantes)
        assert_allclose(rho_tilde, 0.5, atol=1e-10)

    def test_filtrar_densidad_shape(self):
        """ρ̃ debe tener mismo shape que ρ."""
        m = MetricaTDA_SIMP(nex=3, ney=3)
        rho = np.random.rand(9)
        rho_tilde = m.filtrar_densidad(rho)
        assert rho_tilde.shape == rho.shape

    def test_rho_tilde_hist_se_guarda(self):
        """optimizar() debe guardar rho_tilde_hist."""
        m = MetricaTDA_SIMP(nex=2, ney=2, max_iter=2)
        F = np.zeros(m.n_dof)
        F[17] = -1.0
        fixed = np.array([0, 1, 6, 7, 12, 13], dtype=int)
        m.definir_problema(F, fixed)
        m.optimizar(verbose=False)
        assert len(m.rho_tilde_hist) == m.n_iter

    def test_fase_tda_usa_rho_tilde(self):
        """fase_tda debe usar rho_tilde_final para binarización (§8.4)."""
        m = MetricaTDA_SIMP(nex=2, ney=2, max_iter=2)
        F = np.zeros(m.n_dof)
        F[17] = -1.0
        fixed = np.array([0, 1, 6, 7, 12, 13], dtype=int)
        m.definir_problema(F, fixed)
        m.optimizar(verbose=False)
        m.fase_tda(verbose=False)
        # rho_tilde_final debe existir y tener shape correcto
        assert m.rho_tilde_final is not None
        assert m.rho_tilde_final.shape == (4,)
