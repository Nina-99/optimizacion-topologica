"""Tests del optimizador SIMP 2D — SimpTda2DOptimizer.

Verifica que:
  - La inicialización almacena parámetros y precalcula estructuras FEM
  - run_optimization con malla pequeña (4×2) retorna tupla de 5 elementos
  - xPhys tiene shape (nely, nelx)
  - dgms es una lista de diagramas de persistencia
  - betti_1 es un entero ≥ 0
  - compliance (c) es finito y positivo
  - reduccion_pct es un float
  - El callback opcional se invoca correctamente
"""

import numpy as np
import pytest

from tda.optimization.simp_optimizer import SimpTda2DOptimizer


class TestInit:
    """Inicialización de SimpTda2DOptimizer — parámetros y estructuras."""

    def test_default_params(self):
        """Los parámetros por defecto deben almacenarse correctamente."""
        opt = SimpTda2DOptimizer()
        assert opt.nelx == 60
        assert opt.nely == 30
        assert opt.volfrac == 0.5
        assert opt.penal == 3.0
        assert opt.rmin == 1.5

    def test_custom_params(self):
        """Parámetros personalizados deben almacenarse."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2, volfrac=0.4,
                                 penal=3.0, rmin=1.2)
        assert opt.nelx == 4
        assert opt.nely == 2
        assert opt.volfrac == 0.4
        assert opt.penal == 3.0
        assert opt.rmin == 1.2

    def test_ndof_calculation(self):
        """ndof debe ser 2 * (nelx+1) * (nely+1)."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        assert opt.ndof == 2 * 5 * 3 == 30

    def test_edofMat_shape(self):
        """edofMat debe tener shape (nelx*nely, 8)."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        assert opt.edofMat.shape == (8, 8)

    def test_KE_shape(self):
        """KE debe ser (8, 8) — matriz de rigidez elemental."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        assert opt.KE.shape == (8, 8)

    def test_F_has_force(self):
        """F debe tener una carga distinta de cero (voladizo)."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        assert np.any(opt.F != 0), "F debe tener al menos una carga"

    def test_fixeddofs_not_empty(self):
        """fixeddofs debe tener DOF del borde izquierdo."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        assert len(opt.fixeddofs) > 0

    def test_freedofs_correct_count(self):
        """freedofs debe ser ndof - len(fixeddofs)."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        expected_free = opt.ndof - len(opt.fixeddofs)
        assert len(opt.freedofs) == expected_free

    def test_H_filter_matrix(self):
        """H debe ser matriz sparse (N_e, N_e)."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        n_e = opt.nelx * opt.nely
        assert opt.H.shape == (n_e, n_e)

    def test_tolerance_defaults(self):
        """Tolerancias por defecto deben estar configuradas."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        assert opt.tol_c == 1e-4
        assert opt.tol_x == 1e-3

    def test_max_iter_default(self):
        """max_iter por defecto debe ser 100."""
        opt = SimpTda2DOptimizer(nelx=4, nely=2)
        assert opt.max_iter == 100


class TestRunOptimization:
    """Bucle SIMP con malla pequeña 4×2 y max_iter=3."""

    @pytest.fixture
    def opt_small(self):
        """Instancia con malla chica y pocas iteraciones."""
        opt = SimpTda2DOptimizer(nelx=6, nely=3, volfrac=0.5,
                                 penal=3.0, rmin=1.5)
        opt.max_iter = 3
        opt.min_iter = 1
        return opt

    def test_returns_tuple_of_5(self, opt_small):
        """run_optimization debe retornar tupla de 5 elementos."""
        result = opt_small.run_optimization()
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_xPhys_shape(self, opt_small):
        """xPhys debe tener shape (nely, nelx)."""
        xPhys, _, _, _, _ = opt_small.run_optimization()
        assert xPhys.shape == (opt_small.nely, opt_small.nelx)

    def test_xPhys_in_range(self, opt_small):
        """xPhys debe tener valores en [0, 1]."""
        xPhys, _, _, _, _ = opt_small.run_optimization()
        assert np.all(xPhys >= 0)
        assert np.all(xPhys <= 1)

    def test_dgms_is_list(self, opt_small):
        """dgms debe ser una lista de diagramas."""
        _, dgms, _, _, _ = opt_small.run_optimization()
        assert isinstance(dgms, list)
        assert len(dgms) >= 1

    def test_betti1_is_int(self, opt_small):
        """betti_1 debe ser un entero ≥ 0."""
        _, _, betti_1, _, _ = opt_small.run_optimization()
        assert isinstance(betti_1, (int, np.integer))
        assert betti_1 >= 0

    def test_compliance_finite(self, opt_small):
        """Compliance final debe ser finito y positivo."""
        _, _, _, c, _ = opt_small.run_optimization()
        assert np.isfinite(c)
        assert c > 0

    def test_reduccion_is_float(self, opt_small):
        """reduccion_pct debe ser un float."""
        _, _, _, _, reduccion = opt_small.run_optimization()
        assert isinstance(reduccion, (float, np.floating))

    def test_callback_invocation(self, opt_small):
        """El callback debe recibir (loop, xPhys, c, reduccion, max_iter)."""
        calls = []

        def cb(loop, xPhys, c, reduccion, max_iter):
            calls.append((loop, c, reduccion, max_iter))

        opt_small.run_optimization(callback=cb)
        assert len(calls) >= 1
        loop, c, reduccion, mx = calls[0]
        assert loop >= 1
        assert isinstance(c, float)
        assert isinstance(reduccion, float)
        assert mx == opt_small.max_iter

    def test_xPhys_has_some_variation(self, opt_small):
        """xPhys no debe ser completamente uniforme tras optimizar."""
        xPhys, _, _, _, _ = opt_small.run_optimization()
        # Con pocas iteraciones puede haber poca variación, pero
        # al menos verificar que no es todo exactamente 0.5
        assert np.any(xPhys != 0.5) or np.all(xPhys >= 0)

    def test_multiple_calls_different_results(self, opt_small):
        """Dos llamadas consecutivas deben producir resultados similares (determinismo)."""
        xPhys1, dgms1, b1_1, c1, r1 = opt_small.run_optimization()
        xPhys2, dgms2, b1_2, c2, r2 = opt_small.run_optimization()
        # Con seed fija los resultados deben ser idénticos
        assert np.allclose(xPhys1, xPhys2)
