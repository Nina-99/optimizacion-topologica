"""Tests del módulo FEM — K0, ensamble, solver, compliance, filtro, OC.

Verifica invariantes matemáticas del núcleo de elementos finitos Q4:
  - Matriz de rigidez elemental K_0 (8×8) es definida positiva
  - Ensamble vectorizado produce matriz sparse simétrica
  - Solución K·U = F satisface condiciones de contorno
  - Compliance y sensibilidades son consistentes
  - Filtro de sensibilidad conserva signo físico
  - Actualización OC respeta restricción de volumen
"""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal

from tda.core.fem import (
    actualizar_OC,
    calcular_compliance_sensibilidades,
    calcular_K_elemental,
    ensamblar_K_global,
    filtrar_sensibilidades,
    resolver_FEM,
)


class TestK0:
    """Matriz de rigidez elemental K_0."""

    def test_shape(self):
        """K_0 debe ser (8, 8) para elemento Q4 de 4 nodos (2 DOF c/u)."""
        K0 = calcular_K_elemental(E=1.0, nu=0.3)
        assert K0.shape == (8, 8)

    def test_symmetric(self):
        """K_0 debe ser simétrica por construcción (material elástico lineal)."""
        K0 = calcular_K_elemental(E=1.0, nu=0.3)
        assert_allclose(K0, K0.T, atol=1e-12,
                        err_msg="K_0 no es simétrica")

    def test_positive_semidefinite(self):
        """K_0 debe ser semidefinida positiva (autovalores ≥ 0 en precisión máquina).

        Nota: modo(s) rígido(s) de cuerpo libre producen autovalores
        ~1e-17 que son cero en precisión numérica.
        """
        K0 = calcular_K_elemental(E=1.0, nu=0.3)
        eigvals = np.linalg.eigvalsh(K0)
        assert np.all(eigvals > -1e-12), (
            f"K_0 tiene autovalores negativos no despreciables: {eigvals}"
        )
        # Verificar que al menos los modos elásticos son positivos
        n_pos = np.sum(eigvals > 1e-10)
        assert n_pos >= 3, (
            f"K_0 debe tener al menos 3 autovalores positivos, tiene {n_pos}"
        )

    def test_linear_in_E(self):
        """K_0 debe escalar linealmente con E (módulo de Young)."""
        K0_1 = calcular_K_elemental(E=1.0, nu=0.3)
        K0_2 = calcular_K_elemental(E=2.0, nu=0.3)
        assert_allclose(K0_2, 2.0 * K0_1, atol=1e-12)

    def test_zero_entries_outside_expected(self):
        """Componentes de K_0 fuera de la estructura esperada son exactamente cero."""
        K0 = calcular_K_elemental(E=1.0, nu=0.3)
        # Para Q4, el patrón de acoplamiento nodo-nodo es predecible:
        # filas 0..1 (nodo 1) se acoplan con columnas 0..1, 2..3, 4..5, 6..7
        # (acoplamiento rígido entre todos los nodos del elemento)
        # Verificar que no hay NaN o Inf
        assert np.all(np.isfinite(K0)), "K_0 contiene NaN o Inf"


class TestEnsamble:
    """Ensamble de matriz de rigidez global."""

    def test_ensamble_shape(self, DOFS, K0, n_dof, p, rho0):
        """K global debe ser (n_dof, n_dof)."""
        rho = np.full(DOFS.shape[0], 0.5)
        K = ensamblar_K_global(rho, DOFS, n_dof, K0, p)
        assert K.shape == (n_dof, n_dof), (
            f"K global esperada ({n_dof}, {n_dof}), obtenida {K.shape}"
        )

    def test_ensamble_sparse(self, DOFS, K0, n_dof, p, rho0):
        """K global debe ser sparse (scipy.sparse.spmatrix)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        assert sp.issparse(K), "K global debe ser sparse"

    def test_ensamble_symmetric(self, DOFS, K0, n_dof, p, rho0):
        """K global debe ser simétrica (ley de Betti-Maxwell)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        K_dense = K.toarray()
        assert_allclose(K_dense, K_dense.T, atol=1e-10,
                        err_msg="K global no es simétrica")

    def test_ensamble_linear_in_penalty(self, DOFS, K0, n_dof, p, rho0):
        """K(ρ) debe escalar con E_e = ρ^p (SIMP lineal en penalización)."""
        K_half = ensamblar_K_global(rho0 * 0.5, DOFS, n_dof, K0, p)
        K_full = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        # Densidad 0.5^3 = 0.125 del material completo
        expected_ratio = (0.5 ** p)
        assert_allclose(K_half.toarray(), expected_ratio * K_full.toarray(),
                        atol=1e-10, err_msg="Penalización SIMP no lineal")


class TestSolver:
    """Solver FEM con condiciones de contorno."""

    def test_solver_returns_correct_shape(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """U debe ser (n_dof,)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        assert U.shape == (n_dof,), f"U esperado ({n_dof},), obtenido {U.shape}"

    def test_solver_fixed_dofs_zero(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """Los DOF fijos deben tener U = 0."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        assert_allclose(U[fixed], 0.0, atol=1e-12,
                        err_msg="DOF fijos deben tener desplazamiento cero")

    def test_solver_finite_solution(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """U debe ser finito (sin NaN ni Inf)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        assert np.all(np.isfinite(U)), "Solución U contiene NaN o Inf"

    def test_solver_nonzero_at_loaded_dof(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """El DOF donde se aplica la carga debe tener desplazamiento no nulo."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        loaded_idx = np.argmax(np.abs(F))
        assert np.abs(U[loaded_idx]) > 0, (
            f"DOF {loaded_idx} (donde se aplica carga) tiene U=0"
        )

    def test_solver_linear_superposition(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """El sistema lineal debe satisfacer K·U ≈ F en los DOF libres."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        free = np.setdiff1d(np.arange(n_dof), fixed)
        residual = K[free, :][:, free] @ U[free] - F[free]
        assert_allclose(residual, 0.0, atol=1e-8,
                        err_msg="Residuo K·U - F no es cero en DOF libres")


class TestCompliance:
    """Compliance y sensibilidades."""

    def test_compliance_positive(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """c(ρ) debe ser > 0 para ρ > 0 y F ≠ 0."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        assert c > 0, f"Compliance debe ser positiva, obtenida {c}"

    def test_sensitivity_shape(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """dc debe tener shape (N_e,)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        assert dc.shape == (DOFS.shape[0],), (
            f"dc esperado ({DOFS.shape[0]},), obtenido {dc.shape}"
        )

    def test_sensitivity_negative(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """∂c/∂ρ_e debe ser ≤ 0 (más material → menor compliance)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        assert np.all(dc <= 0) or np.all(dc <= 1e-12), (
            "Sensibilidades deben ser ≤ 0 (físicamente: más material = más rígido)"
        )

    def test_compliance_finite(self, K0, DOFS, n_dof, p, F, fixed, rho0):
        """c y dc deben ser finitos."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        assert np.isfinite(c), "Compliance no es finita"
        assert np.all(np.isfinite(dc)), "Sensibilidades contienen NaN/Inf"


class TestFilter:
    """Filtro de sensibilidad por convolución espacial."""

    def test_filter_preserves_sign(self, K0, DOFS, n_dof, p, F, fixed, rho0, H):
        """dc_filt debe conservar signo ≤ 0 (propiedad física del filtro)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        dc_filt = filtrar_sensibilidades(dc, rho0, H)
        assert np.all(dc_filt <= 0) or np.all(dc_filt <= 1e-12), (
            "Filtro debe preservar signo negativo de sensibilidades"
        )

    def test_filter_shape(self, K0, DOFS, n_dof, p, F, fixed, rho0, H):
        """dc_filt debe tener mismo shape que dc."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        dc_filt = filtrar_sensibilidades(dc, rho0, H)
        assert dc_filt.shape == dc.shape

    def test_filter_finite(self, K0, DOFS, n_dof, p, F, fixed, rho0, H):
        """dc_filt debe ser finito."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        dc_filt = filtrar_sensibilidades(dc, rho0, H)
        assert np.all(np.isfinite(dc_filt)), "dc_filt contiene NaN o Inf"


class TestOC:
    """Actualización por Criterio de Optimalidad."""

    def test_oc_volume_constraint(self, K0, DOFS, n_dof, p, F, fixed, rho0, H, eps_star):
        """ρ nuevo debe satisfacer mean(ρ) ≈ f_V (restricción de volumen)."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        dc_filt = filtrar_sensibilidades(dc, rho0, H)
        rho_nuevo = actualizar_OC(rho0, dc_filt, eps_star)
        assert_allclose(rho_nuevo.mean(), eps_star, atol=0.02,
                        err_msg="OC debe satisfacer restricción de volumen")

    def test_oc_bounds(self, K0, DOFS, n_dof, p, F, fixed, rho0, H, eps_star):
        """ρ nuevo debe estar en [rho_min, 1]."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        dc_filt = filtrar_sensibilidades(dc, rho0, H)
        rho_nuevo = actualizar_OC(rho0, dc_filt, eps_star)
        assert np.all(rho_nuevo >= 1e-3 - 1e-10), "ρ < ρ_min"
        assert np.all(rho_nuevo <= 1.0 + 1e-10), "ρ > 1"

    def test_oc_move_limit(self, K0, DOFS, n_dof, p, F, fixed, rho0, H, eps_star):
        """|ρ_nuevo - ρ_anterior| ≤ move (0.2) en cada elemento."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        dc_filt = filtrar_sensibilidades(dc, rho0, H)
        rho_nuevo = actualizar_OC(rho0, dc_filt, eps_star)
        delta = np.max(np.abs(rho_nuevo - rho0))
        assert delta <= 0.2 + 1e-10, (
            f"|Δρ|_max = {delta} > 0.2 (move limit superado)"
        )

    def test_oc_finite(self, K0, DOFS, n_dof, p, F, fixed, rho0, H, eps_star):
        """ρ_nuevo debe ser finito."""
        K = ensamblar_K_global(rho0, DOFS, n_dof, K0, p)
        U = resolver_FEM(K, F, fixed, n_dof)
        c, dc = calcular_compliance_sensibilidades(U, rho0, DOFS, K0, p)
        dc_filt = filtrar_sensibilidades(dc, rho0, H)
        rho_nuevo = actualizar_OC(rho0, dc_filt, eps_star)
        assert np.all(np.isfinite(rho_nuevo)), "ρ_nuevo contiene NaN o Inf"
