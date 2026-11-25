"""Tests del instrumento topológico del Caso 2 (Algoritmo 3, Cap. IV).

Cubren la convención de 8-conectividad (la que induce la unión de cuadrados
cerrados) y la persistencia del campo de densidades (Fig. 5.7c).
"""

import numpy as np

from tda.core.betti2d import (
    betti_doble_computo,
    diagramas_campo,
    persistencia_minima,
)


class TestConectividad:
    def test_anillo_3x3(self):
        """El anillo 3×3 (8 píxeles) tiene β₀=1 y β₁=1 por ambos métodos."""
        S = np.ones((3, 3), dtype=bool)
        S[1, 1] = False
        res = betti_doble_computo(S)
        assert (res["beta0_euler"], res["beta1_euler"]) == (1, 1)
        assert res["concordancia"] is True

    def test_diagonal_2x2_es_una_componente(self):
        """Dos celdas que comparten un vértice son una sola componente (8-conexa)."""
        S = np.array([[True, False], [False, True]])
        res = betti_doble_computo(S)
        assert (res["beta0_euler"], res["beta1_euler"]) == (1, 0)
        assert res["concordancia"] is True

    def test_vacio(self):
        S = np.zeros((4, 4), dtype=bool)
        assert betti_doble_computo(S)["concordancia"] is True


class TestPersistenciaCampo:
    def test_diagrama_campo_tiene_h1(self):
        """Un anillo de densidad 1 con un hueco de densidad 0 produce una barra H₁."""
        rho = np.ones((9, 9))
        rho[3:6, 3:6] = 0.0
        dgm0, dgm1 = diagramas_campo(rho)
        assert dgm1.shape[1] == 2 and len(dgm1) >= 1
        assert persistencia_minima(dgm1) > 0.5

    def test_persistencia_estructural_ignora_ruido(self):
        """Con k=1 se lee el agujero dominante, no la barra de ruido más corta."""
        dgm = np.array([[0.0, 0.9], [0.0, 0.1], [0.0, 0.05]])
        assert persistencia_minima(dgm) == 0.05
        assert persistencia_minima(dgm, k=1) == 0.9
