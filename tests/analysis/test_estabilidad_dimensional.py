"""Tests del protocolo de la Figura 5.2 (estabilidad dimensional).

Cubren la aritmética del protocolo (dimensiones, cota probabilística, lectura
sin barra infinita) y la forma de los tableros que consume la figura. La
corrida completa (1 680 réplicas) se valida aparte contra el CSV del Cap. V.
"""

import numpy as np
import pandas as pd
import pytest

from tda.analysis.calibracion import TAU_CALIBRADO
from tda.analysis.estabilidad_dimensional import (
    DELTA_FIG52, N_FIG52, QS_FIG52, R_FIG52, _replica, cota_probabilistica,
    diagramas_finitos, dimensiones, protocolo_fig52, q_estrella, resumen_razones,
)


class TestDimensiones:
    """Cada nube se evalúa en su dimensión nativa, 10 y 100."""

    def test_nubes_planas(self):
        assert dimensiones("circulo") == [2, 10, 100]
        assert dimensiones("gaussiana") == [2, 10, 100]

    def test_nubes_3d(self):
        assert dimensiones("toro") == [3, 10, 100]
        assert dimensiones("esfera") == [3, 10, 100]


class TestCotaProbabilistica:
    def test_formula_del_cap_v(self):
        sigma, d, n, delta = 0.1, 10, 400, 0.05
        esperado = 2 * sigma * (np.sqrt(d) + np.sqrt(2 * np.log(n / delta)))
        assert cota_probabilistica(sigma, d, n, delta) == pytest.approx(esperado)

    def test_crece_con_la_dimension(self):
        creciente = [cota_probabilistica(0.1, d, 400, DELTA_FIG52)
                     for d in (2, 10, 100)]
        assert creciente[0] < creciente[1] < creciente[2]


class TestDiagramasFinitos:
    def test_descarta_la_barra_infinita(self):
        dgms = [np.array([[0.0, 1.0], [0.0, np.inf]]), np.array([[0.5, 0.9]])]
        h0, h1 = diagramas_finitos(dgms)
        assert len(h0) == 1 and np.isfinite(h0).all()
        assert len(h1) == 1

    def test_h1_vacio(self):
        h0, h1 = diagramas_finitos([np.array([[0.0, 1.0]]), np.empty((0, 2))])
        assert len(h0) == 1 and h1.shape == (0, 2)


class TestReplica:
    def test_es_determinista(self):
        tarea = ("circulo", 2, 0.10, 3, 80, TAU_CALIBRADO)
        a, b = _replica(tarea), _replica(tarea)
        assert a == b

    def test_devuelve_todas_las_columnas(self):
        fila = _replica(("anillo", 10, 0.05, 0, 60, TAU_CALIBRADO))
        for clave in ("nube", "d", "q", "replica", "diam", "sigma", "dB", "dH",
                      "cota_teorema", "cota_prob", "b0", "b1", "b0_teo",
                      "b1_teo", "acierto"):
            assert clave in fila
        assert fila["cota_teorema"] == pytest.approx(2 * fila["dH"])
        assert fila["sigma"] == pytest.approx(0.05 * fila["diam"] / np.sqrt(10))

    def test_no_viola_la_cota_del_teorema(self):
        """dB ≤ 2·dH: sobreestimar el ruido no debe romper el teorema."""
        for d in (2, 10, 100):
            fila = _replica(("toro", d, 0.20, 1, 80, TAU_CALIBRADO))
            assert fila["dB"] <= fila["cota_teorema"] + 1e-9


@pytest.fixture(scope="module")
def corrida():
    """Corrida mínima (7 nubes × 3 dimensiones × 4 niveles, sin réplicas)."""
    return protocolo_fig52(n_points=60, n_rep=1, n_jobs=1)


class TestProtocolo:

    def test_una_fila_por_nube_dimension_q(self, corrida):
        rep = corrida["replicas"]
        esperado = 7 * 3 * len(QS_FIG52)
        assert len(rep) == esperado

    def test_columnas_del_tablero(self, corrida):
        assert {"nube", "d", "q", "acierto", "dB", "dH", "diam"} <= set(
            corrida["replicas"].columns)

    def test_q0_trae_la_lectura_sin_ruido(self, corrida):
        q0 = corrida["q0"]
        assert len(q0) == 7
        assert {"nube", "acierto", "b0", "b1"} <= set(q0.columns)

    def test_tau_es_el_calibrado(self, corrida):
        assert corrida["tau"] == TAU_CALIBRADO


class TestResumenes:
    def _rep(self):
        return pd.DataFrame({
            "nube": ["a", "a", "a", "a"],
            "d": [2, 2, 2, 2],
            "q": [0.05, 0.05, 0.10, 0.10],
            "acierto": [1, 1, 0, 0],
            "dB": [1.0, 1.0, 1.0, 1.0],
            "dH": [1.0, 1.0, 1.0, 1.0],
            "cota_teorema": [2.0, 2.0, 2.0, 2.0],
            "diam": [1.0, 1.0, 1.0, 1.0],
        })

    def test_q_estrella(self):
        q = q_estrella(self._rep())
        assert q.loc[0, "q_estrella"] == 0.05

    def test_q_estrella_sin_nivel_valido_es_cero(self):
        rep = self._rep()
        rep["acierto"] = 0
        assert q_estrella(rep).loc[0, "q_estrella"] == 0.0

    def test_resumen_razones(self):
        r = resumen_razones(self._rep())
        assert r.loc[0, "medio"] == pytest.approx(0.5)
        assert r.loc[0, "maximo"] == pytest.approx(0.5)
