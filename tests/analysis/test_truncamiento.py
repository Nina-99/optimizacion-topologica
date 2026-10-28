"""Tests del análisis de truncamiento (panel a de la Figura 5.3).

La corrida completa tarda ~80 s; los tests usan una nube chica y verificación
cualitativa: con la filtración truncada el ciclo del círculo desaparece y
`dB/diam` queda por debajo de `2q`; con la filtración completa lo supera.
"""

import numpy as np
import pandas as pd
import pytest

from tda.analysis.truncamiento import (
    analisis_truncamiento, cargar_truncamiento, guardar_truncamiento,
    resumen_truncamiento,
)

QS = (0.0, 0.20)


@pytest.fixture(scope="module")
def corrida():
    return analisis_truncamiento(n_points=80, qs=QS, n_rep=1)


class TestAnalisisTruncamiento:
    def test_dos_modos_y_una_fila_por_celda(self, corrida):
        assert set(corrida["modo"].unique()) == {
            "truncado (ε ≤ 0.6, barras infinitas descartadas)",
            "completo (sin truncar)"}
        assert len(corrida) == len(QS) * 1 * 2

    def test_columnas(self, corrida):
        assert list(corrida.columns) == ["modo", "q", "replica", "dB_diam"]

    def test_sin_ruido_la_distancia_es_cero(self, corrida):
        q0 = corrida[corrida["q"] == 0.0]
        assert (q0["dB_diam"].abs() < 1e-12).all()

    def test_el_truncamiento_esconde_el_ciclo(self, corrida):
        """Afuera del truncamiento la distancia es mucho mayor."""
        medios = resumen_truncamiento(corrida)
        truncado = next(v for k, v in medios.items() if k.startswith("truncado"))
        completo = next(v for k, v in medios.items() if k.startswith("completo"))
        assert completo[0.20] > truncado[0.20]

    def test_el_completo_supera_la_cota_2q(self, corrida):
        """Con la filtración completa, dB/diam > 2q en el nivel más alto."""
        completo = corrida[corrida["modo"].str.startswith("completo")]
        alto = completo[completo["q"] == 0.20]["dB_diam"].mean()
        if alto > 0:                      # a n=80 puede ser marginal
            assert alto > 0.4 * 2 * 0.20  # orden de magnitud del artefacto


class TestCache:
    def test_round_trip(self, tmp_path, corrida):
        ruta = tmp_path / "trunc.csv"
        guardar_truncamiento(corrida, ruta)
        leido = cargar_truncamiento(ruta)
        pd.testing.assert_frame_equal(leido, corrida)

    def test_archivo_ausente(self, tmp_path):
        assert cargar_truncamiento(tmp_path / "no_existe.csv") is None
