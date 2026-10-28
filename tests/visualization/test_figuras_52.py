"""Smoke tests de la Figura 5.2 (5 paneles) con datos sintéticos.

No corren el protocolo: construyen un tablero mínimo con la forma que devuelve
`estabilidad_dimensional.protocolo_fig52` y verifican la estructura de la
figura (paneles, títulos, conteo de violaciones, tema).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from tda.visualization.figuras import fig_52_estabilidad

NUBES = ["circulo", "anillo", "dos_circulos", "toro", "disco", "esfera", "gaussiana"]
DIMS = [2, 3, 10, 100]
QS = [0.05, 0.10, 0.15, 0.20]


def _replicas(n_rep: int = 3, violar: bool = False) -> pd.DataFrame:
    filas = []
    rng = np.random.default_rng(0)
    for nube in NUBES:
        for d in DIMS:
            for q in QS:
                for r in range(n_rep):
                    dB = float(rng.uniform(0.02, 0.3))
                    dH = dB / 2.0 * float(rng.uniform(1.0, 2.0))   # dB ≤ 2 dH
                    if violar:
                        dB = 3.0 * dH          # entra en la región prohibida
                    filas.append({
                        "nube": nube, "d": d, "q": q, "replica": r,
                        "diam": 2.0, "sigma": q * 2.0 / np.sqrt(d),
                        "dB": dB, "dH": dH, "cota_teorema": 2.0 * dH,
                        "cota_prob": 1.0, "b0": 1, "b1": 1,
                        "b0_teo": 1, "b1_teo": 1, "acierto": int(q <= 0.10),
                    })
    return pd.DataFrame(filas)


def _q0() -> pd.DataFrame:
    return pd.DataFrame({
        "nube": NUBES,
        "diam": 2.0,
        "b0": 1, "b1": 1, "b0_teo": 1, "b1_teo": 1,
        "acierto": [1, 1, 1, 0, 1, 1, 1],
        "rasgo_real_mas_debil": 0.5, "espurio_mas_fuerte": 0.1,
        "margen_inferior": 0.3, "margen_superior": 0.1, "margen": 0.1,
    })


class TestFigura52:
    def test_cinco_paneles(self):
        fig = fig_52_estabilidad(_replicas(), _q0())
        assert len(fig.axes) == 5
        assert [a.get_title(loc="left")[:4] for a in fig.axes] == [
            "(a1)", "(a2)", "(a3)", "(b) ", "(c) "]

    def test_titulos_de_las_dimensiones(self):
        fig = fig_52_estabilidad(_replicas(), _q0())
        titulos = [a.get_title(loc="left") for a in fig.axes[:3]]
        assert "nativa" in titulos[0]
        assert "d = 10" in titulos[1]
        assert "d = 100" in titulos[2]

    def test_cuenta_las_violaciones(self):
        fig = fig_52_estabilidad(_replicas(), _q0())
        assert "0 violaciones" in fig.axes[3].get_title(loc="left")

        fig_mal = fig_52_estabilidad(_replicas(violar=True), _q0())
        assert "0 violaciones" not in fig_mal.axes[3].get_title(loc="left")

    def test_leyenda_por_dimension(self):
        fig = fig_52_estabilidad(_replicas(), _q0())
        etiquetas = [t.get_text() for t in fig.axes[3].get_legend().get_texts()]
        assert len(etiquetas) == 3
        assert any("nativa" in e for e in etiquetas)

    def test_punto_q_cero_sale_de_la_lectura_sin_ruido(self):
        """El primer punto de cada curva es el acierto sin ruido, no q = 0.05."""
        rep = _replicas()
        fig = fig_52_estabilidad(rep, _q0())
        toro = [(l.get_label(), l) for l in fig.axes[0].get_lines()
                if l.get_label() == "toro"]
        assert toro, "la curva del toro debe existir"
        _, curva = toro[0]
        assert curva.get_ydata()[0] == 0.0       # q0 del toro: falla sin ruido
        assert curva.get_xdata()[0] == 0.0

    def test_tema_oscuro_no_falla(self):
        fig = fig_52_estabilidad(_replicas(), _q0(), tema="oscuro")
        assert len(fig.axes) == 5

    def test_etiquetas_personalizadas(self):
        fig = fig_52_estabilidad(_replicas(), _q0(),
                                 etiquetas={"toro": "TORO"})
        etiquetas = [t.get_text() for t in fig.legends[0].get_texts()]
        assert "TORO" in etiquetas
