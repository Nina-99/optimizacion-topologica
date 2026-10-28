"""Smoke tests de la Figura 5.3 (truncamiento + cota probabilística).

Datos sintéticos: no corren ripser. Verifican los dos paneles, el conteo de
violaciones del panel (b) y el manejo de los dos temas.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from tda.visualization.figuras import fig_53_cota_corregida

MODOS = ("truncado (ε ≤ 0.6, barras infinitas descartadas)",
         "completo (sin truncar)")


def _truncamiento() -> pd.DataFrame:
    filas = []
    for modo in MODOS:
        for q in (0.0, 0.05, 0.10, 0.15, 0.20):
            for r in range(3):
                base = 0.05 if modo.startswith("truncado") else 0.4
                filas.append({"modo": modo, "q": q, "replica": r,
                              "dB_diam": 0.0 if q == 0 else base * q / 0.2})
    return pd.DataFrame(filas)


def _replicas(violar: bool = False) -> pd.DataFrame:
    filas = []
    for q in (0.05, 0.10, 0.15, 0.20):
        for r in range(10):
            cota = 1.0
            filas.append({"q": q, "dB": 1.5 if violar else 0.3 + 0.01 * r,
                          "cota_prob": cota})
    return pd.DataFrame(filas)


class TestFigura53:
    def test_dos_paneles(self):
        fig = fig_53_cota_corregida(_truncamiento(), _replicas())
        assert len(fig.axes) == 2

    def test_titulos(self):
        fig = fig_53_cota_corregida(_truncamiento(), _replicas())
        titulos = [a.get_title(loc="left") for a in fig.axes]
        assert "Artefacto del truncamiento" in titulos[0]
        assert "Cota probabilística" in titulos[1]

    def test_sin_violaciones(self):
        fig = fig_53_cota_corregida(_truncamiento(), _replicas())
        assert "0 violaciones" in fig.axes[1].get_title(loc="left")

    def test_con_violaciones(self):
        fig = fig_53_cota_corregida(_truncamiento(), _replicas(violar=True))
        titulo = fig.axes[1].get_title(loc="left")
        assert ": 0 violaciones" not in titulo
        assert "40 violaciones" in titulo

    def test_banda_por_modo(self):
        """Cada modo aporta una curva de medias además de la cota 2q."""
        fig = fig_53_cota_corregida(_truncamiento(), _replicas())
        etiquetas = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert len(etiquetas) == 3                     # cota + 2 modos
        assert any("truncado" in e for e in etiquetas)
        assert any("completo" in e for e in etiquetas)

    def test_tema_oscuro(self):
        fig = fig_53_cota_corregida(_truncamiento(), _replicas(), tema="oscuro")
        assert len(fig.axes) == 2
