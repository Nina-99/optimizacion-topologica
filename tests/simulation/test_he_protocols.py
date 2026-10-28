"""Tests del protocolo H.E.1 — τ calibrado y curva de calibración.

El protocolo de la app (páginas H.E.1 y H.G.) tiene que usar el MISMO τ que
sale de la Figura 5.1: si `TAU_CALIBRADO` y la calibración se desincronizan,
la página muestra una ventana que no contiene al τ con el que mide.
"""

import numpy as np

from tda.analysis.calibracion import calibrar_c0, curva_calibracion, tau_estrella
from tda.simulation.he_protocols import TAU_CALIBRADO, calibrar_tau


def test_tau_calibrado_es_el_tau_estrella_del_documento():
    """τ adoptado = τ* de la Fig. 5.1 (única ventana con 7/7 en C₀)."""
    assert TAU_CALIBRADO == 0.175


def test_tau_calibrado_es_coherente_con_la_calibracion():
    """Coherencia dura: el τ del protocolo cae dentro de la ventana de C₀."""
    cal = calibrar_c0()
    tau = tau_estrella(cal)
    assert tau["ventana_min"] <= TAU_CALIBRADO <= tau["ventana_max"]

    # La curva corregida usa β₀ Y β₁ sobre 15 umbrales (no solo β₁).
    curva = {round(float(t), 3): int(v) for t, v in curva_calibracion(cal).items()}
    assert len(curva) == 15
    assert curva[0.175] == 7 and max(curva.values()) == 7   # única ventana 7/7
    assert curva[0.15] == 6                                 # esfera aún falla
    assert curva[0.05] == 2                                 # con solo β₁ daba 7


def test_calibrar_tau_devuelve_grilla_y_conteos():
    """La envoltura que usan las páginas: (taus, aciertos) alineados."""
    taus, aciertos = calibrar_tau(60)
    assert len(taus) == len(aciertos) == 15
    assert np.issubdtype(np.asarray(taus).dtype, np.floating)
    assert set(np.asarray(aciertos).tolist()) <= {0, 1, 2, 3, 4, 5, 6, 7}
    assert taus[0] == 0.05 and taus[-1] == 0.4
