"""Tests de la calibración de τ (Figura 5.1, Cap. V).

Cubre tres cosas:
  1. la fórmula (Def. 8.1): β₀ incluye la componente esencial, β₁ cuenta ciclos;
  2. el catálogo corregido: geometría y RNG exactos (incluido el truncado 2σ);
  3. el protocolo completo: reproducción de los valores del documento.
"""

import numpy as np
import pandas as pd
import pytest

from tda.analysis.calibracion import (
    CATALOGO_CORREGIDO,
    ORDEN_CALIBRACION,
    SEED_MEDICION,
    TAUS,
    betti_tau,
    calibrar_c0,
    curva_calibracion,
    margenes,
    persistencias_relativas,
    resultado_calibracion,
    tau_estrella,
)
from tda.processing.sampling import generar_corregida, incrustar

# ═══════════════════════════════════════════════════════════════════
# 1. Fórmula — Def. 8.1
# ═══════════════════════════════════════════════════════════════════

def _dgm(h0, h1):
    """Salida sintética con la forma de ripser: lista de diagramas."""
    return [np.asarray(h0, dtype=float).reshape(-1, 2),
            np.asarray(h1, dtype=float).reshape(-1, 2)]


def test_betti_tau_suma_la_componente_esencial():
    """β₀ = 1 + nº de barras finitas con persistencia > τ·diam."""
    dgm = _dgm([[0.0, np.inf], [0.0, 0.5]], [[0.2, 0.5]])
    # diam = 1, τ = 0.1 → umbral 0.1: la barra [0, 0.5] pasa, la [0.2, 0.5] pasa
    assert betti_tau(dgm, tau=0.1, diam=1.0) == (2, 1)
    # τ = 0.6 → umbral 0.6: ninguna pasa, pero la esencial sigue ahí
    assert betti_tau(dgm, tau=0.6, diam=1.0) == (1, 0)


def test_betti_tau_es_invariante_a_homotecia():
    """Escalar la nube escala igual la persistencia y diam(X): misma lectura."""
    base = _dgm([[0.0, np.inf], [0.0, 0.5]], [[0.2, 0.5]])
    escalado = _dgm([[0.0, np.inf], [0.0, 5.0]], [[2.0, 5.0]])
    assert betti_tau(base, tau=0.1, diam=1.0) == betti_tau(escalado, tau=0.1, diam=10.0)


def test_persistencias_relativas_excluye_la_barra_infinita():
    """La barra infinita no es un rasgo: si entra, el margen da ±inf."""
    p0, p1 = persistencias_relativas(_dgm([[0.0, np.inf], [0.0, 0.4]], [[0.2, 0.5]]), diam=2.0)
    assert np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))
    assert p0[0] == pytest.approx(0.2)      # 0.4 / 2
    assert p1[0] == pytest.approx(0.15)     # 0.3 / 2


# ═══════════════════════════════════════════════════════════════════
# 2. Catálogo corregido
# ═══════════════════════════════════════════════════════════════════

# Diámetros esperados (n = 400, semillas 42 + i, catálogo de medición C) —
# son los que fijan τ·diam(X) en el panel (c) del documento.
_DIAM_ESPERADO = {
    "circulo": 2.0000, "anillo": 1.9951, "dos_circulos": 4.1996,
    "toro": 5.5897, "disco": 1.9875, "esfera": 2.0000, "gaussiana": 1.9798,
}


@pytest.mark.parametrize("shape", ORDEN_CALIBRACION)
def test_generar_corregida_reproduce_la_geometria_del_documento(shape):
    from scipy.spatial.distance import pdist

    i = ORDEN_CALIBRACION.index(shape)
    X = generar_corregida(shape, 400, np.random.default_rng(SEED_MEDICION + i))
    assert X.shape == (400, 3 if shape in ("toro", "esfera") else 2)
    assert pdist(X).max() == pytest.approx(_DIAM_ESPERADO[shape], abs=1e-3)


def test_generar_corregida_es_determinista_por_semilla():
    """Una nube no puede alterar a la siguiente: RNG por nube, no global."""
    a = generar_corregida("circulo", 50, np.random.default_rng(7))
    b = generar_corregida("circulo", 50, np.random.default_rng(7))
    c = generar_corregida("circulo", 50, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gaussiana_truncada_en_2_sigma():
    """Sin truncar, los atípicos producen β₀ ≥ 2 y el patrón deja de servir."""
    rng = np.random.default_rng(900 + ORDEN_CALIBRACION.index("gaussiana"))
    X = generar_corregida("gaussiana", 400, rng)
    assert np.linalg.norm(X, axis=1).max() <= 1.0 + 1e-12   # 2σ × 0.5


def test_generar_corregida_rechaza_forma_desconocida():
    with pytest.raises(ValueError):
        generar_corregida("cubo", 10, np.random.default_rng(0))


def test_incrustar_completa_con_ceros():
    X = generar_corregida("circulo", 10, np.random.default_rng(0))
    assert incrustar(X, 10).shape == (10, 10)
    assert np.all(incrustar(X, 10)[:, 2:] == 0)
    assert incrustar(X, 1).shape == (10, 2)   # nunca recorta


# ═══════════════════════════════════════════════════════════════════
# 3. Protocolo completo — valores del documento
# ═══════════════════════════════════════════════════════════════════

# A(τ) del catálogo C₀ (n = 400, semillas 900 + i)
_CURVA_ESPERADA = {
    0.05: 2, 0.075: 3, 0.1: 5, 0.125: 6, 0.15: 6, 0.175: 7, 0.2: 6,
    0.225: 5, 0.25: 5, 0.275: 5, 0.3: 5, 0.325: 5, 0.35: 5, 0.375: 5, 0.4: 5,
}

# Margen de cada nube de C respecto de τ* = 0.175 (n = 400, semillas 42 + i)
_MARGEN_ESPERADO = {
    "circulo": 0.124, "anillo": 0.127, "dos_circulos": 0.044, "toro": -0.002,
    "disco": 0.096, "esfera": 0.028, "gaussiana": 0.081,
}


def test_protocolo_reproduce_la_figura_5_1():
    """Regresión contra el PDF: curva, τ* único y márgenes del panel (c)."""
    res = resultado_calibracion()

    # (a) curva de calibración
    curva = {round(float(t), 3): int(v) for t, v in curva_calibracion(res["cal"]).items()}
    assert curva == _CURVA_ESPERADA

    # τ* es único: una sola ventana con 7/7
    assert res["tau"] == {"tau_star": 0.175, "ventana_min": 0.175, "ventana_max": 0.175}

    # (c) márgenes
    mg = res["margenes"].set_index("nube").margen
    assert list(mg.index) == ORDEN_CALIBRACION
    for shape, esperado in _MARGEN_ESPERADO.items():
        assert mg[shape] == pytest.approx(esperado, abs=1e-3), shape

    # el toro se lee mal ya sin ruido: su margen es negativo
    assert mg["toro"] < 0
    # y las demás sobreviven a algo de ruido
    assert (mg.drop("toro") > 0).all()


def test_tau_estrella_aborta_si_no_hay_ventana():
    """Sin lectura consistente no se elige otro τ: se aborta."""
    cal = pd.DataFrame([
        dict(nube=s, tau=0.1, b0=1, b1=1, b0_teo=1, b1_teo=1, correcto=0)
        for s in CATALOGO_CORREGIDO
    ])
    with pytest.raises(ValueError, match="ABORTAR"):
        tau_estrella(cal)


def test_curva_calibracion_respeta_la_grilla():
    cal = calibrar_c0(n_points=60, seed=900, taus=TAUS[:3])
    assert len(cal) == len(CATALOGO_CORREGIDO) * 3
    assert set(cal.correcto) <= {0, 1}
    assert list(curva_calibracion(cal).index) == list(TAUS[:3])
