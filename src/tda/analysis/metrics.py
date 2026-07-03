"""Métricas de clasificación y verificación topológica para TDA.

Proporciona funciones para evaluar accuracy de K-Means (con manejo de
etiquetas intercambiadas) y verificar números de Betti esperados.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score


def compute_kmeans_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula accuracy de K-Means manejando el intercambio de etiquetas.

    K-Means asigna etiquetas 0/1 arbitrariamente, que pueden estar
    intercambiadas respecto a ground truth. Esta función prueba ambas
    asignaciones y devuelve la mejor.

    Args:
        y_true: Etiquetas reales (0, 1).
        y_pred: Etiquetas predichas por K-Means (0, 1).

    Returns:
        Accuracy en el rango [0, 1].
    """
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return float(max(acc1, acc2))


def verify_betti_numbers(
    b0_s: int, b1_s: int,
    b0_t: int, b1_t: int
) -> dict:
    """Verifica que los números de Betti coincidan con los esperados.

    Esfera esperada: β₀ = 1, β₁ = 0
    Toro esperado:   β₀ = 1, β₁ = 2

    Args:
        b0_s: β₀ de la esfera.
        b1_s: β₁ de la esfera.
        b0_t: β₀ del toro.
        b1_t: β₁ del toro.

    Returns:
        Dict con flags de corrección y mensaje descriptivo.
    """
    betti_correct = (b0_s == 1 and b1_s == 0 and b0_t == 1 and b1_t == 2)
    b0_stable = (b0_s == 1 and b0_t == 1)
    
    return {
        "tda_correct": betti_correct,
        "b0_stable": b0_stable,
        "b1_toro_ok": b1_t == 2,
        "b1_esfera_ok": b1_s == 0,
        "message": (
            "✓ TDA correcto" if betti_correct else "✗ TDA incorrecto"
        )
    }
