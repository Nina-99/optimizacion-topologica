"""Métricas de clasificación y verificación topológica para TDA.

Proporciona funciones para evaluar exactitud de K-Means (con manejo de
etiquetas intercambiadas) y verificar números de Betti esperados.
"""

from typing import Tuple, Dict
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import accuracy_score


def compute_kmeans_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la exactitud (accuracy) de K-Means manejando el intercambio de etiquetas.

    K-Means asigna etiquetas 0/1 arbitrariamente, que pueden estar
    intercambiadas respecto a ground truth. Esta función prueba ambas
    asignaciones y devuelve la mejor.

    Args:
        y_true: Etiquetas reales (0, 1).
        y_pred: Etiquetas predichas por K-Means (0, 1).

    Returns:
        Exactitud en el rango [0, 1].
    """
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return float(max(acc1, acc2))


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_tda: np.ndarray,
    y_pred_euclidean: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """Aplica el test de McNemar para comparar dos clasificadores binarios.

    Construye la tabla de contingencia 2×2:

                        Euclidian correcto   Euclidian incorrecto
    TDA correcto             n00                    n01
    TDA incorrecto           n10                    n11

    Estadístico de McNemar (sin corrección):
        χ² = (n01 - n10)² / (n01 + n10)

    Ref: §9.4.4 del documento — comparación de clasificadores.

    Args:
        y_true: Etiquetas reales (0, 1).
        y_pred_tda: Predicciones del clasificador TDA (K-Means + Betti).
        y_pred_euclidean: Predicciones del clasificador euclidiano (solo K-Means).
        alpha: Nivel de significancia (default 0.05).

    Returns:
        Dict con:
            contingency_table: np.ndarray (2,2) — tabla de contingencia
            n00: int — ambos correctos
            n01: int — TDA correcto, Eucl incorrecto
            n10: int — Eucl correcto, TDA incorrecto
            n11: int — ambos incorrectos
            statistic: float — χ² de McNemar (None si no aplicable)
            p_value: float — valor p (None si no aplicable)
            significant: bool — True si rechaza H₀ a nivel alpha
            message: str — interpretación legible
    """
    y_true = np.asarray(y_true)
    y_pred_tda = np.asarray(y_pred_tda)
    y_pred_euclidean = np.asarray(y_pred_euclidean)

    correct_tda = y_pred_tda == y_true
    correct_eucl = y_pred_euclidean == y_true

    n00 = int(np.sum(correct_tda & correct_eucl))
    n01 = int(np.sum(correct_tda & ~correct_eucl))
    n10 = int(np.sum(~correct_tda & correct_eucl))
    n11 = int(np.sum(~correct_tda & ~correct_eucl))

    contingency = np.array([[n00, n01], [n10, n11]])

    b_plus_c = n01 + n10
    if b_plus_c == 0:
        return {
            "contingency_table": contingency,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "statistic": None,
            "p_value": None,
            "significant": False,
            "message": (
                "Test de McNemar no aplicable: ambos clasificadores tienen "
                "exactamente los mismos aciertos y errores (b + c = 0)."
            ),
        }

    statistic = (n01 - n10) ** 2 / b_plus_c
    p_value = chi2.sf(statistic, df=1)
    significant = p_value < alpha

    if n01 > n10:
        direction = "TDA clasifica correctamente más muestras que Euclídeo"
    elif n10 > n01:
        direction = "Euclídeo clasifica correctamente más muestras que TDA"
    else:
        direction = "Ambos clasificadores tienen el mismo número de aciertos diferenciados"

    message = (
        f"χ² = {statistic:.4f}, p = {p_value:.6f}. "
        f"{'Se rechaza H₀' if significant else 'No se rechaza H₀'} "
        f"al nivel α = {alpha}. {direction}."
    )

    return {
        "contingency_table": contingency,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": significant,
        "message": message,
    }


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
