"""Métrica compuesta TDA-SIMP y calibración de α*.

Implementa la Definición 1.9 y la Proposición 1.1 del capítulo:
    μ_α(ρ*) = c(ρ*) + α · β_1(Ω_sólido)

Bloque 4 del Algoritmo 1 (Métrica Compuesta TDA-SIMP).
"""

import numpy as np


def metrica_compuesta(c, beta1, alpha):
    """
    Calcula la métrica compuesta TDA-SIMP (Definición 1.9):

        μ_α(ρ*) = c(ρ*) + α · β_1(Ω_sólido)

    Propiedades (Teorema 1.1):
     (i)  Bien definida: c > 0 y β_1 ≥ 0 → μ_α > 0
     (ii) Positiva: μ_α ≥ c_min > 0
     (iii) Monotonía topológica: menor β_1 → menor μ_α (con c fija)
     (iv) Robusta: perturbaciones acotadas → cambios acotados en β_1

    Parámetros
    ──────────
    c     : float  Compliance mecánica c(ρ*)
    beta1 : int    Número de Betti β_1 de la topología resultante
    alpha : float  Peso de ponderación α > 0

    Retorna
    ───────
    mu : float  Valor escalar de la métrica compuesta
    """
    mu = c + alpha * float(beta1)
    return mu


def calibrar_alpha_optimo(c_vals, beta1_vals):
    """
    Calcula α* que equilibra las contribuciones mecánica y topológica
    (Proposición 1.1):

        α* = (c̄ - c_min) / (β_{1,max} - β_{1,min})

    Parámetros
    ──────────
    c_vals    : array-like  Compliance de M diseños de referencia
    beta1_vals: array-like  β_1 de M diseños de referencia

    Retorna
    ───────
    alpha_star : float  Peso óptimo α* (positivo)
    """
    c_arr  = np.array(c_vals, dtype=float)
    b1_arr = np.array(beta1_vals, dtype=float)

    c_media   = c_arr.mean()
    c_min     = c_arr.min()
    b1_rango  = b1_arr.max() - b1_arr.min()

    if b1_rango < 1e-10:
        return 0.01

    alpha_star = (c_media - c_min) / b1_rango
    return float(max(alpha_star, 1e-4))
