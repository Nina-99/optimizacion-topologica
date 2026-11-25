"""Leyendas breves de las figuras del Cap. V, tomadas del PDF.

Una sola fuente para la app, los scripts de réplica y cualquier exportación.
"""

from __future__ import annotations

LEYENDAS_FIGURAS = {
    "5.1": (
        "Figura 5.1. (a) Nubes del catálogo de calibración C₀ con lectura correcta "
        "de (β₀, β₁) frente al umbral relativo τ. (b) Acierto por nube en cada τ. "
        "(c) Margen de cada nube del catálogo C, sin ruido, respecto de τ*."
    ),
    "5.2": (
        "Figura 5.2. (a1–a3) Tasa de acierto de (β₀, β₁) frente al ruido normalizado "
        "q (σ = q·diam X/√d), por nube y dimensión ambiente. (b) d_B frente a 2d_H, "
        "normalizados por diam X, en las 1 680 réplicas. (c) Razón de ajuste d_B/(2d_H)."
    ),
    "5.3": (
        "Figura 5.3. (a) d_B/diam X para el círculo con la definición de ruido anterior, "
        "con filtración truncada en ε ≤ 0,6 y completa; la banda es el rango de las 20 "
        "réplicas. (b) d_B dividida por la cota probabilística "
        "2σ(√d + √(2 ln(n/δ))), δ = 0,05, por nivel de ruido."
    ),
    "5.4": (
        "Figura 5.4. (a) Diseño convergido del voladizo 60×30 (p = 3, f_V = 0,5, "
        "r_min = 2,4); el contorno marca el umbral u_V que conserva el volumen. "
        "(b) Historial de compliance (escala logarítmica) con los referentes físico, "
        "binarizado y cota convexa."
    ),
    "5.5": (
        "Figura 5.5. (a) Compliance penalizada y del diseño binarizado frente a p "
        "(escala logarítmica). (b) Grisura media 4ρ(1−ρ); bajo cada p, β₁ del binarizado."
    ),
    "5.6": (
        "Figura 5.6. (a) Diseños convergidos del voladizo con tres resoluciones y r_min "
        "físico constante (r_min = 2,4; 4,8; 7,2). (b) Compliance penalizada y del "
        "binarizado frente a la malla."
    ),
    "5.7": (
        "Figura 5.7. (a) β₀ y β₁ del diseño base frente al umbral u. (b) Fracción de "
        "volumen del binarizado. (c) Diagrama de persistencia H₁ de la filtración por "
        "subniveles de f = 1 − ρ*; la banda contiene las barras que una perturbación "
        "con ‖f − g‖∞ ≤ 0,45 podría borrar."
    ),
    "5.8": (
        "Figura 5.8. β₁ del diseño binarizado (u = 0,5) frente al radio del filtro en "
        "el voladizo 60×30 y en la viga MBB 120×40. Los círculos abiertos marcan "
        "corridas que alcanzaron el tope de 300 iteraciones."
    ),
}


def leyenda(numero: str) -> str:
    """Devuelve la leyenda de una figura ('5.1' … '5.8')."""
    return LEYENDAS_FIGURAS[str(numero)]
