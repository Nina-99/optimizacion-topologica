"""Módulo central de análisis topológico y núcleo FEM.

Contiene:
- fem:      Núcleo FEM (matriz de rigidez, ensamble, solver, sensibilidades, filtro OC)
- topology: Funciones topológicas (Betti, distancias, binarización, homología persistente)
            — para nubes de puntos (TDA)
- betti2d:  Cálculo de β₀, β₁ de diseños SIMP por doble cómputo
            (Euler 4-conexo + GUDHI cubical) — §8.5 del Perfil
- metric:   Métrica compuesta μ_α y calibración de α*
"""

from tda.core.fem import (
    calcular_K_elemental,
    ensamblar_K_global,
    resolver_FEM,
    calcular_compliance_sensibilidades,
    filtrar_sensibilidades,
    actualizar_OC,
)

from tda.core.topology import (
    wasserstein_distance,
    bottleneck_distance,
    betti_numbers,
    binarizar_y_extraer_nube,
    escala_adaptativa,
    calcular_homologia_betti,
)

from tda.core.betti2d import (
    betti_euler,
    betti_gudhi,
    betti_doble_computo,
    diagramas_gudhi,
)

from tda.core.metric import (
    metrica_compuesta,
    calibrar_alpha_optimo,
)

__all__ = [
    # FEM
    "calcular_K_elemental",
    "ensamblar_K_global",
    "resolver_FEM",
    "calcular_compliance_sensibilidades",
    "filtrar_sensibilidades",
    "actualizar_OC",
    # Topología (nubes de puntos — TDA)
    "wasserstein_distance",
    "bottleneck_distance",
    "betti_numbers",
    "binarizar_y_extraer_nube",
    "escala_adaptativa",
    "calcular_homologia_betti",
    # Betti 2D (diseños SIMP — §8.5)
    "betti_euler",
    "betti_gudhi",
    "betti_doble_computo",
    "diagramas_gudhi",
    # Métrica
    "metrica_compuesta",
    "calibrar_alpha_optimo",
]
