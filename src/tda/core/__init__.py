"""Módulo central de análisis topológico y núcleo FEM.

Contiene:
- fem:     Núcleo FEM (matriz de rigidez, ensamble, solver, sensibilidades, filtro OC)
- topology: Funciones topológicas (Betti, distancias, binarización, homología persistente)
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
    # Topología
    "wasserstein_distance",
    "bottleneck_distance",
    "betti_numbers",
    "binarizar_y_extraer_nube",
    "escala_adaptativa",
    "calcular_homologia_betti",
    # Métrica
    "metrica_compuesta",
    "calibrar_alpha_optimo",
]
