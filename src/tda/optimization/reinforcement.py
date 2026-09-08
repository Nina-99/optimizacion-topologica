"""Módulo para el Rediseño de Refuerzo (Aplicación 2 - Fase 2).

Ejecuta el rediseño SIMP sobre el dominio del puente Z24 dañado, 
evaluando las configuraciones A, B, C, D y E del Cuadro 8.
"""

import numpy as np
from tda.optimization.metric_simp import MetricaTDA_SIMP

def evaluar_configuracion_refuerzo(config_name: str, p: float, f_V: float, 
                                 nex: int = 80, ney: int = 40, max_iter: int = 100):
    """
    Evalúa una configuración de refuerzo para el puente Z24.
    
    Dominio: 8 x 4 m (8000 x 4000 mm). Malla: 80 x 40.
    E_0 = 70 GPa (Fibra de Carbono) -> 70,000 MPa.
    
    Retorna resultados del diseño y la métrica compuesta.
    """
    # Material: Fibra de carbono
    E_cf = 70000.0  # MPa
    Lx = 8000.0     # mm
    Ly = 4000.0     # mm
    t = 10.0        # mm (espesor de la chapa de refuerzo)
    
    opt = MetricaTDA_SIMP(
        nex=nex, ney=ney, E=E_cf, nu=0.3,
        Lx=Lx, Ly=Ly, t=t,
        f_V=f_V, p=p, r_min=3.2, 
        alpha=0.018,  # Calibrado para la App 2
        tol=1e-4, max_iter=max_iter
    )
    
    # Carga de tráfico en el centro superior del dominio de refuerzo
    # F_y = -100 kN = -100,000 N
    nnx = nex + 1
    nny = ney + 1
    n_dof = 2 * nnx * nny
    
    F = np.zeros(n_dof)
    node_load = nex // 2  # Centro del borde superior (y=0)
    F[2 * node_load + 1] = -100000.0
    
    # Empotramientos laterales (los anclajes a la viga sana)
    dofs_fijos = []
    for j in range(nny):
        # Borde izquierdo (x=0) y borde derecho (x=nex)
        dofs_fijos.extend([2 * (j * nnx), 2 * (j * nnx) + 1])
        dofs_fijos.extend([2 * (j * nnx + nex), 2 * (j * nnx + nex) + 1])
        
    opt.definir_problema(F, dofs_fijos)
    opt.optimizar(verbose=False)
    opt.fase_tda(verbose=False)
    
    return opt.obtener_resultados()

def generar_cuadro_8_refuerzos():
    """Ejecuta las 5 configuraciones del Cuadro 8."""
    configs = {
        'A': {'p': 2, 'f_V': 0.30},
        'B': {'p': 3, 'f_V': 0.30},
        'C': {'p': 4, 'f_V': 0.30},
        'D': {'p': 3, 'f_V': 0.25},
        'E': {'p': 3, 'f_V': 0.35},
    }
    
    resultados = {}
    for nombre, params in configs.items():
        resultados[nombre] = evaluar_configuracion_refuerzo(
            config_name=nombre, 
            p=params['p'], 
            f_V=params['f_V'],
            max_iter=80  # Limitar iteraciones para la UI interactiva
        )
        
    return resultados
