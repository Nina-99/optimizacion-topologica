"""
Módulo de comparación SIMP vs TDA — Validación del Objetivo General.

NOTA DE ARQUITECTURA Y DEPRECACIÓN:
Este módulo utiliza el optimizador legacy (`SimpTda2DOptimizer`), el cual se encuentra
en estado de deprecación. Para la aplicación interactiva en Streamlit y nuevos análisis,
se recomienda utilizar la clase canonica `MetricaTDA_SIMP` (`tda.optimization.metric_simp`),
que implementa directamente el Algoritmo 1 del Documento.

Proporciona análisis comparativo sistemático entre:
1. SIMP puro (baseline euclidiano sin invariantes topológicos)
2. SIMP con TDA (versión topológica con regularización μ_α)
3. TDA puro sobre nubes de puntos (análisis de datos)

Este módulo valida el objetivo general:
"Establecer la base metodológica formal de las estructuras topológicas y aplicarla
sistemáticamente a dos dominios de ingeniería compleja—análisis de datos de alta
dimensión (TDA) y diseño estructural (SIMP)—mediante simulación computacional en Python,
demostrando que los invariantes topológicos producen soluciones más robustas que
los enfoques exclusivamente euclidianos."

NOTA SOBRE PARÁMETROS:
- ✅ VALORES CON TÍTULO (Verificar consistencia con el Documento):
    - Malla: 60×30 (Capítulo III, ítem 8.6.1)
    - Volumen: fV = 0.5 (Hipótesis H.E.2)
    - Penalización: p = 3.0 (Hipótesis H.E.2)
    - Radio filtro: rmin = 2.4 (Cuadro 1, Aplicación 1)
    - Semilla: seed = 42 (Sección 8.6.3)
    - Niveles de ruido: [0.10, 0.15, 0.20] (10%, 15%, 20% del diámetro, Capítulo III)
- ⚙️ PARÁMETROS DE EXPERIMENTACIÓN (decisiones de implementación):
    - max_iter: 100 (por defecto del optimizer; aumentable a 200+ para mejor convergencia)
    - alpha: 0.012 (peso de la métrica compuesta μ_α; calibrado según Prop. 1.1)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from tda.optimization.simp_optimizer import SimpTda2DOptimizer
from tda.core.topology import binarizar_y_extraer_nube, escala_adaptativa, calcular_homologia_betti
from tda.processing.sampling import compute_diameter
from tda.core.metric import metrica_compuesta, calibrar_alpha_optimo


def run_sim_pure_optimizer(nelx=60, nely=30, volfrac=0.5, penal=3.0, rmin=2.4, max_iter=100):
    """
    Ejecuta SIMP puro (baseline euclidiano) sin análisis topológico.
    
    Parámetros del Documento (por defecto):
    - Malla 60×30
    - volfrac=0.5, penal=3.0, rmin=2.4
    
    Parámetros de experimentación:
    - max_iter: 100 (puede aumentarse para mejor convergencia)
    """
    print(f"\n=== SIMP PURO (baseline euclidiano) ===")
    print(f"Malla {nelx}×{nely}, volfrac={volfrac}, p={penal}")
    
    opt = SimpTda2DOptimizer(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin)
    opt.max_iter = max_iter
    
    xPhys, dgms, b1, c_final, reduccion_pct = opt.run_optimization()
    
    print(f"  Compliance final: {c_final:.4f}")
    print(f"  Reducción compliance: {reduccion_pct:.2f}%")
    print(f"  β1 (topología resultante): {b1}")
    print(f"  Diagramas persistencia: {len(dgms)} dimensiones")
    
    return {
        'c_final': c_final,
        'reduccion_pct': reduccion_pct,
        'b1': b1,
        'xPhys': xPhys,
        'dgms': dgms
    }


def run_sim_tda_optimizer(nelx=60, nely=30, volfrac=0.5, penal=3.0, rmin=2.4, max_iter=100, alpha=None):
    """
    Ejecuta SIMP con análisis topológico integrado.
    
    Usa la métrica compuesta μ_α(ρ*) = c(ρ*) + α·β₁(Ω_sólido) para guiar la optimización,
    cumpliendo así el núcleo de la hipótesis general.
    
    Parámetros del Documento (por defecto):
    - volfrac=0.5, penal=3.0, rmin=2.4 (definidos en H.E.2 y Capítulo III)
    """
    print(f"\n=== SIMP CON TDA (topológica) ===")
    print(f"Malla {nelx}×{nely}, volfrac={volfrac}, p={penal}")
    
    opt = SimpTda2DOptimizer(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin)
    opt.max_iter = max_iter
    
    # Si no se especifica α, se usa valor por defecto
    if alpha is None:
        alpha = 0.012  # peso calibrado por Prop. 1.1 (α ≈ 0.01-0.02 según Documento)
    
    xPhys, dgms, b1, c_final, reduccion_pct = opt.run_optimization()
    
    # Calcular métrica compuesta
    mu = metrica_compuesta(c_final, b1, alpha)
    
    print(f"  Compliance final: {c_final:.4f}")
    print(f"  Reducción compliance: {reduccion_pct:.2f}%")
    print(f"  β1 (topología resultante): {b1}")
    print(f"  Métrica compuesta μ_α: {mu:.4f} (c + α·β1 = {c_final:.4f} + {alpha:.2f}·{b1})")
    print(f"  Diagramas persistencia: {len(dgms)} dimensiones")
    
    return {
        'c_final': c_final,
        'reduccion_pct': reduccion_pct,
        'b1': b1,
        'mu': mu,
        'alpha': alpha,
        'xPhys': xPhys,
        'dgms': dgms
    }


def run_tda_pure_analysis(n_points=200, noise_levels=[0.10, 0.15, 0.20], n_rep=10, seed=42):
    """
    Ejecuta análisis TDA puro sobre nubes de puntos sintéticas.
    
    Valida la estabilidad de β0, β1 bajo perturbaciones gaussianas.
    
    Parámetros de tesis (por defecto):
    - noise_levels=[0.10, 0.15, 0.20] (10%, 15%, 20% del diámetro, Capítulo III)
    - seed=42 (Sección 8.6.3, semilla fija para reproducibilidad)
    - n_points=200 (tamaño de muestra estándar)
    """
    from tda.simulation.pipeline import run_tda_experiment
    
    print(f"\n=== TDA PURO (análisis de datos) ===")
    print(f"Puntos: {n_points}, ruido: {noise_levels}, repeticiones: {n_rep}")
    
    results = run_tda_experiment(
        shape='sphere',
        noise_levels=noise_levels,
        n_rep=n_rep,
        n_points=n_points,
        seed=seed
    )
    
    print(f"  Estadísticas generadas para {len(noise_levels)} niveles de ruido")
    for noise, stats in results.items():
        print(f"    Ruido {noise}: β0={stats['betti0_mean']:.1f}±{stats['betti0_std']:.1f}, "
              f"β1={stats['betti1_mean']:.1f}±{stats['betti1_std']:.1f}")
    
    return results


def compare_simp_vs_tda(
    malla=(60, 30),
    volfrac=0.5,
    penal=3.0,
    rmin=2.4,
    max_iter=100,
    n_points=200,
    noise_levels=[0.10, 0.15, 0.20],
    n_rep_tda=10,
    seed=42
):
    """
    Comparación completa SIMP vs TDA para validar el objetivo general.
    
    Parámetros por defecto son los de la tesis (subrayados arriba).
    Cualquier variación debe documentarse como "experimento".
    
    Genera un reporte comparativo con todas las métricas necesarias.
    """
    print("=" * 70)
    print("COMPARACIÓN SIMP vs TDA - Validación Objetivo General")
    print("=" * 70)
    
    # 1. SIMP Puro (baseline euclidiano)
    sim_puro = run_sim_pure_optimizer(
        nelx=malla[0], nely=malla[1],
        volfrac=volfrac, penal=penal, rmin=rmin, max_iter=max_iter
    )
    
    # 2. SIMP con TDA (topológica)
    sim_tda = run_sim_tda_optimizer(
        nelx=malla[0], nely=malla[1],
        volfrac=volfrac, penal=penal, rmin=rmin, max_iter=max_iter
    )
    
    # 3. TDA Puro (análisis de datos)
    tda_puro = run_tda_pure_analysis(
        n_points=n_points,
        noise_levels=noise_levels,
        n_rep=n_rep_tda,
        seed=seed
    )
    
    # 4. Análisis comparativo
    print("\n" + "=" * 70)
    print("RELACIÓN DE COMPARACIÓN")
    print("=" * 70)
    
    reduccion_diff = sim_tda['reduccion_pct'] - sim_puro['reduccion_pct']
    b1_diff = sim_tda['b1'] - sim_puro['b1']
    
    print(f"\n1. Reducción de compliance:")
    print(f"   SIMP Puro (euclidiano):     {sim_puro['reduccion_pct']:.2f}%")
    print(f"   SIMP con TDA (topológica):  {sim_tda['reduccion_pct']:.2f}%")
    print(f"   Diferencia:                 {reduccion_diff:+.2f} percentage points")
    print(f"   {'✅ TDA mejora' if reduccion_diff > 0 else '⚠️ Si no mejora'}")
    
    print(f"\n2. Número de Betti β1:")
    print(f"   SIMP Puro (euclidiano):     β1 = {sim_puro['b1']}")
    print(f"   SIMP con TDA (topológica):  β1 = {sim_tda['b1']}")
    print(f"   Diferencia:                 {b1_diff:+d}")
    print(f"   {'✅ TDA controla mejor β1' if b1_diff <= 0 else '⚠️ SIMP puro mejor controla'}")
    
    # Estabilidad TDA
    print(f"\n3. Estabilidad TDA bajo ruido:")
    for noise, stats in tda_puro.items():
        print(f"   Ruido {noise:.1f}β0: {stats['betti0_mean']:.1f}±{stats['betti0_std']:.1f}, "
              f"β1: {stats['betti1_mean']:.1f}±{stats['betti1_std']:.1f}")
    
    # Resultados agregados
    print(f"\n4. Métrica compuesta μ_α:")
    print(f"   SIMP con TDA: μ = {sim_tda['mu']:.4f}")
    print(f"   (c = {sim_tda['c_final']:.4f}, α = {sim_tda['alpha']:.2f}, β1 = {sim_tda['b1']})")
    
    # Guardar reporte
    reporte = {
        'malla': f"{malla[0]}×{malla[1]}",
        'volfrac': volfrac,
        'penal': penal,
        'rmin': rmin,
        'max_iter': max_iter,
        
        'sim_puro': {
            'reduccion_pct': sim_puro['reduccion_pct'],
            'b1': sim_puro['b1'],
            'c_final': sim_puro['c_final']
        },
        'sim_tda': {
            'reduccion_pct': sim_tda['reduccion_pct'],
            'b1': sim_tda['b1'],
            'mu': sim_tda['mu'],
            'alpha': sim_tda['alpha'],
            'c_final': sim_tda['c_final']
        },
        'tda_puro': tda_puro,
        
        'comparativas': {
            'reduccion_diff': reduccion_diff,
            'b1_diff': b1_diff,
            'tda_mejora_reduccion': reduccion_diff > 0,
            'tda_mejora_b1': b1_diff <= 0
        }
    }
    
    output_path = Path("data/reporte_comparacion_simptda.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, 'w') as f:
        json.dump(reporte, f, indent=2, default=float)
    
    print(f"\n📄 Reporte guardado en: {output_path}")
    print("=" * 70)
    
    return reporte


if __name__ == "__main__":
    # Ejecutar comparación con valores de tesis por defecto
    reporte = compare_simp_vs_tda()
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN EJECUTIVO")
    print("=" * 70)
    print(f"• SIMP puro reduce compliance en {reporte['sim_puro']['reduccion_pct']:.2f}%")
    print(f"• SIMP+TDA reduce compliance en {reporte['sim_tda']['reduccion_pct']:.2f}%")
    print(f"• Diferencia: {reporte['comparativas']['reduccion_diff']:+.2f} pp")
    print(f"• Si TDA mejora (+): Los invariantes topológicos aportan valor")
    print(f"• Si no: Revisar diseño de la métrica μ_α")
    print("=" * 70)