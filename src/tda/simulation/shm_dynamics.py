"""Módulo de Dinámica y TDA Continuo para Monitoreo de Salud Estructural (SHM).

Implementa la Aplicación 2 (Fase 1) de la tesis:
1. Generación de señales sintéticas de aceleración del Puente Z24.
2. Teorema de Embedding de Takens para reconstruir el espacio de fases.
3. Cálculo del Indicador de Daño (I_D) usando distancia de Wasserstein.
"""

import numpy as np
from ripser import ripser
import persim

def generar_senal_z24(nivel_dano: int, n_points: int = 8000, fs: float = 100.0) -> np.ndarray:
    """
    Genera una serie temporal sintética de aceleración a_k(t) basada en 
    las frecuencias modales típicas del puente Z24.
    
    El nivel de daño (0 a 6) altera la frecuencia del primer modo y 
    añade no-linealidades (armónicos y ruido estructural).
    
    Parámetros
    ----------
    nivel_dano : int
        Nivel de daño de 0 (Sano) a 6 (Fallo inminente).
    n_points : int
        Número de muestras (default 8000).
    fs : float
        Frecuencia de muestreo en Hz.
        
    Retorna
    -------
    a_t : ndarray
        Serie temporal de aceleraciones.
    """
    t = np.arange(n_points) / fs
    
    # Frecuencias modales base del puente Z24 (aprox)
    f1_base = 3.8  # Modo flexional 1
    f2_base = 9.8  # Modo flexional 2
    f3_base = 12.4 # Modo torsional
    
    # El daño reduce la rigidez -> reduce la frecuencia fundamental
    # Caída máxima del ~15% en el nivel 6
    f1 = f1_base * (1.0 - 0.025 * nivel_dano)
    
    # El daño introduce acoplamiento no lineal (armónicos)
    non_linear_amp = 0.05 * (nivel_dano ** 1.5)
    
    # Ruido ambiental (tráfico, viento)
    np.random.seed(42 + nivel_dano) # Para reproducibilidad en la demo
    ruido = np.random.normal(0, 0.15, n_points)
    
    # Construcción de la señal
    a_t = (
        1.0 * np.sin(2 * np.pi * f1 * t) +
        0.5 * np.sin(2 * np.pi * f2_base * t) +
        0.3 * np.sin(2 * np.pi * f3_base * t) +
        non_linear_amp * np.sin(2 * np.pi * (2 * f1) * t) + # Armónico no lineal
        ruido
    )
    
    return a_t

def takens_embedding(serie: np.ndarray, tau: int = 5, m: int = 6) -> np.ndarray:
    """
    Teorema de Embedding de Takens.
    Reconstruye el atractor topológico desde una serie 1D.
    
    Parámetros
    ----------
    serie : ndarray
        Serie temporal 1D.
    tau : int
        Retardo (time delay).
    m : int
        Dimensión de embedding.
        
    Retorna
    -------
    X_tda : ndarray
        Nube de puntos en R^m.
    """
    n = len(serie)
    N_tda = n - (m - 1) * tau
    if N_tda <= 0:
        raise ValueError("La serie es demasiado corta para estos parámetros de Takens.")
        
    X_tda = np.zeros((N_tda, m))
    for i in range(m):
        X_tda[:, i] = serie[i*tau : i*tau + N_tda]
        
    return X_tda

def calcular_diagrama_takens(X_tda: np.ndarray, max_points: int = 1500) -> np.ndarray:
    """
    Calcula el diagrama de persistencia Dgm_1 de la nube de Takens.
    Se hace un subsampling si la nube es muy grande para no saturar Ripser.
    """
    # Subsampling uniforme para eficiencia si es muy grande
    if len(X_tda) > max_points:
        idx = np.linspace(0, len(X_tda)-1, max_points, dtype=int)
        X_proc = X_tda[idx]
    else:
        X_proc = X_tda
        
    # Calcular homología usando ripser
    # Solo necesitamos H1 para el indicador de daño
    res = ripser(X_proc, maxdim=1)
    dgm1 = res['dgms'][1]
    
    return dgm1

def calcular_indicador_dano(dgm1_ref: np.ndarray, dgm1_eval: np.ndarray) -> float:
    """
    Calcula el Indicador de Daño I_D usando la distancia de Wasserstein (orden 2).
    
    I_D(t) = d_{W_2}^2(Dgm_1^{(0)}, Dgm_1^{(t)})
    
    Retorna el valor bruto (sin calibración) de d_W².
    """
    # Filtrar puntos vacíos o infinitos por seguridad
    if len(dgm1_ref) == 0: dgm1_ref = np.array([[0, 0]])
    if len(dgm1_eval) == 0: dgm1_eval = np.array([[0, 0]])
    
    # Calcular distancia de Wasserstein
    # persim devuelve d_W, el documento usa d_W^2
    d_w = persim.wasserstein(dgm1_ref, dgm1_eval, matching=False)
    
    return float(d_w ** 2)


def barrido_completo_niveles(n_points: int = 8000, fs: float = 100.0) -> dict:
    """
    Ejecuta el pipeline completo Takens → Ripser → Wasserstein para los 7 
    niveles de daño (0-6) contra la referencia (nivel 0).
    
    Retorna
    -------
    dict con:
        'niveles': lista de niveles [0..6]
        'I_D_crudos': lista de valores I_D (d_W²) sin normalizar
        'I_D_normalizados': lista de I_D normalizados a [0, 1]
        'dgm_ref': diagrama de referencia
        'dgms_eval': dict de diagramas por nivel
    """
    # Generar referencia (nivel 0)
    a_ref = generar_senal_z24(0, n_points=n_points, fs=fs)
    X_ref = takens_embedding(a_ref, tau=5, m=6)
    dgm_ref = calcular_diagrama_takens(X_ref)
    
    niveles = list(range(7))
    I_D_crudos = []
    dgms_eval = {}
    
    for nivel in niveles:
        if nivel == 0:
            I_D_crudos.append(0.0)
            dgms_eval[nivel] = dgm_ref
        else:
            a_eval = generar_senal_z24(nivel, n_points=n_points, fs=fs)
            X_eval = takens_embedding(a_eval, tau=5, m=6)
            dgm_eval = calcular_diagrama_takens(X_eval)
            dgms_eval[nivel] = dgm_eval
            I_D = calcular_indicador_dano(dgm_ref, dgm_eval)
            I_D_crudos.append(I_D)
    
    # Normalizar a [0, 1] usando el máximo computado
    max_I_D = max(I_D_crudos) if max(I_D_crudos) > 0 else 1.0
    I_D_normalizados = [v / max_I_D for v in I_D_crudos]
    
    return {
        'niveles': niveles,
        'I_D_crudos': I_D_crudos,
        'I_D_normalizados': I_D_normalizados,
        'dgm_ref': dgm_ref,
        'dgms_eval': dgms_eval
    }
