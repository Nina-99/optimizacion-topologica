"""Análisis de estabilidad topológica frente a ruido.

Implementa el barrido sistemático de niveles de ruido para verificar
la hipótesis H.E.1: los números de Betti permanecen estables bajo
perturbaciones del 15-20% en los datos de entrada.
"""

from typing import List, Tuple
import numpy as np
from ripser import ripser
from sklearn.cluster import KMeans

from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.analysis.metrics import compute_kmeans_accuracy


def compute_noise_sweep(
    n_points: int,
    n_clusters: int,
    noise_min: float = 0.0,
    noise_max: float = 0.30,
    n_steps: int = 10,
    random_seed: int = 42,
    progress_callback = None,
) -> dict:
    """Ejecuta un barrido sistemático de ruido y evalúa TDA vs K-Means.

    Para cada nivel de ruido genera nubes de puntos (esfera + toro),
    calcula los números de Betti vía homología persistente (Ripser)
    y el accuracy de clasificación K-Means.

    Args:
        n_points: Puntos por forma geométrica.
        n_clusters: Número de clústeres para K-Means.
        noise_min: Nivel de ruido mínimo (default 0.0).
        noise_max: Nivel de ruido máximo (default 0.30).
        n_steps: Cantidad de pasos en el barrido (default 10).
        random_seed: Semilla para reproducibilidad (default 42).
        progress_callback: Función opcional para reportar progreso.
            Se llama con (paso_actual, pasos_totales) después de cada nivel.

    Returns:
        Dict con:
            noise_vals (np.ndarray): Niveles de ruido evaluados.
            acc (List[float]): Accuracy K-Means por nivel.
            betti_s (List[Tuple[int,int]]): (β₀, β₁) esfera.
            betti_t (List[Tuple[int,int]]): (β₀, β₁) toro.
            diagrams_s (List[np.ndarray]): Diagramas de persistencia de esfera.
            diagrams_t (List[np.ndarray]): Diagramas de persistencia de toro.
    """
    np.random.seed(random_seed)
    noise_vals = np.linspace(noise_min, noise_max, n_steps)
    
    sweep_acc: List[float] = []
    sweep_betti_s: List[Tuple[int, int]] = []
    sweep_betti_t: List[Tuple[int, int]] = []
    sweep_dgms_s: List = []
    sweep_dgms_t: List = []
    
    for i, noise in enumerate(noise_vals):
        # Generar nubes limpias
        pts_s = generate_cloud("sphere", n_points)
        pts_t = generate_cloud("torus", n_points)
        pts_t[:, 0] += 1.5  # separación para evitar superposición total
        
        # Añadir ruido
        pts_s_n = add_gaussian_noise(pts_s, noise)
        pts_t_n = add_gaussian_noise(pts_t, noise)
        
        # K-Means sobre conjunto combinado
        dataset_n = np.vstack([pts_s_n, pts_t_n])
        km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        y_pred = km.fit_predict(dataset_n)
        
        y_true = np.array([0] * n_points + [1] * n_points)
        sweep_acc.append(compute_kmeans_accuracy(y_true, y_pred))
        
        # TDA: homología persistente
        res_s = ripser(pts_s_n, maxdim=1)['dgms']
        res_t = ripser(pts_t_n, maxdim=1)['dgms']
        sweep_betti_s.append((len(res_s[0]), len(res_s[1])))
        sweep_betti_t.append((len(res_t[0]), len(res_t[1])))
        sweep_dgms_s.append(res_s)
        sweep_dgms_t.append(res_t)
        
        if progress_callback is not None:
            progress_callback(i + 1, n_steps)
    
    return {
        "noise_vals": noise_vals,
        "acc": sweep_acc,
        "betti_s": sweep_betti_s,
        "betti_t": sweep_betti_t,
        "diagrams_s": sweep_dgms_s,
        "diagrams_t": sweep_dgms_t,
    }
