"""Funciones de muestreo sintético para generar nubes de puntos de prueba.

Proporciona generación de formas geométricas (esfera, toro) y
adiçón de ruido gaussiano controlado.
"""

import numpy as np


def generate_cloud(shape: str, n_points: int) -> np.ndarray:
    """Genera una nube de puntos 3D sintética para una esfera o toro.

    Args:
        shape: Forma a generar. "sphere" o "torus".
        n_points: Número de puntos a muestrear.

    Returns:
        Arreglo de forma (n_points, 3) representando la nube de puntos.
    """
    if shape == "sphere":
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.arccos(np.random.uniform(-1, 1, n_points))
        return np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
    elif shape == "torus":
        R, r = 2.0, 1.0
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        return np.column_stack([
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi)
        ])
    else:
        raise ValueError(f"Forma desconocida: {shape}. Use 'sphere' o 'torus'.")


def add_gaussian_noise(points: np.ndarray, noise_std: float) -> np.ndarray:
    """Agrega ruido gaussiano a una nube de puntos relativo a su diámetro.

    Args:
        points: Nube de puntos original de forma (N, D).
        noise_std: Factor de desviación estándar del ruido (fracción del diámetro).

    Returns:
        Nube de puntos con ruido de forma (N, D).
    """
    from scipy.spatial.distance import pdist
    diam = float(pdist(points).max())
    return points + np.random.normal(0, noise_std * diam, points.shape)
