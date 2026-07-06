"""Funciones de muestreo sintético para generar nubes de puntos de prueba.

Proporciona generación de formas geométricas (esfera, toro) y
adiçón de ruido gaussiano controlado.
"""

import numpy as np


def generate_cloud(shape: str, n_points: int) -> np.ndarray:
    """Genera una nube de puntos 3D sintética para una esfera, toro o cubo.

    Args:
        shape: Forma a generar. "sphere", "torus" o "cube".
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
    elif shape == "cube":
        faces = np.random.randint(0, 6, n_points)
        uv = np.random.uniform(0, 1, (n_points, 2))
        points = np.empty((n_points, 3))
        for i in range(6):
            mask = faces == i
            n_face = mask.sum()
            if n_face == 0:
                continue
            if i == 0:
                points[mask] = np.column_stack([np.zeros(n_face), uv[mask, 0], uv[mask, 1]])
            elif i == 1:
                points[mask] = np.column_stack([np.ones(n_face), uv[mask, 0], uv[mask, 1]])
            elif i == 2:
                points[mask] = np.column_stack([uv[mask, 0], np.zeros(n_face), uv[mask, 1]])
            elif i == 3:
                points[mask] = np.column_stack([uv[mask, 0], np.ones(n_face), uv[mask, 1]])
            elif i == 4:
                points[mask] = np.column_stack([uv[mask, 0], uv[mask, 1], np.zeros(n_face)])
            else:
                points[mask] = np.column_stack([uv[mask, 0], uv[mask, 1], np.ones(n_face)])
        return points
    else:
        raise ValueError(f"Forma desconocida: {shape}. Use 'sphere', 'torus' o 'cube'.")


def compute_diameter(points: np.ndarray) -> float:
    """Calcula la distancia máxima entre pares en una nube de puntos.

    Args:
        points: Coordenadas de la nube de puntos de forma (N, D).

    Returns:
        Distancia máxima entre pares (diámetro). 0.0 si hay 0 o 1 puntos.
    """
    if points.shape[0] < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    return float(pdist(points).max())


def add_gaussian_noise(points: np.ndarray, noise_std: float) -> np.ndarray:
    """Agrega ruido gaussiano a una nube de puntos relativo a su diámetro.

    Args:
        points: Nube de puntos original de forma (N, D).
        noise_std: Factor de desviación estándar del ruido (fracción del diámetro).

    Returns:
        Nube de puntos con ruido de forma (N, D).
    """
    diam = compute_diameter(points)
    return points + np.random.normal(0, noise_std * diam, points.shape)
