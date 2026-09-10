"""Catálogo de 7 nubes de referencia con topología conocida a priori.

Implementa el catálogo del §9.4.1 del Perfil y §5 de la metodología:
  1. círculo     β₀=1, β₁=1   (1D manifold en ℝ²)
  2. disco       β₀=1, β₁=0   (contractible en ℝ²)
  3. anillo      β₀=1, β₁=1   (contractible menos un disco interior)
  4. dos_círculos β₀=2, β₁=2  (dos componentes disjuntas)
  5. toro        β₀=1, β₁=2   (2D manifold en ℝ³)
  6. esfera      β₀=1, β₁=0   (contractible en homología H₁)
  7. gaussiana   β₀=1, β₁=0   (contractible)

El par disco/anillo está construido para tener media, covarianza y soporte
convexo casi idénticos (§9.4.1): sin ese par H.E.1b sería vacía.

Funciones adicionales:
  - add_gaussian_noise: ruido gaussiano escalado por diámetro
  - apply_random_affine: transformación afín aleatoria invertible
  - compute_diameter: distancia máxima entre pares
"""

import numpy as np
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════
# CATÁLOGO DE 7 NUBAS — Betti teóricos (H₁, k ≤ 1)
# ═══════════════════════════════════════════════════════════════════

# Parámetros geométricos del par disco/anillo (§9.4.1)
_R_DISCO = 2.0          # Radio del disco
_R_ANILLO_EXT = 2.0     # Radio exterior del anillo (= radio del disco)
_R_ANILLO_INT = 0.8     # Radio interior del anillo
_SEP_DOS_CIRCULOS = 5.0 # Separación entre centros de los dos círculos

BETTI_TEORICO = {
    "circulo":      (1, 1),
    "disco":        (1, 0),
    "anillo":       (1, 1),
    "dos_circulos": (2, 2),
    "toro":         (1, 2),
    "esfera":       (1, 0),
    "gaussiana":    (1, 0),
    # Legacy shapes (3D, used in old pipeline)
    "cubo":         (1, 0),
}

# Nombres legacy (inglés) para compatibilidad
SHAPE_ALIASES = {
    "circle":       "circulo",
    "disk":         "disco",
    "annulus":      "anillo",
    "two_circles":  "dos_circulos",
    "torus":        "toro",
    "sphere":       "esfera",
    "gaussian":     "gaussiana",
    "cube":         "cubo",
}


def _resolve_shape(shape: str) -> str:
    """Resuelve alias en inglés al nombre canónico en español."""
    return SHAPE_ALIASES.get(shape, shape)


# ═══════════════════════════════════════════════════════════════════
# GENERADORES POR FORMA
# ═══════════════════════════════════════════════════════════════════

def _generate_circulo(n: int) -> np.ndarray:
    """Círculo unitario en ℝ² (1D manifold). β₁=1."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _generate_disco(n: int) -> np.ndarray:
    """Disco unitario rellenado en ℝ². β₁=0.
    Muestreo uniforme: r = R·√U, θ ~ Uniform(0, 2π).
    Radio R = _R_DISCO para ser congruente con el anillo.
    """
    R = _R_DISCO
    r = R * np.sqrt(np.random.uniform(0, 1, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _generate_anillo(n: int) -> np.ndarray:
    """Anillo en ℝ² con radio interior _R_ANILLO_INT y exterior _R_ANILLO_EXT.
    β₁=1. Muestreo uniforme: r = √(r_min² + U·(R_max² - r_min²)).
    """
    r_min, r_max = _R_ANILLO_INT, _R_ANILLO_EXT
    r = np.sqrt(r_min**2 + np.random.uniform(0, 1, n) * (r_max**2 - r_min**2))
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _generate_dos_circulos(n: int) -> np.ndarray:
    """Dos círculos unitarios disjuntos en ℝ². β₀=2, β₁=2.
    Separados horizontalmente por _SEP_DOS_CIRCULOS.
    """
    n1 = n // 2
    n2 = n - n1
    theta1 = np.random.uniform(0, 2 * np.pi, n1)
    theta2 = np.random.uniform(0, 2 * np.pi, n2)
    sep = _SEP_DOS_CIRCULOS / 2
    c1 = np.column_stack([np.cos(theta1) - sep, np.sin(theta1)])
    c2 = np.column_stack([np.cos(theta2) + sep, np.sin(theta2)])
    return np.vstack([c1, c2])


def _generate_toro(n: int) -> np.ndarray:
    """Toro en ℝ³ con R=2, r=1. β₁=2."""
    R, r = 2.0, 1.0
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(0, 2 * np.pi, n)
    return np.column_stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi),
    ])


def _generate_esfera(n: int) -> np.ndarray:
    """Esfera unitaria en ℝ³. β₁=0 (H₁=0)."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(np.random.uniform(-1, 1, n))
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def _generate_gaussiana(n: int) -> np.ndarray:
    """Gaussiana estándar en ℝ². β₁=0 (contractible)."""
    return np.random.normal(0, 1, (n, 2))


def _generate_cubo(n: int) -> np.ndarray:
    """Cubo unitario en ℝ³ (superficie de [0,1]³). β₁=0 (legacy shape)."""
    pts = np.random.uniform(0, 1, (n, 3))
    # Poner al menos una coordenada en 0 o 1 para que estén en la superficie
    face = np.random.randint(0, 3, n)
    value = np.random.choice([0.0, 1.0], n)
    for i in range(n):
        pts[i, face[i]] = value[i]
    return pts


_GENERADORES = {
    "circulo":      _generate_circulo,
    "disco":        _generate_disco,
    "anillo":       _generate_anillo,
    "dos_circulos": _generate_dos_circulos,
    "toro":         _generate_toro,
    "esfera":       _generate_esfera,
    "gaussiana":    _generate_gaussiana,
    "cubo":         _generate_cubo,
}


# ═══════════════════════════════════════════════════════════════════
# API PÚBLICA
# ═══════════════════════════════════════════════════════════════════

def generate_cloud(shape: str, n_points: int) -> np.ndarray:
    """Genera una nube de puntos sintética de la forma especificada.

    Soporta nombres en español e inglés (aliases).

    Args:
        shape: Forma a generar. Ver SHAPE_ALIASES para los válidos.
        n_points: Número de puntos a muestrear (mínimo recomendado: 400).

    Returns:
        Arreglo de forma (n_points, d) con d ∈ {2, 3} según la forma.

    Raises:
        ValueError: Si la forma no es reconocida.
    """
    key = _resolve_shape(shape)
    if key not in _GENERADORES:
        validos = list(_GENERADORES.keys()) + list(SHAPE_ALIASES.keys())
        raise ValueError(
            f"Forma desconocida: '{shape}'. Válidos: {validos}"
        )
    return _GENERADORES[key](n_points)


def betti_teorico(shape: str) -> Tuple[int, int]:
    """Retorna (β₀, β₁) teóricos de una forma del catálogo.

    Args:
        shape: Nombre de la forma (español o inglés).

    Returns:
        Tupla (β₀, β₁) conocida a priori.
    """
    key = _resolve_shape(shape)
    if key not in BETTI_TEORICO:
        raise ValueError(f"Forma desconocida: '{shape}'")
    return BETTI_TEORICO[key]


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


def add_gaussian_noise(points: np.ndarray, q: float) -> np.ndarray:
    """Agrega ruido gaussiano relativo al diámetro: σ = q · diam(X).

    Ref: §9.4.1 — perturbación gaussiana a niveles q ∈ {0, .05, .10, .15, .20}.

    Args:
        points: Nube de puntos original de forma (N, D).
        q: Nivel relativo de ruido (fracción del diámetro).

    Returns:
        Nube de puntos perturbada de forma (N, D).
    """
    diam = compute_diameter(points)
    return points + np.random.normal(0, q * diam, points.shape)


def apply_random_affine(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica una transformación afín aleatoria invertible: Y = X·Aᵀ + b.

    Ref: §9.4.3 (Limitaciones) — la deformación afín es un homeomorfismo
    que preserva βₖ pero destruye el espectro PCA, la inercia de k-medias
    y la dispersión radial, eliminando la fuga metodológica del baseline.

    La matriz A se genera como I + ε·N(0,1) con ε = 0.3 (perturbación
    moderada que preserva la estructura topológica).

    Args:
        points: Nube de puntos de forma (N, D).

    Returns:
        Y: Nube transformada de forma (N, D).
        A: Matriz de transformación (D, D) invertible.
    """
    d = points.shape[1]
    # Generar matriz cercana a identidad (invertible con probabilidad 1)
    A = np.eye(d) + 0.3 * np.random.randn(d, d)
    # Verificar que sea invertible (determinante no nulo)
    while abs(np.linalg.det(A)) < 0.1:
        A = np.eye(d) + 0.3 * np.random.randn(d, d)
    b = 0.5 * np.random.randn(d)
    Y = points @ A.T + b
    return Y, A


def catalogo_completo() -> dict:
    """Retorna el catálogo completo con nombres, Betti teóricos y dimensiones.

    Returns:
        Dict con entrada por forma: {nombre: {betti: (b0, b1), dim: d}}
    """
    catalogo = {}
    for shape in BETTI_TEORICO:
        b0, b1 = BETTI_TEORICO[shape]
        # Generar una nube pequeña para determinar la dimensionalidad
        pts = generate_cloud(shape, 10)
        d = pts.shape[1]
        catalogo[shape] = {"betti": (b0, b1), "dim": d}
    return catalogo
