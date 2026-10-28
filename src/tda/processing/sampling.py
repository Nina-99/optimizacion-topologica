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
  - add_gaussian_noise: ruido gaussiano σ = q·diam(X)/√d (normalizado por la
    dimensión ambiente; ver Item 29 de docs/Correcciones_Documento_Principal.md)
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

    Parámetros:
        shape: Forma a generar. Ver SHAPE_ALIASES para los válidos.
        n_points: Número de puntos a muestrear (mínimo recomendado: 400).

    Retorna:
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

    Parámetros:
        shape: Nombre de la forma (español o inglés).

    Retorna:
        Tupla (β₀, β₁) conocida a priori.
    """
    key = _resolve_shape(shape)
    if key not in BETTI_TEORICO:
        raise ValueError(f"Forma desconocida: '{shape}'")
    return BETTI_TEORICO[key]


def compute_diameter(points: np.ndarray) -> float:
    """Calcula la distancia máxima entre pares en una nube de puntos.

    Parámetros:
        points: Coordenadas de la nube de puntos de forma (N, D).

    Retorna:
        Distancia máxima entre pares (diámetro). 0.0 si hay 0 o 1 puntos.
    """
    if points.shape[0] < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    return float(pdist(points).max())


def add_gaussian_noise(points: np.ndarray, q: float) -> np.ndarray:
    """Agrega ruido gaussiano normalizado por la dimensión: σ = q · diam(X) / √d.

    Ref: §9.4.1 — perturbación gaussiana a niveles q ∈ {0, .05, .10, .15, .20}.
    Corregido según el Cap. V («La figura original no normalizaba el ruido por la
    dimensión»): sin el factor 1/√d, el desplazamiento medio de cada punto crece
    con d y en ℝ¹⁰⁰ la nube se destruye aunque q sea chico.

    Parámetros:
        points: Nube de puntos original de forma (N, d).
        q: Nivel relativo de ruido (fracción del diámetro, normalizado por √d).

    Retorna:
        Nube de puntos perturbada de forma (N, d).
    """
    diam = compute_diameter(points)
    sigma = q * diam / np.sqrt(points.shape[1])
    return points + np.random.normal(0, sigma, points.shape)


def apply_random_affine(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica una transformación afín aleatoria invertible: Y = X·Aᵀ + b.

    Ref: §9.4.3 (Limitaciones) — la deformación afín es un homeomorfismo
    que preserva βₖ pero destruye el espectro PCA, la inercia de k-medias
    y la dispersión radial, eliminando la fuga metodológica del baseline.

    La matriz A se genera como I + ε·N(0,1) con ε = 0.3 (perturbación
    moderada que preserva la estructura topológica).

    Parámetros:
        points: Nube de puntos de forma (N, D).

    Retorna:
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

    Retorna:
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


# ═══════════════════════════════════════════════════════════════════
# CATÁLOGO CORREGIDO (Cap. V — protocolo figcap5)
# ═══════════════════════════════════════════════════════════════════
#
# Geometría y RNG exactos del catálogo corregido del Cap. V. Difiere de
# `generate_cloud` en tres puntos que son los que hacen reproducible la
# Figura 5.1:
#   1. RNG por nube: `np.random.default_rng(semilla)` en lugar del estado
#      global `np.random` (una nube no puede alterar a la siguiente).
#   2. Geometría: anillo r ∈ [0,75; 1], dos círculos radio 0,6 separados en
#      ±1,5, toro R=2 r=0,8 — los diámetros son los que fijan τ·diam.
#   3. Gaussiana truncada en 2σ: sin truncar, los atípicos aislados producen
#      β₀ = 2 o 3 a toda escala y el catálogo deja de ser patrón de oro.
#
# ORDEN_CATALOGO fija el orden de filas de los paneles (b) y (c) de la
# Figura 5.1 (círculo, anillo, dos círculos, toro, disco, esfera, gaussiana).

CATALOGO_CORREGIDO = {
    "circulo":      (1, 1),
    "anillo":       (1, 1),
    "dos_circulos": (2, 2),
    "toro":         (1, 2),
    "disco":        (1, 0),
    "esfera":       (1, 0),
    "gaussiana":    (1, 0),   # truncada en 2 sigma
}

ORDEN_CALIBRACION = list(CATALOGO_CORREGIDO)

ETIQUETA_NUBE = {
    "circulo": "círculo",
    "anillo": "anillo",
    "dos_circulos": "dos círculos",
    "toro": "toro",
    "disco": "disco",
    "esfera": "esfera",
    "gaussiana": "gaussiana",
}


def generar_corregida(shape: str, n: int, rng) -> np.ndarray:
    """Genera la nube `shape` con la geometría del catálogo corregido.

    `rng` es un generador sembrado (p. ej. `np.random.default_rng(900 + i)`),
    NO el estado global: así el catálogo C₀ es independiente del catálogo de
    medición C.

    Raises:
        ValueError: si la forma no pertenece a CATALOGO_CORREGIDO.
    """
    if shape not in CATALOGO_CORREGIDO:
        raise ValueError(f"Forma desconocida en el catálogo corregido: '{shape}'")
    if shape == "circulo":
        t = rng.uniform(0, 2 * np.pi, n)
        return np.c_[np.cos(t), np.sin(t)]
    if shape == "anillo":
        t = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(rng.uniform(0.75 ** 2, 1.0, n))
        return np.c_[r * np.cos(t), r * np.sin(t)]
    if shape == "dos_circulos":
        t = rng.uniform(0, 2 * np.pi, n)
        c = np.where(rng.random(n) < 0.5, -1.5, 1.5)
        return np.c_[c + 0.6 * np.cos(t), 0.6 * np.sin(t)]
    if shape == "toro":
        R, r = 2.0, 0.8
        u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        return np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]
    if shape == "disco":
        t = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(rng.uniform(0, 1, n))
        return np.c_[r * np.cos(t), r * np.sin(t)]
    if shape == "esfera":
        z = rng.normal(size=(n, 3))
        return z / np.linalg.norm(z, axis=1, keepdims=True)
    if shape == "gaussiana":
        # Truncada en 2 sigma (soporte compacto): sin truncar, los atípicos
        # aislados vuelven ambiguo β₀ a la escala τ·diam.
        z = np.empty((0, 2))
        while len(z) < n:
            w = rng.normal(size=(2 * n, 2))
            z = np.r_[z, w[np.linalg.norm(w, axis=1) <= 2.0]]
        return z[:n] * 0.5
    raise ValueError(shape)   # pragma: no cover


def incrustar(X: np.ndarray, d: int) -> np.ndarray:
    """Incrusta `X` en ℝᵈ completando con ceros (d ≥ dimensión nativa).

    Usado por el protocolo corregido para evaluar la lectura en dimensiones
    ambiente d ∈ {nativa, 10, 100} (Fig. 5.2).
    """
    if X.shape[1] >= d:
        return X
    return np.c_[X, np.zeros((X.shape[0], d - X.shape[1]))]
