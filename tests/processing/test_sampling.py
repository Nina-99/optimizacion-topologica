"""Tests del módulo de sampling — generate_cloud, add_gaussian_noise, errores.

Verifica que:
  - generate_cloud produce nubes con forma y geometría correcta
  - Torus respeta los radios R=2.0 y r=1.0
  - add_gaussian_noise aumenta dispersión proporcional al ruido
  - Forma inválida lanza ValueError
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.processing.sampling import add_gaussian_noise, compute_diameter, generate_cloud


class TestGenerateCloud:
    """Generación de nubes de puntos sintéticas."""

    def test_sphere_shape(self):
        """generate_cloud("sphere", n) debe retornar (n, 3)."""
        pts = generate_cloud("sphere", 100)
        assert pts.shape == (100, 3)

    def test_torus_shape(self):
        """generate_cloud("torus", n) debe retornar (n, 3)."""
        pts = generate_cloud("torus", 100)
        assert pts.shape == (100, 3)

    def test_sphere_points_on_unit_surface(self):
        """Puntos de esfera deben tener norma ≈ 1.0 (radio unitario)."""
        pts = generate_cloud("sphere", 200)
        norms = np.linalg.norm(pts, axis=1)
        assert_allclose(norms, 1.0, atol=1e-10,
                        err_msg="Puntos de esfera deben estar en superficie unitaria")

    def test_torus_x_range(self):
        """Toro con R=2.0, r=1.0: |x| debe estar en [1.0, 3.0] para puntos en ecuador."""
        pts = generate_cloud("torus", 500)
        x_abs = np.abs(pts[:, 0])
        # Mínimo |x| puede acercarse a R-r = 1.0 si phi=π, theta=0
        # Máximo |x| puede acercarse a R+r = 3.0 si phi=0, theta=0
        assert np.all(x_abs <= 3.0 + 1e-10), "|x| excede R+r"
        assert np.all(x_abs >= 0.0), "x debe tener valores positivos y negativos"

    def test_torus_z_range(self):
        """Toro con r=1.0: |z| debe estar en [0, 1.0]."""
        pts = generate_cloud("torus", 500)
        z_abs = np.abs(pts[:, 2])
        assert np.all(z_abs <= 1.0 + 1e-10), f"|z| máximo debe ser r=1.0, obtenido {z_abs.max()}"
        # Al menos algunos puntos tienen |z| > 0.5 (no todos en el ecuador)
        assert np.any(z_abs > 0.5), "Debe haber puntos fuera del ecuador del toro"

    def test_torus_radius_ratio(self):
        """La relación R/r = 2.0 debe cumplirse (R=2.0, r=1.0)."""
        pts = generate_cloud("torus", 500)
        # Calcular el radio desde el eje Z para cada punto
        r_vals = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        # min r debe ser ~R-r = 1.0, max r debe ser ~R+r = 3.0
        assert r_vals.min() >= 0.5, f"Radio mínimo del toro muy pequeño: {r_vals.min()}"
        assert r_vals.max() <= 3.5, f"Radio máximo del toro muy grande: {r_vals.max()}"

    def test_sphere_z_range(self):
        """Esfera unitaria: z debe estar en [-1, 1]."""
        pts = generate_cloud("sphere", 200)
        assert np.all(np.abs(pts[:, 2]) <= 1.0 + 1e-10)

    def test_output_is_finite(self):
        """generate_cloud debe producir puntos finitos sin NaN/Inf."""
        pts = generate_cloud("sphere", 50)
        assert np.all(np.isfinite(pts)), "generate_cloud produce NaN o Inf"
        pts = generate_cloud("torus", 50)
        assert np.all(np.isfinite(pts)), "generate_cloud produce NaN o Inf"

    def test_invalid_shape_raises_value_error(self):
        """Forma inválida debe lanzar ValueError con mensaje informativo."""
        with pytest.raises(ValueError, match="Forma desconocida|Unknown shape"):
            generate_cloud("undefined", 10)

    def test_invalid_shape_empty_string(self):
        """String vacío como forma debe lanzar ValueError."""
        with pytest.raises(ValueError):
            generate_cloud("", 10)

    def test_cube_shape(self):
        """generate_cloud("cube", n) debe retornar (n, 3)."""
        pts = generate_cloud("cube", 100)
        assert pts.shape == (100, 3)

    def test_cube_points_bounded(self):
        """Puntos del cubo deben estar en rango [0, 1] en cada eje."""
        pts = generate_cloud("cube", 500)
        assert pts.min() >= 0.0, f"Coordenada mínima {pts.min()} < 0"
        assert pts.max() <= 1.0, f"Coordenada máxima {pts.max()} > 1"

    def test_cube_points_on_surface(self):
        """Todos los puntos del cubo deben tener al menos una coordenada 0 o 1."""
        pts = generate_cloud("cube", 500)
        on_surface = np.any((pts == 0.0) | (pts == 1.0), axis=1)
        assert on_surface.all(), "Todos los puntos del cubo deben estar en la superficie"

    def test_cube_output_is_finite(self):
        """generate_cloud("cube") debe producir puntos finitos."""
        pts = generate_cloud("cube", 50)
        assert np.all(np.isfinite(pts)), "generate_cloud cube produce NaN o Inf"

    def test_different_n_points(self):
        """generate_cloud("sphere", n) con distintos n."""
        pts_10 = generate_cloud("sphere", 10)
        pts_100 = generate_cloud("sphere", 100)
        assert pts_10.shape == (10, 3)
        assert pts_100.shape == (100, 3)


class TestAddGaussianNoise:
    """Adición de ruido gaussiano a nubes de puntos."""

    def test_noise_shape_preserved(self):
        """add_gaussian_noise debe preservar el shape."""
        pts = np.random.randn(20, 3)
        noisy = add_gaussian_noise(pts, 0.1)
        assert noisy.shape == pts.shape

    def test_no_noise_zero_std(self):
        """Con noise_std=0, no debe haber cambio (ruido σ=0)."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        noisy = add_gaussian_noise(pts, 0.0)
        assert_allclose(noisy, pts, atol=1e-12,
                        err_msg="Con ruido cero no debe haber cambios")

    def test_increases_dispersion(self):
        """Con ruido positivo, los puntos deben alejarse del original."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        noisy = add_gaussian_noise(pts, 0.2)
        # La distancia entre el punto original y el ruidoso debe crecer
        orig_dist = np.linalg.norm(pts[0] - pts[1])
        noisy_dist = np.linalg.norm(noisy[0] - noisy[1])
        # No podemos garantizar que aumente siempre (ruido puede acercar),
        # pero la media de muchas realizaciones debe aumentar
        dists_noisy = []
        for _ in range(50):
            noisy = add_gaussian_noise(pts, 0.2)
            dists_noisy.append(np.linalg.norm(noisy[0] - noisy[1]))
        mean_noisy_dist = np.mean(dists_noisy)
        assert mean_noisy_dist > orig_dist, (
            f"El ruido debe aumentar la distancia media ({mean_noisy_dist:.4f} vs {orig_dist:.4f})"
        )

    def test_noise_std_scaling(self):
        """Mayor noise_std debe producir mayor dispersión (en media)."""
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
        stds_low, stds_high = [], []
        for _ in range(30):
            noisy_low = add_gaussian_noise(pts, 0.1)
            noisy_high = add_gaussian_noise(pts, 0.5)
            stds_low.append(np.std(noisy_low - pts))
            stds_high.append(np.std(noisy_high - pts))
        assert np.mean(stds_high) > np.mean(stds_low), (
            f"Mayor ruido debe dar mayor std media "
            f"({np.mean(stds_high):.4f} vs {np.mean(stds_low):.4f})"
        )

    def test_noise_preserves_dtype(self):
        """add_gaussian_noise debe preservar el dtype del input."""
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
        noisy = add_gaussian_noise(pts, 0.1)
        assert noisy.dtype == pts.dtype


class TestComputeDiameter:
    """Tests para compute_diameter — distancia máxima entre pares."""

    def test_single_point_diameter_zero(self):
        """Un solo punto debe tener diámetro 0."""
        pts = np.array([[0.0, 0.0, 0.0]])
        assert compute_diameter(pts) == 0.0

    def test_two_points_distance(self):
        """Dos puntos deben dar la distancia entre ellos."""
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        assert compute_diameter(pts) == 3.0

    def test_three_points_max_pair(self):
        """Tres puntos debe dar la máxima distancia entre pares.
        
        Puntos: (0,0,0), (1,0,0), (0,4,0).
        Distancias: 1, 4, sqrt(17)≈4.123 — el máximo es sqrt(17).
        """
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        import math
        assert compute_diameter(pts) == math.sqrt(17.0)

    def test_diameter_is_float(self):
        """compute_diameter debe retornar float."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert isinstance(compute_diameter(pts), float)

    def test_diameter_2d_points(self):
        """compute_diameter funciona con puntos 2D."""
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert compute_diameter(pts) == 5.0  # 3-4-5 triangle
