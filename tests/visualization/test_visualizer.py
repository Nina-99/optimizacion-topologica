"""Tests del módulo de visualización — visualizer.py.

Verifica que las funciones de visualización producen los tipos de retorno
esperados sin necesidad de backend interactivo (Agg).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tda.visualization.visualizer import (
    reduce_dimensions,
    visualize_point_cloud_2d,
    visualize_point_cloud_3d,
)


class TestVisualize2D:
    """Nube de puntos 2D con matplotlib."""

    def test_returns_fig_ax(self, points_2d):
        """visualize_point_cloud_2d debe retornar (Figure, Axes)."""
        fig, ax = visualize_point_cloud_2d(points_2d, title="Test 2D")
        assert fig is not None
        assert ax is not None

    def test_default_title_when_none(self, points_2d):
        """Sin título explícito debe usar valor por defecto."""
        fig, ax = visualize_point_cloud_2d(points_2d)
        assert ax.get_title() == "Point Cloud (2D)"

    def test_custom_title_displayed(self, points_2d):
        """Título personalizado debe aparecer en el gráfico."""
        title = "Custom Title Test"
        fig, ax = visualize_point_cloud_2d(points_2d, title=title)
        assert ax.get_title() == title

    def test_scatter_data_matches_input(self, points_2d):
        """Los puntos graficados deben coincidir con los datos de entrada."""
        fig, ax = visualize_point_cloud_2d(points_2d)
        collections = ax.collections
        assert len(collections) >= 1
        offsets = collections[0].get_offsets()
        assert_allclose(offsets, points_2d, atol=1e-12)

    def test_ax_equal_aspect(self, points_2d):
        """El aspecto debe ser 'equal' (o 1.0 con Agg) para no distorsionar formas.

        Nota: matplotlib con backend Agg puede retornar 1.0 en vez del string 'equal'.
        """
        fig, ax = visualize_point_cloud_2d(points_2d)
        aspect = ax.get_aspect()
        assert aspect in ('equal', 1.0), f"Aspecto inesperado: {aspect}"

    def test_accepts_existing_axes(self, points_2d):
        """Debe aceptar ejes existentes y retornarlos."""
        import matplotlib.pyplot as plt
        _, existing_ax = plt.subplots()
        fig, ax = visualize_point_cloud_2d(points_2d, ax=existing_ax)
        assert ax is existing_ax


class TestVisualize3D:
    """Nube de puntos 3D — error cuando dim < 3."""

    def test_raises_value_error_for_2d_input(self, points_2d):
        """Puntos 2D deben lanzar ValueError."""
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            visualize_point_cloud_3d(points_2d)

    def test_raises_value_error_for_1d_input(self):
        """Puntos 1D deben lanzar ValueError."""
        pts = np.random.rand(10, 1)
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            visualize_point_cloud_3d(pts)

    def test_returns_figure_for_3d_input(self, points_3d):
        """Puntos 3D deben retornar una Figure."""
        fig = visualize_point_cloud_3d(points_3d)
        assert fig is not None

    def test_has_3d_projection(self, points_3d):
        """La figura 3D debe tener una proyección 3D configurada."""
        fig = visualize_point_cloud_3d(points_3d)
        ax = fig.axes[0]
        assert hasattr(ax, 'get_proj'), "El axes debe ser 3D (tener get_proj)"


class TestReduceDimensions:
    """Reducción de dimensionalidad vía PCA."""

    def test_reduces_to_2d(self):
        """PCA debe reducir array (N, 5) a (N, 2)."""
        pts = np.random.rand(20, 5)
        reduced = reduce_dimensions(pts, n_components=2)
        assert reduced.shape == (20, 2)

    def test_reduces_to_3d(self):
        """PCA debe reducir array (N, 10) a (N, 3)."""
        pts = np.random.rand(15, 10)
        reduced = reduce_dimensions(pts, n_components=3)
        assert reduced.shape == (15, 3)

    def test_reduced_values_are_finite(self):
        """Valores reducidos no deben contener NaN o Inf."""
        pts = np.random.rand(20, 5)
        reduced = reduce_dimensions(pts, n_components=2)
        assert np.all(np.isfinite(reduced))

    def test_preserves_row_count(self):
        """Número de observaciones debe preservarse."""
        pts = np.random.rand(30, 8)
        reduced = reduce_dimensions(pts, n_components=2)
        assert reduced.shape[0] == 30

    def test_different_n_components(self, points_3d):
        """Con N=3 y n_components=1 debe retornar (N, 1)."""
        reduced = reduce_dimensions(points_3d, n_components=1)
        assert reduced.shape == (len(points_3d), 1)

    def test_no_op_when_sklearn_missing(self, monkeypatch):
        """Sin sklearn debe truncar a n_components columnas."""
        import tda.visualization.visualizer as vis
        monkeypatch.setattr(vis, "PCA_KMEANS_AVAILABLE", False)
        pts = np.random.rand(10, 5)
        reduced = reduce_dimensions(pts, n_components=2)
        assert reduced.shape == (10, 2)
        assert_allclose(reduced, pts[:, :2])
