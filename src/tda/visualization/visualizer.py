import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Check for optional dependencies
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    PCA_KMEANS_AVAILABLE = True
except ImportError:
    PCA_KMEANS_AVAILABLE = False

try:
    import persim
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False

# Import TDA pipeline and metrics functions
from tda.simulation.pipeline import generate_cloud, add_gaussian_noise, compute_persistence_diagram
from tda.core.topology import betti_numbers

# Create data directory if needed
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def _format_diagrams_for_betti(dgms):
    """Converts the list of ripser diagrams to (N, 3) format expected by betti_numbers.

    Prevents logical errors if some homology dimension (H0 or H1) is empty.

    Args:
        dgms (list of numpy.ndarray): A list of persistence diagrams for each dimension.

    Returns:
        numpy.ndarray: Combined diagram array with birth, death, and dimension columns.

    Examples:
        >>> import numpy as np
        >>> dgms = [np.array([[0.0, 1.0]]), np.array([[1.0, 2.0]])]
        >>> _format_diagrams_for_betti(dgms)
        array([[0., 1., 0.],
               [1., 2., 1.]])
    """
    formatted_dgms = []
    for dim, dgm in enumerate(dgms):
        dgm_arr = np.array(dgm)
        if dgm_arr.size > 0 and dgm_arr.ndim == 2:
            dim_col = np.full((dgm_arr.shape[0], 1), dim)
            dgm_with_dim = np.hstack([dgm_arr, dim_col])
            formatted_dgms.append(dgm_with_dim)
            
    if formatted_dgms:
        return np.vstack(formatted_dgms)
    return np.empty((0, 3))


# ==========================================
# PRUEBAS DE LA HIPÓTESIS H.E.1
# ==========================================
def plot_stability_analysis(shape='torus', noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25], n_rep=100, n_points=250):
    """Runs statistical stability analysis of Betti numbers over multiple repetitions.

    Args:
        shape (str, optional): The shape type ('torus', 'sphere', etc.). Defaults to 'torus'.
        noise_levels (list of float, optional): Noise levels as a fraction of diameter.
            Defaults to [0.0, 0.05, 0.10, 0.15, 0.20, 0.25].
        n_rep (int, optional): Number of repetitions per noise level. Defaults to 100.
        n_points (int, optional): Number of points per point cloud. Defaults to 250.

    Returns:
        matplotlib.figure.Figure: Generated matplotlib figure.

    Examples:
        >>> fig = plot_stability_analysis(shape='sphere', n_rep=2, n_points=50)
    """
    mean_betas = []
    std_betas = []
    
    print(f"Iniciando análisis de estabilidad para {shape} ({n_rep} realizaciones por nivel)...")
    
    for noise in noise_levels:
        betas = []
        for _ in range(n_rep):
            points = generate_cloud(shape, n_points)
            noisy_points = add_gaussian_noise(points, noise)
            dgms = compute_persistence_diagram(noisy_points)
            
            full_dgm_array = _format_diagrams_for_betti(dgms)
            
            if full_dgm_array.shape[0] > 0:
                _, b1 = betti_numbers(full_dgm_array)
            else:
                b1 = 0
            
            betas.append(b1)
            
        mean_betas.append(np.mean(betas))
        std_betas.append(np.std(betas))
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(noise_levels, mean_betas, yerr=std_betas, fmt='-o', capsize=5, 
                color='royalblue', ecolor='gray', label=r'$\beta_1$ promedio')
    
    ax.set_title(rf"Prueba Empírica H.E.1: Robustez de $\beta_1$ ante ruido\nForma: {shape.capitalize()} | {n_rep} realizaciones")
    ax.set_xlabel("Nivel de Ruido (Factor del diámetro)")
    ax.set_ylabel(r"Valor promedio de $\beta_1$")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Guardar
    save_path = DATA_DIR / f"robustez_HE1_{shape}.png"
    fig.savefig(save_path, dpi=300)
    print(f"Gráfico de robustez guardado en: {save_path}")
    
    return fig


def plot_tda_vs_kmeans_pca(n_points=250, noise_level=0.15):
    """Compares PCA + K-Means (Euclidean metrics) vs TDA (topological invariants).

    Args:
        n_points (int, optional): Number of points per shape. Defaults to 250.
        noise_level (float, optional): Noise standard deviation ratio. Defaults to 0.15.

    Returns:
        matplotlib.figure.Figure or None: Generated comparison figure, or None if
            scikit-learn is not available.

    Examples:
        >>> fig = plot_tda_vs_kmeans_pca(n_points=50, noise_level=0.1)
    """
    if not PCA_KMEANS_AVAILABLE:
        print("Error: scikit-learn (KMeans, PCA) no está disponible.")
        return None

    print(f"Iniciando comparación TDA vs Euclidiano (Ruido {noise_level*100}%)...")

    # 1. Generación de datos superpuestos
    pts_sphere = generate_cloud("sphere", n_points)
    pts_torus = generate_cloud("torus", n_points)
    pts_torus[:, 0] += 1.5  # Superposición intencional en el eje X
    
    dataset = np.vstack([pts_sphere, pts_torus])
    dataset_noisy = add_gaussian_noise(dataset, noise_level)
    y_true = np.array([0]*n_points + [1]*n_points)
    
    # 2. Métodos Euclidianos
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(dataset_noisy)
    
    pca = PCA(n_components=2)
    dataset_pca = pca.fit_transform(dataset_noisy)
    
    # 3. Análisis Topológico (TDA) de cada entidad en presencia de ruido
    sphere_noisy = add_gaussian_noise(pts_sphere, noise_level)
    torus_noisy = add_gaussian_noise(pts_torus, noise_level)
    
    dgms_sphere = compute_persistence_diagram(sphere_noisy)
    dgms_torus = compute_persistence_diagram(torus_noisy)
    
    arr_sphere = _format_diagrams_for_betti(dgms_sphere)
    arr_torus = _format_diagrams_for_betti(dgms_torus)
    
    b0_s, b1_s = betti_numbers(arr_sphere) if arr_sphere.shape[0] > 0 else (0, 0)
    b0_t, b1_t = betti_numbers(arr_torus) if arr_torus.shape[0] > 0 else (0, 0)

    # 4. Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(dataset_pca[:, 0], dataset_pca[:, 1], c=y_true, cmap='plasma', alpha=0.6, s=20)
    axes[0].set_title("Verdad Terreno (Ground Truth)\nEstructuras Reales en Proyección PCA")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(dataset_pca[:, 0], dataset_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6, s=20)
    axes[1].set_title("Clasificación Euclidiana (K-Medias)\nFalla al separar geometrías superpuestas")
    axes[1].set_xlabel("PCA 1")
    axes[1].grid(True, alpha=0.3)
    
    tda_text = (rf"Resultados TDA (Robustez Invariante):\n"
                rf"Esfera: $\beta_0={b0_s}$, $\beta_1={b1_s}$ (Esperado: 1, 0)\n"
                rf"Toro: $\beta_0={b0_t}$, $\beta_1={b1_t}$ (Esperado: 1, 2)")
    
    fig.suptitle("Comparación: Descriptores Euclidianos vs. Estructura Topológica (H.E.1)", fontsize=14)
    fig.text(0.5, 0.02, tda_text, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    save_path = DATA_DIR / "comparacion_TDA_vs_KMedias.png"
    fig.savefig(save_path, dpi=300)
    print(f"Gráfico comparativo guardado en: {save_path}")
    
    return fig


# ==========================================
# FUNCIONES BÁSICAS DE VISUALIZACIÓN
# ==========================================
def visualize_point_cloud_2d(points, title=None, ax=None, backend='matplotlib'):
    """Plots a 2D point cloud using Matplotlib or Plotly.

    Args:
        points (numpy.ndarray): 2D points array of shape (N, 2).
        title (str, optional): Title of the plot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Pre-existing axes to plot on. Defaults to None.
        backend (str, optional): Plotting library choice ('matplotlib' or 'plotly').
            Defaults to 'matplotlib'.

    Returns:
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or plotly.graph_objects.Figure:
            The figure and axes (Matplotlib) or the figure object (Plotly).

    Examples:
        >>> import numpy as np
        >>> pts = np.random.rand(10, 2)
        >>> fig, ax = visualize_point_cloud_2d(pts, title="Test")
    """
    if backend == 'matplotlib':
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title or 'Point Cloud (2D)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        return fig, ax
    elif backend == 'plotly':
        fig = go.Figure(data=go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers'))
        return fig

def visualize_point_cloud_3d(points, title=None, backend='matplotlib'):
    """Plots a 3D point cloud using Matplotlib or Plotly.

    Args:
        points (numpy.ndarray): 3D points array of shape (N, 3) or higher.
        title (str, optional): Plot title. Defaults to None.
        backend (str, optional): Plotting library choice ('matplotlib' or 'plotly').
            Defaults to 'matplotlib'.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The figure object.

    Raises:
        ValueError: If the input point cloud has fewer than 3 dimensions.

    Examples:
        >>> import numpy as np
        >>> pts = np.random.rand(10, 3)
        >>> fig = visualize_point_cloud_3d(pts, title="Test 3D")
    """
    if points.shape[1] < 3:
        raise ValueError("Point cloud must have at least 3 dimensions.")
    if backend == 'matplotlib':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.6, s=10)
        return fig
    elif backend == 'plotly':
        fig = go.Figure(data=go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers'))
        return fig

def reduce_dimensions(points, n_components=2):
    """Reduces dimensional space of a point cloud using PCA.

    Args:
        points (numpy.ndarray): Multi-dimensional points array of shape (N, D).
        n_components (int, optional): Target dimensions. Defaults to 2.

    Returns:
        numpy.ndarray: Projected low-dimensional point cloud coordinates of shape (N, n_components).

    Examples:
        >>> import numpy as np
        >>> pts = np.random.rand(10, 5)
        >>> reduced = reduce_dimensions(pts, n_components=2)
    """
    if not PCA_KMEANS_AVAILABLE:
        return points[:, :n_components]
    pca = PCA(n_components=n_components)
    return pca.fit_transform(points)

def visualize_persistence_diagram(points, title=None, backend='matplotlib'):
    """Computes and plots persistence diagram for dimension 1 (H1).

    Args:
        points (numpy.ndarray): Point cloud coordinates of shape (N, D).
        title (str, optional): Title of the diagram. Defaults to None.
        backend (str, optional): Plotting library choice ('matplotlib' or 'plotly').
            Defaults to 'matplotlib'.

    Returns:
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None: The figure
            and axes (Matplotlib) or None (Plotly/unsupported).

    Raises:
        ValueError: If the persim library is not available.

    Examples:
        >>> import numpy as np
        >>> pts = np.random.rand(50, 3)
        >>> try:
        ...     fig, ax = visualize_persistence_diagram(pts)
        ... except ValueError:
        ...     pass
    """
    if not PERSIM_AVAILABLE:
        raise ValueError("persim library is not available.")
    
    dgms = compute_persistence_diagram(points)
    h1_dgm = dgms[1] if len(dgms) > 1 else np.array([])
    
    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=(8, 6))
        if len(h1_dgm) > 0:
            h1_dgm = np.array(h1_dgm)
            ax.scatter(h1_dgm[:, 0], h1_dgm[:, 1], alpha=0.7, s=50)
        ax.set_title(title or 'Persistence Diagram (H₁)')
        ax.grid(True, alpha=0.3)
        return fig, ax
    return None


def main():
    """Main execution function for Visualizer script.

    Args:
        None

    Returns:
        None
    """
    import argparse
    parser = argparse.ArgumentParser(description="TDA Visualizer CLI")
    parser.add_argument('--analyze-stability', action='store_true', help='Ejecuta análisis de estabilidad H.E.1 (100 realizaciones)')
    parser.add_argument('--compare-kmeans', action='store_true', help='Ejecuta comparación visual de TDA vs PCA y K-Medias')
    parser.add_argument('--shape', type=str, default='torus')
    args = parser.parse_args()
    
    if args.analyze_stability:
        plot_stability_analysis(shape=args.shape)
    elif args.compare_kmeans:
        plot_tda_vs_kmeans_pca()
    else:
        print("Modo de uso: visualizer.py [--analyze-stability] [--compare-kmeans] [--shape SHAPE]")

if __name__ == "__main__":
    main()