import numpy as np
from typing import Tuple

try:
    from persim import wasserstein_distance as persim_wasserstein
    from persim import bottleneck_distance as persim_bottleneck
except ImportError:
    persim_wasserstein = None
    persim_bottleneck = None


def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calculates the Wasserstein distance between two persistence diagrams.

    Args:
        dgm1 (np.ndarray): First persistence diagram of shape (n, 2) where each row is [birth, death].
        dgm2 (np.ndarray): Second persistence diagram of shape (m, 2) where each row is [birth, death].

    Returns:
        float: The Wasserstein distance between the two diagrams.

    Raises:
        ImportError: If persim library is not available.
        ValueError: If input arrays are not of shape (n, 2) or (m, 2).

    Examples:
        >>> import numpy as np
        >>> dgm1 = np.array([[0.0, 1.0], [1.2, 2.0]])
        >>> dgm2 = np.array([[0.0, 1.1], [1.0, 1.8]])
        >>> # Assuming persim is installed:
        >>> # wasserstein_distance(dgm1, dgm2)
    """
    if persim_wasserstein is None:
        raise ImportError("persim library is required for wasserstein_distance")
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")
    return float(persim_wasserstein(dgm1, dgm2))


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calculates the Bottleneck distance between two persistence diagrams.

    Args:
        dgm1 (np.ndarray): First persistence diagram of shape (n, 2) where each row is [birth, death].
        dgm2 (np.ndarray): Second persistence diagram of shape (m, 2) where each row is [birth, death].

    Returns:
        float: The Bottleneck distance between the two diagrams.

    Raises:
        ImportError: If persim library is not available.
        ValueError: If input arrays are not of shape (n, 2) or (m, 2).

    Examples:
        >>> import numpy as np
        >>> dgm1 = np.array([[0.0, 1.0], [1.2, 2.0]])
        >>> dgm2 = np.array([[0.0, 1.1], [1.0, 1.8]])
        >>> # Assuming persim is installed:
        >>> # bottleneck_distance(dgm1, dgm2)
    """
    if persim_bottleneck is None:
        raise ImportError("persim library is required for bottleneck_distance")
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")
    return float(persim_bottleneck(dgm1, dgm2))


def betti_numbers(persistence_diagram: np.ndarray) -> Tuple[int, int]:
    """Extracts Betti numbers (beta_0, beta_1) from a persistence diagram.

    Args:
        persistence_diagram (np.ndarray): Persistence diagram of shape (n, 3) where each row is [birth, death, dimension].
            Dimension 0 corresponds to H_0 (connected components), dimension 1 to H_1 (loops).

    Returns:
        Tuple[int, int]: A tuple (beta_0, beta_1) where:
            beta_0: Number of connected components (points in H_0 with infinite death or finite)
            beta_1: Number of 1-dimensional holes (points in H_1)

    Raises:
        ValueError: If input array is not of shape (n, 3) or contains invalid dimensions.

    Examples:
        >>> import numpy as np
        >>> dgm = np.array([
        ...     [0.0, 1.0, 0.0],
        ...     [0.0, np.inf, 0.0],
        ...     [0.5, 1.2, 1.0]
        ... ])
        >>> betti_numbers(dgm)
        (2, 1)
    """
    if persistence_diagram.ndim != 2 or persistence_diagram.shape[1] != 3:
        raise ValueError("persistence_diagram must be of shape (n, 3) with [birth, death, dimension]")
    if not np.all(np.isin(persistence_diagram[:, 2], [0, 1])):
        raise ValueError("Dimension column must contain only 0 (H_0) or 1 (H_1)")

    h0 = persistence_diagram[persistence_diagram[:, 2] == 0]
    h1 = persistence_diagram[persistence_diagram[:, 2] == 1]

    # beta_0: number of connected components (finite or infinite death)
    beta_0 = int(h0.shape[0])
    # beta_1: number of 1-dimensional holes
    beta_1 = int(h1.shape[0])

    return (beta_0, beta_1)
