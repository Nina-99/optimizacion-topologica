"""Preprocessing utilities for persistence diagrams in TDA pipeline.

Provides functions for filtering outliers, scaling, and summarizing diagrams.
"""
from typing import List
import numpy as np


def filter_persistence_diagram(dgm: List[List[float]], threshold: float) -> List[List[float]]:
    """Remove persistence intervals shorter than the threshold.

    Args:
        dgm (List[List[float]]): Persistence diagram of shape (n_pairs, 2).
        threshold (float): Minimum persistence length to retain a point.

    Returns:
        List[List[float]]: Diagram after removing short-lived intervals.

    Examples:
        >>> dgm = [[[0.1, 1.2], [0.5, 0.6]]]
        >>> filter_persistence_diagram(dgm, 0.5)
        [[[0.1, 1.2]]]
    """
    filtered = []
    for dim_dgm in dgm:
        kept = []
        for birth, death in dim_dgm:
            persistence = death - birth
            if persistence >= threshold:
                kept.append([birth, death])
        filtered.append(kept)
    return filtered


def normalize_diagram(dgm: List[List[float]], diameter: float) -> List[List[float]]:
    """Normalize persistence diagram by scaling birth/death coordinates.

    Args:
        dgm (List[List[float]]): Persistence diagram of shape (n_pairs, 2).
        diameter (float): Diameter of the point cloud (max pairwise distance).

    Returns:
        List[List[float]]: Scaled diagram.

    Examples:
        >>> dgm = [[[0.5, 1.5]]]
        >>> normalize_diagram(dgm, 2.0)
        [[[0.25, 0.75]]]
    """
    normalized = []
    for dim_dgm in dgm:
        norm_dim = []
        for birth, death in dim_dgm:
            # Avoid division by zero
            factor = 1.0 / max(diameter, 1e-8)
            norm_birth = birth * factor
            norm_death = death * factor
            norm_dim.append([norm_birth, norm_death])
        normalized.append(norm_dim)
    return normalized


def get_persistence_histogram(dgm: List[List[float]], bins: int = 10) -> np.ndarray:
    """Compute histogram of persistence lifetimes.

    Args:
        dgm (List[List[float]]): Persistence diagram (n_pairs, 2).
        bins (int): Number of bins for the histogram. Defaults to 10.

    Returns:
        numpy.ndarray: 1D array representing the distribution of lifetimes.

    Examples:
        >>> import numpy as np
        >>> dgm = np.array([[1.2, 0.2], [0.8, 0.4]])
        >>> get_persistence_histogram(dgm, bins=3)
        array([1., 0., 1.])
    """
    # Flatten all lifetimes
    lifetimes = []
    for death, birth in dgm:  # Note: persim returns [death, birth] order sometimes
        if len(death.shape) > 1:  # Handle matrix case
            for b, d in death:
                lifetimes.append(d - b)
        else:
            lifetimes.append(death - birth)
    
    if not lifetimes:
        return np.zeros(bins)
    
    # Determine histogram range based on max lifetime
    max_life = max(lifetimes)
    if max_life == 0:
        max_life = 1.0
    
    hist, _ = np.histogram(lifetimes, bins=bins, range=(0, max_life))
    return hist
