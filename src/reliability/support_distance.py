from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_reference_standardization(
    reference_observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit reference-only z-score statistics.

    This matches the frozen Gaussian methodology.
    """
    reference_observations = np.asarray(
        reference_observations,
        dtype=np.float64,
    )

    if reference_observations.ndim != 2:
        raise ValueError(
            "reference_observations must have shape (N, D)."
        )

    if len(reference_observations) == 0:
        raise ValueError(
            "reference_observations is empty."
        )

    mean = np.mean(
        reference_observations,
        axis=0,
    )

    std = np.std(
        reference_observations,
        axis=0,
    )

    std = np.maximum(
        std,
        1e-8,
    )

    return mean, std


def standardize(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply reference-only z-score standardization."""
    observations = np.asarray(
        observations,
        dtype=np.float64,
    )

    mean = np.asarray(
        mean,
        dtype=np.float64,
    )

    std = np.asarray(
        std,
        dtype=np.float64,
    )

    return (
        observations - mean
    ) / std


def build_reference_index(
    standardized_reference: np.ndarray,
) -> NearestNeighbors:
    """Build the canonical Euclidean nearest-neighbor index."""
    standardized_reference = np.asarray(
        standardized_reference,
        dtype=np.float64,
    )

    if standardized_reference.ndim != 2:
        raise ValueError(
            "standardized_reference must have shape (N, D)."
        )

    index = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    index.fit(
        standardized_reference
    )

    return index


def nearest_support_distance(
    observation: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    reference_index: NearestNeighbors,
) -> tuple[float, int]:
    """
    Return nearest-reference distance and reference index.

    Distance is measured in the reference-standardized observation
    space, matching the frozen Gaussian methodology.
    """
    standardized = standardize(
        np.asarray(
            observation,
            dtype=np.float64,
        ).reshape(1, -1),
        mean,
        std,
    )

    distances, indices = reference_index.kneighbors(
        standardized,
        n_neighbors=1,
    )

    return (
        float(distances[0, 0]),
        int(indices[0, 0]),
    )
