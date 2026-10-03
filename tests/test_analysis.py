import numpy as np

from spherepackgen.analysis.descriptors import nearest_neighbor_distribution, pair_correlation
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.utils.cell_list import minimum_image_delta


def _brute_nearest(positions: np.ndarray, box_length: float) -> np.ndarray:
    nearest = np.full(len(positions), np.inf, dtype=float)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = minimum_image_delta(positions[i], positions[j], box_length)
            distance = float(np.linalg.norm(delta))
            nearest[i] = min(nearest[i], distance)
            nearest[j] = min(nearest[j], distance)
    return nearest


def _brute_pair_histogram(positions: np.ndarray, box_length: float, bins: int) -> np.ndarray:
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = minimum_image_delta(positions[i], positions[j], box_length)
            distances.append(float(np.linalg.norm(delta)))
    counts, _ = np.histogram(np.asarray(distances, dtype=float), bins=bins, range=(0.0, box_length / 2.0))
    return counts


def test_periodic_ckdtree_nearest_neighbor_matches_bruteforce():
    particles = ParticleSet(
        positions=np.array(
            [
                [0.10, 0.10, 0.10],
                [9.90, 0.10, 0.10],
                [4.00, 4.00, 4.00],
                [6.20, 4.00, 4.00],
            ],
            dtype=float,
        ),
        radii=np.full(4, 0.1, dtype=float),
        box_length=10.0,
    )

    result = nearest_neighbor_distribution(particles)

    assert result["backend"] == "periodic_ckdtree"
    assert np.allclose(result["distances"], _brute_nearest(particles.positions, particles.box_length))


def test_periodic_ckdtree_pair_correlation_counts_match_bruteforce():
    particles = ParticleSet(
        positions=np.array(
            [
                [0.10, 0.10, 0.10],
                [9.90, 0.10, 0.10],
                [2.00, 2.00, 2.00],
                [4.00, 2.00, 2.00],
                [7.50, 7.50, 7.50],
            ],
            dtype=float,
        ),
        radii=np.full(5, 0.1, dtype=float),
        box_length=10.0,
    )

    result = pair_correlation(particles, bins=10)

    assert result["backend"] == "periodic_ckdtree_count_neighbors"
    assert np.array_equal(result["counts"], _brute_pair_histogram(particles.positions, particles.box_length, bins=10))
