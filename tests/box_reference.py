"""Small-system reference geometry, independent of production neighbor indexing."""
import itertools
import numpy as np


def image_distances(positions, lengths):
    positions = np.asarray(positions)
    lengths = np.broadcast_to(lengths, (3,))
    offsets = np.array(list(itertools.product((-1, 0, 1), repeat=3))) * lengths
    result = np.full((len(positions), len(positions)), np.inf)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.min(np.linalg.norm(positions[i] - positions[j] + offsets, axis=1))
            result[i, j] = result[j, i] = distance
    return result
