"""Periodic cell-list utilities for overlap checks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from spherepackgen.domain.domain import as_box_lengths


class PeriodicCellList:
    """Simple periodic cell list for broad-phase neighbor queries."""

    def __init__(self, box_length: float, cutoff: float):
        self.box_length = as_box_lengths(box_length)
        self.cutoff = max(float(cutoff), 1.0e-12)
        self.n_cells = np.maximum(1, np.floor(self.box_length / self.cutoff).astype(int))
        self.cell_size = self.box_length / self.n_cells
        self.cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        self.positions: np.ndarray | None = None

    def _cell_index(self, position: np.ndarray) -> tuple[int, int, int]:
        idx = np.floor(np.mod(position, self.box_length) / self.cell_size).astype(int)
        idx = np.mod(idx, self.n_cells)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def build(self, positions: np.ndarray):
        self.cells.clear()
        self.positions = np.asarray(positions, dtype=float)
        for i, pos in enumerate(self.positions):
            self.cells[self._cell_index(pos)].append(i)

    def add(self, index: int, position: np.ndarray):
        self.cells[self._cell_index(position)].append(int(index))

    def nearby_indices(self, position: np.ndarray, search_radius: float | None = None):
        if search_radius is None:
            span = np.ones(3, dtype=int)
        else:
            span = np.maximum(1, np.ceil(search_radius / self.cell_size).astype(int))
        base = self._cell_index(position)
        axes = []
        for axis in range(3):
            count, extent = int(self.n_cells[axis]), int(span[axis])
            axes.append(range(count) if 2 * extent + 1 >= count else
                        [(base[axis] + d) % count for d in range(-extent, extent + 1)])
        for x in axes[0]:
            for y in axes[1]:
                for z in axes[2]:
                    yield from self.cells.get((x, y, z), [])


def minimum_image_delta(a: np.ndarray, b: np.ndarray, box_length: float) -> np.ndarray:
    diff = np.asarray(a) - np.asarray(b)
    box_length = np.asarray(box_length, dtype=float)
    return diff - box_length * np.rint(diff / box_length)


def pair_gap(position_i, radius_i, position_j, radius_j, box_length: float) -> float:
    delta = minimum_image_delta(position_i, position_j, box_length)
    return float(np.linalg.norm(delta) - radius_i - radius_j)


def find_overlap_pairs(
    positions: np.ndarray,
    radii: np.ndarray,
    box_length: float,
    tolerance: float = 0.0,
):
    """Return periodic overlap pairs and minimum gap."""
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(radii) < 2:
        return [], float("inf")
    max_cutoff = 2.0 * float(np.max(radii))
    cells = PeriodicCellList(box_length, max_cutoff)
    cells.build(positions)
    pairs = []
    min_gap = float("inf")
    max_r = float(np.max(radii))
    for i, pos in enumerate(positions):
        search = radii[i] + max_r
        for j in cells.nearby_indices(pos, search):
            if j <= i:
                continue
            gap = pair_gap(pos, radii[i], positions[j], radii[j], box_length)
            min_gap = min(min_gap, gap)
            if gap < -tolerance:
                pairs.append((int(i), int(j), float(gap)))
    return pairs, min_gap
