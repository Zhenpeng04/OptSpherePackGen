"""Periodic domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_box_lengths(value) -> np.ndarray:
    """Normalize a legacy scalar or three orthogonal side lengths."""
    lengths = np.asarray(value, dtype=float)
    if lengths.ndim == 0:
        lengths = np.repeat(lengths, 3)
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Box dimensions must be three positive finite lengths or a positive scalar.")
    return lengths.copy()


def box_volume(value) -> float:
    return float(np.prod(as_box_lengths(value)))


@dataclass(frozen=True)
class PeriodicBox:
    box_lengths: np.ndarray

    def __post_init__(self):
        lengths = as_box_lengths(self.box_lengths)
        lengths.setflags(write=False)
        object.__setattr__(self, "box_lengths", lengths)

    @property
    def box_length(self):
        """Legacy scalar for cubes; a vector for rectangular boxes."""
        if np.all(self.box_lengths == self.box_lengths[0]):
            return float(self.box_lengths[0])
        return self.box_lengths

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        return np.mod(positions, self.box_lengths)

    def delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = np.asarray(a) - np.asarray(b)
        return diff - self.box_lengths * np.rint(diff / self.box_lengths)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(self.delta(a, b)))

    @property
    def volume(self) -> float:
        return float(np.prod(self.box_lengths))


class PeriodicCube(PeriodicBox):
    """Backward-compatible cubic domain constructor."""

    def __init__(self, box_length: float):
        super().__init__(as_box_lengths(box_length))
