"""Particle geometry container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spherepackgen.domain.domain import PeriodicBox, as_box_lengths


@dataclass
class ParticleSet:
    positions: np.ndarray
    radii: np.ndarray
    box_length: float | np.ndarray
    species_id: np.ndarray | None = None
    material_id: np.ndarray | None = None

    def __post_init__(self):
        lengths = as_box_lengths(self.box_length)
        self.box_length = float(lengths[0]) if np.all(lengths == lengths[0]) else lengths
        self.positions = np.asarray(self.positions, dtype=float)
        self.radii = np.asarray(self.radii, dtype=float)
        if self.radii.ndim != 1:
            raise ValueError("radii must have shape (N,)")
        n = len(self.radii)
        if self.species_id is None:
            self.species_id = np.zeros(n, dtype=int)
        if self.material_id is None:
            self.material_id = np.zeros(n, dtype=int)
        self.species_id = np.asarray(self.species_id)
        self.material_id = np.asarray(self.material_id)
        self.validate_geometry()

    def validate_geometry(self) -> None:
        """Check input integrity again after possible in-place API mutations."""
        lengths = as_box_lengths(self.box_length)
        if not np.isfinite(np.prod(lengths)) or np.prod(lengths) <= 0:
            raise ValueError("Box volume must be positive and finite")
        positions = np.asarray(self.positions)
        radii = np.asarray(self.radii)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if radii.ndim != 1 or len(radii) != len(positions):
            raise ValueError("radii must have shape (N,)")
        if not len(radii):
            raise ValueError("ParticleSet must contain at least one particle")
        if not np.all(np.isfinite(positions)):
            raise ValueError("all positions must be finite")
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError("all radii must be positive and finite")
        for name in ("species_id", "material_id"):
            values = np.asarray(getattr(self, name))
            if values.shape != (len(radii),):
                raise ValueError(f"{name} must have shape (N,)")
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f"{name} must contain non-negative integers")
            if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError(f"{name} must contain non-negative integers")

    @property
    def n_particles(self) -> int:
        return int(len(self.radii))

    @property
    def domain(self) -> PeriodicBox:
        return PeriodicBox(self.box_lengths)

    @property
    def box_lengths(self) -> np.ndarray:
        return as_box_lengths(self.box_length)

    @property
    def packing_fraction(self) -> float:
        return float(np.sum(4.0 * np.pi * self.radii**3 / 3.0) / self.domain.volume)

    def wrapped(self) -> "ParticleSet":
        return ParticleSet(
            positions=self.domain.wrap(self.positions),
            radii=self.radii.copy(),
            box_length=self.box_length,
            species_id=self.species_id.copy(),
            material_id=self.material_id.copy(),
        )

    def scaled(self, factor: float) -> "ParticleSet":
        return ParticleSet(
            positions=self.positions * factor,
            radii=self.radii * factor,
            box_length=self.box_length * factor,
            species_id=self.species_id.copy(),
            material_id=self.material_id.copy(),
        )
