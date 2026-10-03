"""Workflow result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.domain import as_box_lengths


@dataclass
class ResolvedParameters:
    n_particles: int
    box_length: float | np.ndarray
    target_phi: float
    mean_diameter_m: float
    target_medium_thickness_m: float | None
    box_length_m: float | np.ndarray
    output_unit: str
    output_unit_m: float
    radii: np.ndarray
    radius_mean: float
    radius_cv: float
    fixed_dimensions: bool = False
    resolved_phi: float | None = None
    density_resolution_tolerance: float = 0.0

    @property
    def box_lengths(self) -> np.ndarray:
        return as_box_lengths(self.box_length)

    @property
    def box_lengths_m(self) -> np.ndarray:
        return self.box_lengths * self.mean_diameter_m

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box_lengths"] = self.box_lengths.tolist()
        data["box_lengths_m"] = self.box_lengths_m.tolist()
        for key in ("box_length", "box_length_m"):
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        data["radii"] = {
            "count": int(len(self.radii)),
            "min": float(np.min(self.radii)),
            "max": float(np.max(self.radii)),
            "mean": float(np.mean(self.radii)),
            "cv": float(np.std(self.radii) / np.mean(self.radii)),
        }
        return data


@dataclass
class PackingResult:
    particle_set: ParticleSet
    generator_name: str
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    elapsed_time_s: float = 0.0


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]
    tolerances: dict[str, float]


@dataclass
class StructuralDescriptors:
    nearest_neighbor: dict[str, Any] = field(default_factory=dict)
    g2: dict[str, Any] = field(default_factory=dict)
    Sk: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultBundle:
    config: Any
    resolved: ResolvedParameters
    result: PackingResult
    validation: ValidationReport
    descriptors: StructuralDescriptors
    exported_files: list[str]
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def readiness(self) -> dict[str, bool | None]:
        """Geometry validity and readiness for the requested scientific use.

        A null target means that the generator defines no additional target.
        Candidate files may be exported even when export_ready is false.
        """
        target = self.result.diagnostics.get("structure_target_met")
        target = None if target is None else bool(target)
        geometry = bool(self.validation.passed)
        return {
            "geometry_valid": geometry,
            "structure_target_met": target,
            "export_ready": bool(geometry and target is not False and self.result.status == "success"),
        }
