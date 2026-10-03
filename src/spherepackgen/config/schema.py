"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math
from numbers import Integral

from spherepackgen.domain.enums import DensityClass, SizeDistributionClass, SpatialOrderClass


@dataclass
class ProjectConfig:
    name: str = "spherepackgen_case"
    description: str = ""


@dataclass
class StructureConfig:
    size_distribution: SizeDistributionClass = SizeDistributionClass.MONODISPERSE
    spatial_order: SpatialOrderClass = SpatialOrderClass.HARD_CORE_RANDOM
    density_class: DensityClass = DensityClass.SPARSE


@dataclass
class PhysicalConfig:
    mean_diameter_m: float
    packing_fraction: float
    medium_thickness_m: float | None = None
    covered_volume_fraction: float | None = None
    nominal_volume_fraction: float | None = None


@dataclass
class DomainConfig:
    type: str = "periodic_cube"
    box_size_m: float | None = None
    length_m: float | None = None
    depth_m: float | None = None

    @property
    def fixed_dimensions(self) -> bool:
        return self.length_m is not None or self.depth_m is not None


@dataclass
class ParticlesConfig:
    num_particles: int | None = None


@dataclass
class SizeDistributionConfig:
    type: str = "monodisperse"
    cv_radius: float = 0.0
    min_radius_factor: float | None = None
    max_radius_factor: float | None = None
    distribution: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    species: list[dict[str, Any]] = field(default_factory=list)
    size_file: str | None = None
    file_sha256: str | None = None
    file_values: str = "diameter"
    file_column: str | int | None = None


@dataclass
class AlgorithmConfig:
    name: str = "default"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    compute_g2: bool = True
    compute_Sk: bool = True
    bins: int = 80
    k_max_index: int = 5


@dataclass
class ValidationConfig:
    tolerance_phi: float = 1.0e-6
    tolerance_overlap: float = 1.0e-5
    require_non_overlap: bool = True

    def __post_init__(self):
        for name in ("tolerance_phi", "tolerance_overlap"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"validation.{name} must be non-negative and finite")


@dataclass
class OutputConfig:
    path: Path = Path("results/case")
    coordinate_unit: str = "um"
    formats: list[str] = field(default_factory=lambda: ["csv", "json"])
    save_dimensionless_coordinates: bool = True
    save_real_coordinates: bool = True
    make_plots: bool = True
    overwrite: bool = True


@dataclass
class RuntimeConfig:
    random_seed: int | None = None
    log_level: str = "info"

    def __post_init__(self):
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, Integral) or self.random_seed < 0
        ):
            raise ValueError("runtime.random_seed must be a non-negative integer or null")


@dataclass
class PackingConfig:
    project: ProjectConfig
    structure: StructureConfig
    physical: PhysicalConfig
    domain: DomainConfig
    particles: ParticlesConfig
    size_distribution: SizeDistributionConfig
    algorithm: AlgorithmConfig
    analysis: AnalysisConfig
    validation: ValidationConfig
    output: OutputConfig
    runtime: RuntimeConfig
    raw: dict[str, Any] = field(default_factory=dict)
