"""YAML configuration loading."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from spherepackgen.config.schema import (
    AlgorithmConfig,
    AnalysisConfig,
    DomainConfig,
    OutputConfig,
    PackingConfig,
    ParticlesConfig,
    PhysicalConfig,
    ProjectConfig,
    RuntimeConfig,
    SizeDistributionConfig,
    StructureConfig,
    ValidationConfig,
)
from spherepackgen.config.units import parse_length_to_meters
from spherepackgen.domain.enums import DensityClass, SizeDistributionClass, SpatialOrderClass, infer_density_class


def _get(mapping: dict[str, Any], key: str, default=None):
    value = mapping.get(key, default)
    return default if value == "auto" else value


def _num_particles(value) -> int | None:
    if value is None or str(value).strip().lower() == "auto":
        return None
    return int(value)


def load_config(path: str | Path) -> PackingConfig:
    """Load a YAML config into a ``PackingConfig`` dataclass."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return config_from_mapping(raw, path.parent)


def config_from_mapping(raw: dict[str, Any], base_directory: str | Path = ".") -> PackingConfig:
    """Parse in-memory configuration using the same rules as YAML input."""
    base_directory = Path(base_directory)

    project_raw = raw.get("project", {})
    structure_raw = raw.get("structure", {})
    physical_raw = raw.get("physical", {})
    domain_raw = raw.get("domain", raw.get("simulation_box", {}))
    particles_raw = raw.get("particles", {})
    size_raw = raw.get("size_distribution", {})
    algorithm_raw = raw.get("algorithm", {})
    analysis_raw = raw.get("analysis", {})
    validation_raw = raw.get("validation", {})
    output_raw = raw.get("output", {})
    runtime_raw = raw.get("runtime", {})

    mean_diameter = physical_raw.get("mean_diameter")
    if mean_diameter is None:
        raise ValueError("physical.mean_diameter is required")
    medium_thickness = physical_raw.get("medium_thickness", physical_raw.get("thickness"))

    size_type = structure_raw.get("size_distribution", size_raw.get("type", "monodisperse"))
    spatial_order = structure_raw.get("spatial_order", "hard_core_random")
    spatial_order_enum = SpatialOrderClass.from_value(spatial_order)

    packing_fraction = physical_raw.get("packing_fraction")
    covered_volume_fraction = physical_raw.get(
        "covered_volume_fraction",
        physical_raw.get("target_covered_volume_fraction", physical_raw.get("coverage_fraction")),
    )
    nominal_volume_fraction = physical_raw.get("nominal_volume_fraction")

    if spatial_order_enum == SpatialOrderClass.OVERLAPPING_RANDOM:
        if covered_volume_fraction is not None:
            covered_volume_fraction = float(covered_volume_fraction)
            if not (0.0 < covered_volume_fraction < 1.0):
                raise ValueError("physical.covered_volume_fraction must be between 0 and 1 for overlapping_random")
            nominal_volume_fraction = -math.log1p(-covered_volume_fraction)
            packing_fraction = nominal_volume_fraction
        elif nominal_volume_fraction is not None:
            nominal_volume_fraction = float(nominal_volume_fraction)
            if nominal_volume_fraction <= 0:
                raise ValueError("physical.nominal_volume_fraction must be positive for overlapping_random")
            covered_volume_fraction = 1.0 - math.exp(-nominal_volume_fraction)
            packing_fraction = nominal_volume_fraction
        elif packing_fraction is not None:
            nominal_volume_fraction = float(packing_fraction)
            if nominal_volume_fraction <= 0:
                raise ValueError("physical.packing_fraction must be positive for overlapping_random")
            covered_volume_fraction = 1.0 - math.exp(-nominal_volume_fraction)
            packing_fraction = nominal_volume_fraction
        else:
            raise ValueError(
                "overlapping_random requires physical.covered_volume_fraction "
                "or physical.nominal_volume_fraction"
            )
    elif packing_fraction is None:
        raise ValueError("physical.packing_fraction is required")

    inferred_density_class = (
        DensityClass.OVERLAPPING_NOMINAL
        if spatial_order_enum == SpatialOrderClass.OVERLAPPING_RANDOM
        else infer_density_class(float(packing_fraction))
    )

    box_size = _get(domain_raw, "box_size", None)
    box_size_m = None if box_size is None else parse_length_to_meters(box_size)
    length = domain_raw.get("length")
    depth = domain_raw.get("depth")
    domain_type = str(domain_raw.get("type", domain_raw.get("boundary_condition", "periodic_cube")))
    if domain_type not in {"periodic_cube", "periodic_box"}:
        raise ValueError("domain.type must be periodic_cube or periodic_box")
    if (length is None) != (depth is None):
        raise ValueError("Specify both domain.length and domain.depth for a fixed periodic box.")
    if domain_type == "periodic_box" and length is None:
        raise ValueError("periodic_box requires domain.length and domain.depth")
    if length is not None and (box_size is not None or medium_thickness is not None):
        raise ValueError("Use domain.length/depth without legacy box_size or medium_thickness inputs.")

    output_path = Path(output_raw.get("path", f"results/{project_raw.get('name', 'case')}"))
    if output_raw.get("relative_to_config", False) and not output_path.is_absolute():
        output_path = base_directory / output_path
    size_file = size_raw.get("size_file")
    if size_file:
        size_file = Path(size_file)
        if not size_file.is_absolute():
            size_file = base_directory / size_file
        size_file = str(size_file)

    return PackingConfig(
        project=ProjectConfig(
            name=str(project_raw.get("name", "spherepackgen_case")),
            description=str(project_raw.get("description", "")),
        ),
        structure=StructureConfig(
            size_distribution=SizeDistributionClass.from_value(size_type),
            spatial_order=spatial_order_enum,
            density_class=inferred_density_class,
        ),
        physical=PhysicalConfig(
            mean_diameter_m=parse_length_to_meters(mean_diameter),
            packing_fraction=float(packing_fraction),
            medium_thickness_m=None if medium_thickness is None else parse_length_to_meters(medium_thickness),
            covered_volume_fraction=None if covered_volume_fraction is None else float(covered_volume_fraction),
            nominal_volume_fraction=None if nominal_volume_fraction is None else float(nominal_volume_fraction),
        ),
        domain=DomainConfig(
            type="periodic_box" if length is not None else domain_type,
            box_size_m=box_size_m,
            length_m=None if length is None else parse_length_to_meters(length),
            depth_m=None if depth is None else parse_length_to_meters(depth),
        ),
        particles=ParticlesConfig(
            num_particles=_num_particles(particles_raw.get("num_particles", domain_raw.get("num_particles"))),
        ),
        size_distribution=SizeDistributionConfig(
            type=str(size_raw.get("type", size_type)),
            cv_radius=float(size_raw.get("CV_radius", size_raw.get("cv_radius",
                0.03 if SizeDistributionClass.from_value(size_type) == SizeDistributionClass.QUASI_MONODISPERSE else 0.0))),
            min_radius_factor=size_raw.get("min_radius_factor"),
            max_radius_factor=size_raw.get("max_radius_factor"),
            distribution=size_raw.get("distribution"),
            parameters=dict(size_raw.get("parameters", {})),
            species=list(size_raw.get("species", [])),
            size_file=size_file,
            file_sha256=size_raw.get("file_sha256"),
            file_values=str(size_raw.get("file_values", size_raw.get("file_value", "diameter"))),
            file_column=size_raw.get("file_column"),
        ),
        algorithm=AlgorithmConfig(
            name=str(algorithm_raw.get("name", runtime_raw.get("generator", "default"))),
            parameters=dict(algorithm_raw.get("parameters", {})),
        ),
        analysis=AnalysisConfig(
            compute_g2=bool(analysis_raw.get("compute_g2", True)),
            compute_Sk=bool(analysis_raw.get("compute_Sk", analysis_raw.get("compute_sk", True))),
            bins=int(analysis_raw.get("bins", 80)),
            k_max_index=int(analysis_raw.get("k_max_index", 5)),
        ),
        validation=ValidationConfig(
            tolerance_phi=float(validation_raw.get("tolerance_phi", 1.0e-6)),
            tolerance_overlap=float(validation_raw.get("tolerance_overlap", 1.0e-5)),
            require_non_overlap=bool(
                validation_raw.get("require_non_overlap", spatial_order_enum != SpatialOrderClass.OVERLAPPING_RANDOM)
            ),
        ),
        output=OutputConfig(
            path=output_path,
            coordinate_unit=str(output_raw.get("coordinate_unit", "um")),
            formats=list(output_raw.get("formats", ["csv", "json"])),
            save_dimensionless_coordinates=bool(output_raw.get("save_dimensionless_coordinates", True)),
            save_real_coordinates=bool(output_raw.get("save_real_coordinates", True)),
            make_plots=bool(output_raw.get("make_plots", True)),
            overwrite=bool(output_raw.get("overwrite", True)),
        ),
        runtime=RuntimeConfig(
            random_seed=runtime_raw.get("random_seed"),
            log_level=str(runtime_raw.get("log_level", "info")),
        ),
        raw=raw,
    )
