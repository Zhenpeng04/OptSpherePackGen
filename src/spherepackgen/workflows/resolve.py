"""Parameter resolution for generation workflows."""

from __future__ import annotations

import numpy as np

from spherepackgen.config.schema import PackingConfig
from spherepackgen.config.units import unit_scale_to_meters
from spherepackgen.domain.enums import LatticeType, SpatialOrderClass, SizeDistributionClass, lattice_basis_count
from spherepackgen.domain.domain import as_box_lengths
from spherepackgen.domain.result import ResolvedParameters
from spherepackgen.generators.radii import sample_radii


def _crystal_n_from_unit_cells(config: PackingConfig) -> int | None:
    if config.structure.spatial_order != SpatialOrderClass.PERIODIC_CRYSTAL:
        return None
    params = config.algorithm.parameters
    if "unit_cells" not in params:
        return None
    lattice = LatticeType.from_value(params.get("lattice_type", "FCC"))
    return lattice_basis_count(lattice) * int(params["unit_cells"]) ** 3


def _basis_count(config: PackingConfig) -> int:
    params = config.algorithm.parameters
    lattice = LatticeType.from_value(params.get("lattice_type", "FCC"))
    return lattice_basis_count(lattice)


def _target_box_length_from_macro_inputs(config: PackingConfig) -> float | None:
    """Return the dimensionless target side length implied by macro physical inputs."""
    if config.physical.medium_thickness_m is not None:
        return config.physical.medium_thickness_m / config.physical.mean_diameter_m
    if config.domain.box_size_m is not None:
        return config.domain.box_size_m / config.physical.mean_diameter_m
    return None


def _estimate_mean_particle_volume(config: PackingConfig, rng: np.random.Generator) -> float:
    """Estimate dimensionless mean particle volume for automatic N calculation."""
    probe_count = 20000
    probe_radii, _ = sample_radii(config, probe_count, rng)
    return float(np.mean(4.0 * np.pi * probe_radii**3 / 3.0))


def _auto_particle_count(config: PackingConfig, rng: np.random.Generator) -> int | None:
    target_box_length = _target_box_length_from_macro_inputs(config)
    if target_box_length is None:
        return None
    mean_particle_volume = _estimate_mean_particle_volume(config, rng)
    estimate = int(round(config.physical.packing_fraction * target_box_length**3 / mean_particle_volume))
    estimate = max(1, estimate)
    if config.structure.spatial_order == SpatialOrderClass.PERIODIC_CRYSTAL:
        basis_count = _basis_count(config)
        unit_cells = max(1, int(round((estimate / basis_count) ** (1.0 / 3.0))))
        return basis_count * unit_cells**3
    return estimate


def _fixed_box_radii(config: PackingConfig, lengths: np.ndarray, rng, max_particles=None):
    """Choose a count using actual sampled volumes, keeping mean diameter fixed."""
    volume = float(np.prod(lengths))
    target_volume = config.physical.packing_fraction * volume
    if config.structure.spatial_order == SpatialOrderClass.PERIODIC_CRYSTAL:
        params = config.algorithm.parameters
        lattice = LatticeType.from_value(params.get("lattice_type", "FCC"))
        if lattice == LatticeType.HCP:
            raise ValueError("For HCP, use a periodic_cube configuration with automatically resolved dimensions.")
        if config.structure.size_distribution != SizeDistributionClass.MONODISPERSE:
            raise ValueError("Fixed-box crystals require monodisperse particles.")
        basis = lattice_basis_count(lattice)
        spacing = (basis * np.pi / (6.0 * config.physical.packing_fraction)) ** (1.0 / 3.0)
        requested = params.get("unit_cells")
        if requested is not None:
            cells = np.asarray(requested)
            cells = np.repeat(cells, 3) if cells.ndim == 0 else cells
            if cells.shape != (3,) or np.any(cells < 1) or np.any(cells != cells.astype(int)):
                raise ValueError("unit_cells must be a positive integer or three positive integers.")
            cells = cells.astype(int)
        else:
            cells = np.maximum(1, np.rint(lengths / spacing).astype(int))
        spacings = lengths / cells
        if not np.allclose(spacings, spacings[0], rtol=1e-8, atol=1e-10):
            raise ValueError("Crystal dimensions must fit complete, unstrained unit cells. Set unit_cells: [nx, ny, nz] with equal lengths per cell.")
        n = int(basis * np.prod(cells))
        if max_particles is not None and n > max_particles:
            raise ValueError(f"Resolved particle count exceeds {max_particles:,}.")
        if config.particles.num_particles is not None and config.particles.num_particles != n:
            raise ValueError("Manual crystal particle count conflicts with the complete unit-cell grid.")
        return np.full(n, 0.5)
    if config.particles.num_particles is not None:
        if config.particles.num_particles < 1:
            raise ValueError("particles.num_particles must be positive")
        if max_particles is not None and config.particles.num_particles > max_particles:
            raise ValueError(f"Particle count exceeds {max_particles:,}.")
        return sample_radii(config, config.particles.num_particles, rng)[0]
    if config.structure.size_distribution == SizeDistributionClass.MONODISPERSE:
        n = max(1, int(round(target_volume / (np.pi / 6.0))))
        if max_particles is not None and n > max_particles:
            raise ValueError(f"Resolved particle count exceeds {max_particles:,}.")
        return np.full(n, 0.5)
    mean_volume = _estimate_mean_particle_volume(config, rng)
    pool_size = max(16, int(np.ceil(1.5 * target_volume / mean_volume)) + 2)
    for _ in range(8):
        if max_particles is not None:
            pool_size = min(pool_size, max_particles + 3)
        pool, _ = sample_radii(config, pool_size, rng)
        counts = np.arange(1, pool_size + 1)
        # Existing distributions normalize the sample mean diameter to one.
        # Evaluate that normalization for every prefix without repeated sampling.
        scales = 0.5 * counts / np.cumsum(pool)
        volumes = (4.0 * np.pi / 3.0) * np.cumsum(pool**3) * scales**3
        best = int(np.argmin(np.abs(volumes - target_volume)))
        if best < pool_size - 2:
            if max_particles is not None and best + 1 > max_particles:
                raise ValueError(f"Resolved particle count exceeds {max_particles:,}.")
            return pool[:best + 1] * scales[best]
        if max_particles is not None and pool_size == max_particles + 3:
            raise ValueError(f"This target requires more than {max_particles:,} particles for the sampled distribution.")
        pool_size *= 2
    raise ValueError("Could not resolve a particle count for this size distribution and box.")


def resolve_parameters(config: PackingConfig, max_particles=None) -> ResolvedParameters:
    """Resolve N, radii, and box length from a user config."""
    seed = config.runtime.random_seed
    rng = np.random.default_rng(seed)
    if not np.isfinite(config.physical.packing_fraction) or config.physical.packing_fraction <= 0:
        raise ValueError("packing_fraction must be positive and finite")
    if not np.isfinite(config.physical.mean_diameter_m) or config.physical.mean_diameter_m <= 0:
        raise ValueError("mean_diameter must be positive and finite")
    if config.domain.fixed_dimensions:
        if config.domain.length_m is None or config.domain.depth_m is None:
            raise ValueError("Specify both domain.length and domain.depth")
        lengths = as_box_lengths([config.domain.length_m, config.domain.length_m, config.domain.depth_m]) / config.physical.mean_diameter_m
        radii = _fixed_box_radii(config, lengths, rng, max_particles)
        volumes = 4.0 * np.pi * radii**3 / 3.0
        phi = float(np.sum(volumes) / np.prod(lengths))
        count_tolerance = float(0.5 * np.max(volumes) / np.prod(lengths))
        if config.structure.spatial_order == SpatialOrderClass.PERIODIC_CRYSTAL:
            count_tolerance = 0.0
        if abs(phi - config.physical.packing_fraction) > count_tolerance + config.validation.tolerance_phi:
            raise ValueError("Particle count, size distribution, fixed dimensions and target density are inconsistent. Use automatic particle count, adjust the target density or dimensions; crystal cells must also match the density.")
        if config.structure.spatial_order != SpatialOrderClass.OVERLAPPING_RANDOM and 2 * np.max(radii) > np.min(lengths) + config.validation.tolerance_overlap:
            raise ValueError("Largest particle overlaps its own periodic image: increase the shortest box dimension.")
        box = float(lengths[0]) if np.all(lengths == lengths[0]) else lengths
        return ResolvedParameters(
            n_particles=len(radii), box_length=box, target_phi=float(config.physical.packing_fraction),
            mean_diameter_m=config.physical.mean_diameter_m, target_medium_thickness_m=config.domain.depth_m,
            box_length_m=box * config.physical.mean_diameter_m,
            output_unit=config.output.coordinate_unit, output_unit_m=unit_scale_to_meters(config.output.coordinate_unit),
            radii=radii, radius_mean=float(np.mean(radii)), radius_cv=float(np.std(radii) / np.mean(radii)),
            fixed_dimensions=True, resolved_phi=phi, density_resolution_tolerance=count_tolerance,
        )
    n_particles = config.particles.num_particles or _crystal_n_from_unit_cells(config) or _auto_particle_count(config, rng)
    if n_particles is None:
        raise ValueError(
            "Specify physical.medium_thickness, particles.num_particles, "
            "algorithm.parameters.unit_cells, or domain.box_size."
        )

    radii, _ = sample_radii(config, int(n_particles), rng)
    target_phi = float(config.physical.packing_fraction)
    particle_volume = float(np.sum(4.0 * np.pi * radii**3 / 3.0))
    # After N is integer-valued, recompute L from the sampled radii and target fraction.
    # For overlapping_random this target is the internally computed nominal fraction eta.
    # For hard-core structures it is the requested packing fraction.
    box_length = (particle_volume / target_phi) ** (1.0 / 3.0)

    radius_mean = float(np.mean(radii))
    radius_cv = float(np.std(radii) / radius_mean) if radius_mean else 0.0
    return ResolvedParameters(
        n_particles=int(n_particles),
        box_length=float(box_length),
        target_phi=target_phi,
        mean_diameter_m=float(config.physical.mean_diameter_m),
        target_medium_thickness_m=config.physical.medium_thickness_m,
        box_length_m=float(box_length * config.physical.mean_diameter_m),
        output_unit=config.output.coordinate_unit,
        output_unit_m=unit_scale_to_meters(config.output.coordinate_unit),
        radii=radii,
        radius_mean=radius_mean,
        radius_cv=radius_cv,
    )
