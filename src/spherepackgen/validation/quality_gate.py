"""Validation and quality-gate checks."""

from __future__ import annotations

import numpy as np

from spherepackgen.config.schema import PackingConfig
from spherepackgen.domain.enums import SpatialOrderClass
from spherepackgen.domain.result import ValidationReport
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.utils.cell_list import find_overlap_pairs


def validate_particle_set(particles: ParticleSet, config: PackingConfig, resolved=None) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    tol_overlap = float(config.validation.tolerance_overlap)
    tol_phi = float(config.validation.tolerance_phi)

    # ParticleSet is mutable. Do not rely only on its constructor checks or on
    # floating-point comparisons, which silently miss NaN values.
    try:
        particles.validate_geometry()
        config.validation.__post_init__()
        for name in ("mean_diameter_m", "packing_fraction"):
            value = float(getattr(config.physical, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"physical.{name} must be positive and finite")
    except (ValueError, TypeError) as exc:
        return ValidationReport(
            passed=False, errors=[str(exc)], warnings=[],
            metrics={"input_valid": False, "packing_fraction_actual": None},
            tolerances={"overlap": tol_overlap if np.isfinite(tol_overlap) else None,
                        "packing_fraction": tol_phi if np.isfinite(tol_phi) else None},
        )

    positions = particles.positions
    radii = particles.radii
    box = particles.box_lengths
    out_of_bounds = int(np.sum((positions < -1.0e-12) | (positions >= box + 1.0e-12)))
    if out_of_bounds:
        errors.append(f"{out_of_bounds} coordinate values are outside [0, L).")

    require_non_overlap = bool(config.validation.require_non_overlap)
    overlap_allowed_by_model = config.structure.spatial_order == SpatialOrderClass.OVERLAPPING_RANDOM
    overlap_check_skipped = bool(overlap_allowed_by_model)
    overlap_check_reason = "overlaps_allowed_by_marked_poisson_boolean_model" if overlap_check_skipped else ""
    self_image_overlap = max(0.0, float(2 * np.max(radii) - np.min(box)))
    if overlap_check_skipped:
        overlap_pairs = []
        min_gap = None
        max_overlap = None
        require_non_overlap = False
    else:
        overlap_pairs, min_gap = find_overlap_pairs(positions, radii, box, tolerance=tol_overlap)
        max_overlap = max(0.0, -float(min_gap)) if np.isfinite(min_gap) else 0.0
        min_gap = min(float(min_gap), float(np.min(box) - 2 * np.max(radii)))
        max_overlap = max(max_overlap, self_image_overlap)

    if require_non_overlap and not overlap_check_skipped and self_image_overlap > tol_overlap:
        errors.append("A particle overlaps its own periodic image along the shortest box dimension.")

    if require_non_overlap and overlap_pairs:
        errors.append(f"{len(overlap_pairs)} overlapping particle pairs exceed tolerance.")
    elif overlap_pairs:
        warnings.append(f"{len(overlap_pairs)} overlapping pairs detected; require_non_overlap=false.")

    phi_actual = particles.packing_fraction
    if not np.isfinite(phi_actual):
        errors.append("Generated particle volume fraction is not finite.")
    phi_target = float(config.physical.packing_fraction)
    phi_error = abs(phi_actual - phi_target)
    resolution_tolerance = 0.0
    if config.domain.fixed_dimensions:
        expected_box = np.asarray([config.domain.length_m, config.domain.length_m, config.domain.depth_m], dtype=float) / config.physical.mean_diameter_m
        if not np.allclose(box, expected_box, rtol=1e-12, atol=1e-12):
            errors.append("Generated box dimensions differ from the fixed user dimensions.")
        if resolved is not None:
            resolution_tolerance = resolved.density_resolution_tolerance
            if abs(phi_actual - resolved.resolved_phi) > tol_phi:
                errors.append("Generated particle volume differs from the resolved particle volume.")
            if particles.n_particles != resolved.n_particles:
                errors.append("Generated particle count differs from the resolved particle count.")
        else:
            resolution_tolerance = float(0.5 * np.max(4 * np.pi * radii**3 / 3) / np.prod(box))
    if phi_error > tol_phi + resolution_tolerance:
        errors.append(f"Packing fraction error {phi_error:.3e} exceeds tolerance {tol_phi + resolution_tolerance:.3e}.")
    elif phi_error > tol_phi and config.domain.fixed_dimensions:
        warnings.append("Actual density differs from target within the particle-count resolution allowance; box dimensions are fixed.")

    radius_mean = float(np.mean(radii))
    radius_cv = float(np.std(radii) / radius_mean) if radius_mean else 0.0
    metrics = {
        "input_valid": True,
        "n_particles": particles.n_particles,
        "box_length": float(box[0]) if np.all(box == box[0]) else box.tolist(),
        "box_lengths": box.tolist(),
        "box_volume": float(np.prod(box)),
        "density_resolution_tolerance": resolution_tolerance,
        "self_image_overlap": None if overlap_check_skipped else self_image_overlap,
        "packing_fraction_actual": float(phi_actual),
        "packing_fraction_target": phi_target,
        "packing_fraction_error": float(phi_error),
        "min_gap": None if min_gap is None else float(min_gap),
        "max_overlap": None if max_overlap is None else float(max_overlap),
        "overlap_pair_count": None if overlap_check_skipped else len(overlap_pairs),
        "overlap_check_skipped": overlap_check_skipped,
        "overlap_check_reason": overlap_check_reason,
        "out_of_bounds_value_count": out_of_bounds,
        "radius_min": float(np.min(radii)),
        "radius_max": float(np.max(radii)),
        "radius_mean": radius_mean,
        "radius_cv": radius_cv,
    }
    if config.structure.spatial_order == SpatialOrderClass.OVERLAPPING_RANDOM:
        covered_target = config.physical.covered_volume_fraction
        if covered_target is None:
            covered_target = float(1.0 - np.exp(-phi_target))
        covered_actual = float(1.0 - np.exp(-phi_actual))
        metrics.update(
            {
                "nominal_volume_fraction": float(phi_actual),
                "target_nominal_volume_fraction": float(phi_target),
                "target_covered_volume_fraction": float(covered_target),
                "expected_covered_volume_fraction": covered_actual,
                "expected_boolean_covered_fraction": float(1.0 - np.exp(-phi_actual)),
                "expected_boolean_uncovered_fraction": float(np.exp(-phi_actual)),
                "covered_volume_fraction_error": float(abs(covered_actual - covered_target)),
            }
        )
    return ValidationReport(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
        tolerances={"overlap": tol_overlap, "packing_fraction": tol_phi},
    )
