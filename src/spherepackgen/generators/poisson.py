"""Marked Poisson Boolean model for overlapping spheres."""

from __future__ import annotations

import math
import time

from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.result import PackingResult
from spherepackgen.generators.base import GeneratorContext


class MarkedPoissonBooleanGenerator:
    """Generate independent sphere centers with independently sampled radius marks.

    The nominal volume fraction is the sum of sphere volumes divided by box
    volume. In a Poisson Boolean model this value may exceed one because
    overlaps are part of the model definition.
    """

    name = "marked_poisson_boolean"

    def generate(self, context: GeneratorContext) -> PackingResult:
        start = time.perf_counter()
        positions = context.rng.random((context.resolved.n_particles, 3)) * context.domain.box_length
        nominal_volume_fraction = context.resolved.resolved_phi if context.resolved.fixed_dimensions else context.resolved.target_phi
        expected_covered_fraction = 1.0 - math.exp(-nominal_volume_fraction)
        expected_uncovered_fraction = math.exp(-nominal_volume_fraction)
        target_covered_fraction = context.config.physical.covered_volume_fraction
        if target_covered_fraction is None:
            target_covered_fraction = expected_covered_fraction
        ps = ParticleSet(
            positions=positions,
            radii=context.resolved.radii.copy(),
            box_length=context.domain.box_length,
        )
        return PackingResult(
            particle_set=ps,
            generator_name=self.name,
            status="success",
            diagnostics={
                "model": "marked_poisson_boolean",
                "note": "Independent random centers with polydisperse radius marks; overlaps are allowed by definition.",
                "nominal_volume_fraction": float(nominal_volume_fraction),
                "target_covered_volume_fraction": float(target_covered_fraction),
                "expected_boolean_covered_fraction": float(expected_covered_fraction),
                "expected_boolean_uncovered_fraction": float(expected_uncovered_fraction),
            },
            elapsed_time_s=time.perf_counter() - start,
        )


PoissonGenerator = MarkedPoissonBooleanGenerator
