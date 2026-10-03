"""Local Metropolis hard-sphere relaxation."""

from __future__ import annotations

import time

import numpy as np

from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.result import PackingResult
from spherepackgen.generators.base import GeneratorContext
from spherepackgen.generators.rsa import rsa_place
from spherepackgen.utils.cell_list import pair_gap


def _move_is_valid(positions, radii, idx, candidate, box_length):
    radius = radii[idx]
    for j in range(len(radii)):
        if j == idx:
            continue
        if pair_gap(candidate, radius, positions[j], radii[j], box_length) < 0.0:
            return False
    return True


class MetropolisGenerator:
    name = "metropolis"

    def generate(self, context: GeneratorContext) -> PackingResult:
        start = time.perf_counter()
        params = context.config.algorithm.parameters
        sweeps = int(params.get("sweeps", 100))
        step_size = float(params.get("step_size", 0.10))
        max_attempts = int(params.get("max_attempts_per_particle", 10000))
        positions, init_diag = rsa_place(
            context.resolved.radii,
            context.domain.box_length,
            context.rng,
            max_attempts_per_particle=max_attempts,
        )
        radii = context.resolved.radii.copy()
        accepted = 0
        attempted = 0
        for _ in range(sweeps):
            for idx in context.rng.permutation(len(radii)):
                attempted += 1
                displacement = context.rng.normal(0.0, step_size, size=3)
                candidate = np.mod(positions[idx] + displacement, context.domain.box_length)
                if _move_is_valid(positions, radii, idx, candidate, context.domain.box_length):
                    positions[idx] = candidate
                    accepted += 1
        ps = ParticleSet(positions=positions, radii=radii, box_length=context.domain.box_length)
        diagnostics = {
            "initialization": "rsa",
            "rsa": init_diag,
            "sweeps": sweeps,
            "step_size": step_size,
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "acceptance_rate": accepted / attempted if attempted else 0.0,
        }
        return PackingResult(
            particle_set=ps.wrapped(),
            generator_name=self.name,
            status="success",
            diagnostics=diagnostics,
            elapsed_time_s=time.perf_counter() - start,
        )

