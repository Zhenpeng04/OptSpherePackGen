"""Periodic crystal generators."""

from __future__ import annotations

import time

import numpy as np

from spherepackgen.domain.enums import LatticeType
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.result import PackingResult
from spherepackgen.generators.base import GeneratorContext
from spherepackgen.utils.exceptions import GenerationError


class CrystalGenerator:
    name = "crystal"

    BASIS = {
        LatticeType.SC: np.array([[0.0, 0.0, 0.0]], dtype=float),
        LatticeType.BCC: np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float),
        LatticeType.FCC: np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
            dtype=float,
        ),
        LatticeType.HCP: np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 1.0 / 6.0, 0.5], [0.0, 2.0 / 3.0, 0.5]],
            dtype=float,
        ),
        LatticeType.DIAMOND_CUBIC: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ],
            dtype=float,
        ),
    }

    def generate(self, context: GeneratorContext) -> PackingResult:
        start = time.perf_counter()
        params = context.config.algorithm.parameters
        lattice = LatticeType.from_value(params.get("lattice_type", "FCC"))
        basis = self.BASIS[lattice]
        if context.resolved.fixed_dimensions:
            # Use a single lattice spacing on every axis: no affine distortion.
            spacing = (context.domain.volume * len(basis) / context.resolved.n_particles) ** (1.0 / 3.0)
            cells = np.rint(context.domain.box_lengths / spacing).astype(int)
            if np.any(cells < 1) or not np.allclose(cells * spacing, context.domain.box_lengths, rtol=1e-8, atol=1e-10):
                raise GenerationError("Fixed crystal dimensions must contain complete unstrained unit cells.")
            grid = np.indices(tuple(cells), dtype=float).reshape(3, -1).T
            positions = ((grid[:, None, :] + basis[None, :, :]) * spacing).reshape(-1, 3)
            ps = ParticleSet(positions, np.full(len(positions), 0.5), context.domain.box_length)
            return PackingResult(
                particle_set=ps.wrapped(), generator_name=self.name, status="success",
                diagnostics={"lattice_type": lattice.value, "unit_cells": cells.tolist(),
                             "basis_count": len(basis), "generated_particle_count": len(positions),
                             "coordinate_backend": "vectorized_unit_cell_grid"},
                elapsed_time_s=time.perf_counter() - start,
            )
        unit_cells = int(params.get("unit_cells", round((context.resolved.n_particles / len(basis)) ** (1 / 3))))
        unit_cells = max(1, unit_cells)
        generated_count = int(len(basis) * unit_cells**3)
        if generated_count != context.resolved.n_particles:
            raise GenerationError(
                f"{lattice.value} crystal requires a complete unit-cell grid: "
                f"basis_count * unit_cells^3 = {generated_count}, but resolved n_particles is "
                f"{context.resolved.n_particles}. Use particles.num_particles={generated_count} "
                f"or set algorithm.parameters.unit_cells={unit_cells}."
            )
        grid = np.indices((unit_cells, unit_cells, unit_cells), dtype=float).reshape(3, -1).T
        positions = ((grid[:, None, :] + basis[None, :, :]) / unit_cells * context.domain.box_length).reshape(-1, 3)
        n = len(positions)
        radii = np.full(n, 0.5, dtype=float)
        ps = ParticleSet(positions=positions, radii=radii, box_length=context.domain.box_length)
        return PackingResult(
            particle_set=ps.wrapped(),
            generator_name=self.name,
            status="success",
            diagnostics={
                "lattice_type": lattice.value,
                "unit_cells": unit_cells,
                "basis_count": len(basis),
                "generated_particle_count": generated_count,
                "coordinate_backend": "vectorized_unit_cell_grid",
            },
            elapsed_time_s=time.perf_counter() - start,
        )

