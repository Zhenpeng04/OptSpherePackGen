"""Random sequential adsorption generator."""

from __future__ import annotations

import time

import numpy as np

from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.result import PackingResult
from spherepackgen.generators.base import GeneratorContext
from spherepackgen.utils.cell_list import PeriodicCellList, pair_gap
from spherepackgen.utils.exceptions import GenerationError


POISSON_DISK_BATCH_DISTANCE_LIMIT = 12000


def rsa_place(
    radii: np.ndarray,
    box_length: float,
    rng: np.random.Generator,
    max_attempts_per_particle: int = 10000,
):
    """Place particles by RSA, sorting larger particles first for polydisperse cases."""
    radii = np.asarray(radii, dtype=float)
    order = np.argsort(-radii)
    sorted_radii = radii[order]
    positions_sorted = np.empty((len(radii), 3), dtype=float)
    max_radius = float(np.max(radii))
    cell_list = PeriodicCellList(box_length, cutoff=max(2.0 * max_radius, 1.0e-12))
    accepted = 0
    total_attempts = 0
    for local_idx, radius in enumerate(sorted_radii):
        placed = False
        for _ in range(max_attempts_per_particle):
            total_attempts += 1
            candidate = rng.random(3) * box_length
            ok = True
            for j in cell_list.nearby_indices(candidate, radius + max_radius):
                if pair_gap(candidate, radius, positions_sorted[j], sorted_radii[j], box_length) < 0.0:
                    ok = False
                    break
            if ok:
                positions_sorted[local_idx] = candidate
                cell_list.add(local_idx, candidate)
                accepted += 1
                placed = True
                break
        if not placed:
            raise GenerationError(
                f"RSA failed after {max_attempts_per_particle} attempts for particle {local_idx}; "
                "try a lower packing fraction, smaller N, or a dense generator."
            )
    positions = np.empty_like(positions_sorted)
    positions[order] = positions_sorted
    return positions, {"total_attempts": total_attempts, "accepted": accepted}


def _candidate_min_gap(
    candidate: np.ndarray,
    radius: float,
    positions: np.ndarray,
    radii: np.ndarray,
    accepted: int,
    box_length: float,
) -> float:
    if accepted == 0:
        return float("inf")
    deltas = candidate - positions[:accepted]
    deltas = deltas - box_length * np.rint(deltas / box_length)
    distances = np.linalg.norm(deltas, axis=1)
    gaps = distances - radius - radii[:accepted]
    return float(np.min(gaps))


def _candidate_min_gaps(
    candidates: np.ndarray,
    radius: float,
    positions: np.ndarray,
    radii: np.ndarray,
    accepted: int,
    box_length: float,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=float)
    if accepted == 0:
        return np.full(len(candidates), float("inf"), dtype=float)
    if accepted * len(candidates) > POISSON_DISK_BATCH_DISTANCE_LIMIT:
        return np.asarray(
            [
                _candidate_min_gap(candidate, radius, positions, radii, accepted, box_length)
                for candidate in candidates
            ],
            dtype=float,
        )
    deltas = candidates[:, None, :] - positions[None, :accepted, :]
    deltas -= box_length * np.rint(deltas / box_length)
    distances = np.linalg.norm(deltas, axis=2)
    gaps = distances - float(radius) - radii[None, :accepted]
    return np.min(gaps, axis=1)


def _nearby_indices_fast(cell_list: PeriodicCellList, position: np.ndarray, search_radius: float) -> list[int]:
    return list(cell_list.nearby_indices(position, search_radius))



def _candidate_min_gaps_cell_list(
    candidates: np.ndarray,
    radius: float,
    positions: np.ndarray,
    radii: np.ndarray,
    accepted: int,
    box_length: float,
    cell_list: PeriodicCellList,
    search_radius: float,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=float)
    if accepted == 0:
        return np.full(len(candidates), float("inf"), dtype=float)
    min_gaps = np.full(len(candidates), float("inf"), dtype=float)
    for idx, candidate in enumerate(candidates):
        neighbor_indices = _nearby_indices_fast(cell_list, candidate, search_radius)
        if not neighbor_indices:
            continue
        neighbors = np.asarray(neighbor_indices, dtype=int)
        deltas = candidate - positions[neighbors]
        deltas -= box_length * np.rint(deltas / box_length)
        distances = np.linalg.norm(deltas, axis=1)
        gaps = distances - float(radius) - radii[neighbors]
        min_gaps[idx] = float(np.min(gaps))
    return min_gaps


def poisson_disk_place(
    radii: np.ndarray,
    box_length: float,
    rng: np.random.Generator,
    max_attempts_per_particle: int = 10000,
    candidates_per_particle: int = 8,
    shell_outer_factor: float = 1.0,
):
    """Place variable-radius hard spheres with cell-list best-candidate sampling."""
    radii = np.asarray(radii, dtype=float)
    order = np.argsort(-radii)
    sorted_radii = radii[order]
    positions_sorted = np.empty((len(radii), 3), dtype=float)
    total_attempts = 0
    global_candidate_attempts = 0
    valid_candidate_count = 0
    accepted = 0
    max_radius = float(np.max(radii))
    cell_list = PeriodicCellList(box_length, cutoff=max(2.0 * max_radius, 1.0e-12))
    candidate_search_factor = max(1.0, float(shell_outer_factor))

    for local_idx, radius in enumerate(sorted_radii):
        if accepted == 0:
            positions_sorted[local_idx] = rng.random(3) * box_length
            cell_list.add(local_idx, positions_sorted[local_idx])
            accepted += 1
            continue

        best_candidate = None
        best_gap = -np.inf
        attempts_for_particle = 0
        batch_size = max(1, int(candidates_per_particle))
        while attempts_for_particle < max_attempts_per_particle:
            current_batch = min(batch_size, max_attempts_per_particle - attempts_for_particle)
            candidates = rng.random((current_batch, 3)) * box_length
            attempts_for_particle += current_batch
            total_attempts += current_batch
            global_candidate_attempts += current_batch
            overlap_search_radius = float(radius + max_radius)
            candidate_search_radius = candidate_search_factor * overlap_search_radius
            gaps = _candidate_min_gaps_cell_list(
                candidates,
                radius,
                positions_sorted,
                sorted_radii,
                accepted,
                box_length,
                cell_list,
                candidate_search_radius,
            )
            valid = gaps >= 0.0
            valid_candidate_count += int(np.count_nonzero(valid))
            if np.any(valid):
                candidate_gaps = np.where(valid, gaps, -np.inf)
                batch_best_index = int(np.argmax(candidate_gaps))
                if float(candidate_gaps[batch_best_index]) > best_gap:
                    best_gap = float(candidate_gaps[batch_best_index])
                    best_candidate = candidates[batch_best_index].copy()
            if best_candidate is not None:
                break

        if best_candidate is None:
            raise GenerationError(
                f"Variable-radius Poisson-disk placement failed after {max_attempts_per_particle} "
                f"attempts for particle {local_idx}; try RSA, a lower packing fraction, or a dense generator."
            )
        positions_sorted[local_idx] = best_candidate
        cell_list.add(local_idx, best_candidate)
        accepted += 1

    positions = np.empty_like(positions_sorted)
    positions[order] = positions_sorted
    return positions, {
        "total_attempts": total_attempts,
        "global_candidate_attempts": global_candidate_attempts,
        "valid_candidate_count": valid_candidate_count,
        "accepted": accepted,
        "candidates_per_particle": int(candidates_per_particle),
        "candidate_strategy": "cell_list_best_candidate",
        "candidate_gap_backend": "cell_list_local_exact_overlap",
        "candidate_search_factor": float(candidate_search_factor),
    }


class RSAGenerator:
    name = "rsa"

    def generate(self, context: GeneratorContext) -> PackingResult:
        start = time.perf_counter()
        max_attempts = int(context.config.algorithm.parameters.get("max_attempts_per_particle", 10000))
        positions, diagnostics = rsa_place(
            context.resolved.radii,
            context.domain.box_length,
            context.rng,
            max_attempts_per_particle=max_attempts,
        )
        ps = ParticleSet(
            positions=positions,
            radii=context.resolved.radii.copy(),
            box_length=context.domain.box_length,
        )
        diagnostics["max_attempts_per_particle"] = max_attempts
        return PackingResult(
            particle_set=ps.wrapped(),
            generator_name=self.name,
            status="success",
            diagnostics=diagnostics,
            elapsed_time_s=time.perf_counter() - start,
        )


class VariableRadiusPoissonDiskGenerator:
    name = "poisson_disk"

    def generate(self, context: GeneratorContext) -> PackingResult:
        start = time.perf_counter()
        params = context.config.algorithm.parameters
        max_attempts = int(params.get("max_attempts_per_particle", 10000))
        candidates = int(params.get("candidates_per_particle", params.get("poisson_disk_candidates", 8)))
        shell_outer_factor = float(params.get("shell_outer_factor", 1.0))
        positions, diagnostics = poisson_disk_place(
            context.resolved.radii,
            context.domain.box_length,
            context.rng,
            max_attempts_per_particle=max_attempts,
            candidates_per_particle=candidates,
            shell_outer_factor=shell_outer_factor,
        )
        ps = ParticleSet(
            positions=positions,
            radii=context.resolved.radii.copy(),
            box_length=context.domain.box_length,
        )
        diagnostics["max_attempts_per_particle"] = max_attempts
        diagnostics["note"] = (
            "Variable-radius Poisson-disk placement using cell-list best-candidate sampling; "
            "intended for low-volume-fraction hard-core random structures."
        )
        return PackingResult(
            particle_set=ps.wrapped(),
            generator_name=self.name,
            status="success",
            diagnostics=diagnostics,
            elapsed_time_s=time.perf_counter() - start,
        )
