"""Force-biased hard-sphere relaxation generator."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.domain import as_box_lengths, box_volume
from spherepackgen.domain.result import PackingResult
from spherepackgen.generators.base import GeneratorContext
from spherepackgen.generators.rsa import poisson_disk_place
from spherepackgen.utils.cell_list import PeriodicCellList, minimum_image_delta


NOMINAL_DENSITY_RATIO = 1.2
DEFAULT_OUTER_DIAMETER_RATIO = NOMINAL_DENSITY_RATIO ** (1.0 / 3.0)
HIGH_DENSITY_PHI_THRESHOLD = 0.60
HIGH_DENSITY_INITIAL_RADIUS_FRACTION = 0.65
HIGH_DENSITY_OUTER_DIAMETER_RATIO = 1.08
HIGH_DENSITY_RELAXATION_STEPS_PER_STAGE = 800
HIGH_DENSITY_FINAL_CLEANUP_STEPS = 2000


class _VerletPairCache:
    def __init__(
        self,
        radii: np.ndarray,
        box_length: float,
        outer_ratio_cutoff: float,
        skin: float,
    ):
        self.radii = np.asarray(radii, dtype=float)
        self.box_length = as_box_lengths(box_length)
        self.outer_ratio_cutoff = float(outer_ratio_cutoff)
        self.skin = max(float(skin), 1.0e-12)
        self.reference_positions: np.ndarray | None = None
        self.pair_i = np.empty(0, dtype=np.int64)
        self.pair_j = np.empty(0, dtype=np.int64)
        self.rebuild_count = 0
        self.max_displacement_since_rebuild = 0.0
        self.pair_search_backend = "unbuilt"

    def set_outer_ratio_cutoff(self, outer_ratio_cutoff: float):
        outer_ratio_cutoff = max(float(outer_ratio_cutoff), 1.0)
        if outer_ratio_cutoff > self.outer_ratio_cutoff + 1.0e-12:
            self.reference_positions = None
        self.outer_ratio_cutoff = outer_ratio_cutoff

    def ensure_current(self, positions: np.ndarray):
        if self.reference_positions is None:
            self._rebuild(positions)
            return
        deltas = positions - self.reference_positions
        deltas -= self.box_length * np.rint(deltas / self.box_length)
        max_displacement = float(np.max(np.linalg.norm(deltas, axis=1))) if len(deltas) else 0.0
        self.max_displacement_since_rebuild = max_displacement
        if max_displacement > 0.5 * self.skin:
            self._rebuild(positions)

    def _rebuild(self, positions: np.ndarray):
        n = len(self.radii)
        pair_i = np.empty(0, dtype=np.int64)
        pair_j = np.empty(0, dtype=np.int64)
        backend = "ckdtree"
        if n >= 2:
            max_radius = float(np.max(self.radii))
            max_cutoff = max(2.0 * self.outer_ratio_cutoff * max_radius + self.skin, 1.0e-12)
            try:
                tree = cKDTree(np.mod(positions, self.box_length), boxsize=self.box_length)
                pairs = tree.query_pairs(max_cutoff, output_type="ndarray")
            except Exception:
                backend = "cell_list"
                pairs = np.empty((0, 2), dtype=np.int64)

            if backend == "ckdtree":
                if len(pairs):
                    pairs = np.asarray(pairs, dtype=np.int64)
                    deltas = positions[pairs[:, 0]] - positions[pairs[:, 1]]
                    deltas -= self.box_length * np.rint(deltas / self.box_length)
                    distances_squared = np.einsum("ij,ij->i", deltas, deltas)
                    cutoffs = (
                        self.outer_ratio_cutoff * (self.radii[pairs[:, 0]] + self.radii[pairs[:, 1]])
                        + self.skin
                    )
                    pairs = pairs[distances_squared <= cutoffs * cutoffs]
                    if len(pairs):
                        pair_i = pairs[:, 0].astype(np.int64, copy=True)
                        pair_j = pairs[:, 1].astype(np.int64, copy=True)
            else:
                pair_i_list: list[int] = []
                pair_j_list: list[int] = []
                cell_list = PeriodicCellList(self.box_length, cutoff=max_cutoff)
                cell_list.build(positions)
                for i in range(n):
                    search_radius = self.outer_ratio_cutoff * (self.radii[i] + max_radius) + self.skin
                    for j in cell_list.nearby_indices(positions[i], search_radius):
                        if j <= i:
                            continue
                        delta = minimum_image_delta(positions[i], positions[j], self.box_length)
                        distance_squared = float(np.dot(delta, delta))
                        cutoff = self.outer_ratio_cutoff * (self.radii[i] + self.radii[j]) + self.skin
                        if distance_squared <= cutoff * cutoff:
                            pair_i_list.append(i)
                            pair_j_list.append(j)
                pair_i = np.asarray(pair_i_list, dtype=np.int64)
                pair_j = np.asarray(pair_j_list, dtype=np.int64)

        self.pair_i = pair_i
        self.pair_j = pair_j
        self.pair_search_backend = backend
        self.reference_positions = positions.copy()
        self.rebuild_count += 1
        self.max_displacement_since_rebuild = 0.0


def _accumulate_pair_forces(n_particles: int, active_i: np.ndarray, active_j: np.ndarray, pair_forces: np.ndarray) -> np.ndarray:
    forces = np.empty((int(n_particles), 3), dtype=float)
    for axis in range(3):
        plus = np.bincount(active_i, weights=pair_forces[:, axis], minlength=n_particles)
        minus = np.bincount(active_j, weights=pair_forces[:, axis], minlength=n_particles)
        forces[:, axis] = plus - minus
    return forces


def _outer_shell_forces(
    positions: np.ndarray,
    radii: np.ndarray,
    box_length: float,
    outer_ratio: float,
    rng: np.random.Generator,
    pair_cache: _VerletPairCache | None = None,
):
    n = len(radii)
    forces = np.zeros_like(positions)
    if n < 2:
        return forces, {
            "max_shell_overlap": 0.0,
            "shell_overlap_count": 0,
            "shell_overlap_sum": 0.0,
            "max_overlap": 0.0,
            "overlap_count": 0,
            "overlap_sum": 0.0,
            "inner_diameter_ratio": float("inf"),
            "candidate_pair_count": 0,
        }

    if pair_cache is None:
        pair_cache = _VerletPairCache(radii, box_length, outer_ratio, 1.0e-12)
    else:
        pair_cache.set_outer_ratio_cutoff(outer_ratio)
    pair_cache.ensure_current(positions)
    pair_i = pair_cache.pair_i
    pair_j = pair_cache.pair_j
    candidate_pair_count = int(len(pair_i))
    if candidate_pair_count == 0:
        return forces, {
            "max_shell_overlap": 0.0,
            "shell_overlap_count": 0,
            "shell_overlap_sum": 0.0,
            "max_overlap": 0.0,
            "overlap_count": 0,
            "overlap_sum": 0.0,
            "inner_diameter_ratio": float("inf"),
            "candidate_pair_count": 0,
        }

    deltas = positions[pair_i] - positions[pair_j]
    deltas -= box_length * np.rint(deltas / box_length)
    distances = np.linalg.norm(deltas, axis=1)
    true_contacts = radii[pair_i] + radii[pair_j]
    valid = true_contacts > 0.0
    if not np.any(valid):
        return forces, {
            "max_shell_overlap": 0.0,
            "shell_overlap_count": 0,
            "shell_overlap_sum": 0.0,
            "max_overlap": 0.0,
            "overlap_count": 0,
            "overlap_sum": 0.0,
            "inner_diameter_ratio": float("inf"),
            "candidate_pair_count": candidate_pair_count,
        }

    inner_ratio = float(np.min(distances[valid] / true_contacts[valid]))
    shell_overlaps = outer_ratio * true_contacts - distances
    shell_active = shell_overlaps > 0.0
    shell_overlap_count = int(np.count_nonzero(shell_active))
    shell_overlap_sum = float(np.sum(shell_overlaps[shell_active])) if shell_overlap_count else 0.0
    max_shell_overlap = float(np.max(shell_overlaps[shell_active])) if shell_overlap_count else 0.0

    true_overlaps = true_contacts - distances
    overlap_active = true_overlaps > 0.0
    overlap_count = int(np.count_nonzero(overlap_active))
    overlap_sum = float(np.sum(true_overlaps[overlap_active])) if overlap_count else 0.0
    max_overlap = float(np.max(true_overlaps[overlap_active])) if overlap_count else 0.0

    if shell_overlap_count:
        active_i = pair_i[shell_active]
        active_j = pair_j[shell_active]
        active_deltas = deltas[shell_active]
        active_distances = distances[shell_active]
        directions = np.zeros_like(active_deltas)
        nonzero = active_distances > 1.0e-12
        directions[nonzero] = active_deltas[nonzero] / active_distances[nonzero, None]
        if np.any(~nonzero):
            random_directions = rng.normal(size=(int(np.count_nonzero(~nonzero)), 3))
            norms = np.linalg.norm(random_directions, axis=1)
            bad = norms <= 1.0e-12
            if np.any(bad):
                random_directions[bad] = np.array([1.0, 0.0, 0.0])
                norms[bad] = 1.0
            directions[~nonzero] = random_directions / norms[:, None]
        pair_forces = shell_overlaps[shell_active, None] * directions
        forces = _accumulate_pair_forces(n, active_i, active_j, pair_forces)

    return forces, {
        "max_shell_overlap": float(max_shell_overlap),
        "shell_overlap_count": int(shell_overlap_count),
        "shell_overlap_sum": float(shell_overlap_sum),
        "max_overlap": float(max_overlap),
        "overlap_count": int(overlap_count),
        "overlap_sum": float(overlap_sum),
        "inner_diameter_ratio": float(inner_ratio),
        "candidate_pair_count": candidate_pair_count,
    }


def _packing_fraction(radii: np.ndarray, box_length: float) -> float:
    return float(np.sum(4.0 * np.pi * radii**3 / 3.0) / box_volume(box_length))


def _updated_outer_ratio(
    *,
    outer_ratio: float,
    initial_outer_ratio: float,
    inner_ratio: float,
    packing_fraction: float,
    contraction_rate: float,
) -> float:
    if not np.isfinite(inner_ratio):
        return 1.0

    inner_porosity = 1.0 - packing_fraction * inner_ratio**3
    nominal_porosity = 1.0 - packing_fraction * NOMINAL_DENSITY_RATIO
    packing_fraction_difference = inner_porosity - nominal_porosity
    if packing_fraction_difference <= 0.0:
        return outer_ratio

    j = float(np.ceil(-np.log10(max(packing_fraction_difference, 1.0e-300))))
    decrement = (0.5**j) * initial_outer_ratio * float(contraction_rate)
    return max(1.0, float(outer_ratio) - decrement)


def _force_biased_relax(
    positions: np.ndarray,
    radii: np.ndarray,
    box_length: float,
    rng: np.random.Generator,
    *,
    max_iterations: int,
    initial_outer_ratio: float,
    contraction_rate: float,
    force_scaling_factor: float,
    max_displacement_fraction: float,
    tolerance_overlap: float,
    verlet_skin: float,
):
    outer_ratio = float(initial_outer_ratio)
    active_phi = _packing_fraction(radii, box_length)
    last_diag = {
        "max_shell_overlap": 0.0,
        "shell_overlap_count": 0,
        "shell_overlap_sum": 0.0,
        "max_overlap": 0.0,
        "overlap_count": 0,
        "overlap_sum": 0.0,
        "inner_diameter_ratio": float("inf"),
    }
    iterations = 0
    stop_reason = "max_iterations"
    diameters = np.maximum(2.0 * radii, 1.0e-12)
    max_displacement = np.maximum(float(max_displacement_fraction) * radii, 1.0e-12)
    pair_cache = _VerletPairCache(radii, box_length, outer_ratio, verlet_skin)
    forces, last_diag = _outer_shell_forces(positions, radii, box_length, outer_ratio, rng, pair_cache)

    for _ in range(max_iterations):
        inner_ratio = float(last_diag["inner_diameter_ratio"])
        if last_diag["shell_overlap_count"] == 0:
            stop_reason = "no_outer_shell_contacts"
            break
        if outer_ratio <= inner_ratio and last_diag["max_overlap"] <= tolerance_overlap:
            stop_reason = "outer_ratio_reached_inner_ratio"
            break
        if outer_ratio <= 1.0 + 1.0e-12 and last_diag["max_overlap"] <= tolerance_overlap:
            stop_reason = "true_non_overlap_reached"
            break

        displacement = (
            float(force_scaling_factor)
            * outer_ratio**2
            * forces
            / (2.0 * diameters[:, None])
        )
        norms = np.linalg.norm(displacement, axis=1)
        too_large = norms > max_displacement
        if np.any(too_large):
            displacement[too_large] *= (max_displacement[too_large] / norms[too_large])[:, None]

        positions = np.mod(positions + displacement, box_length)
        iterations += 1

        forces, last_diag = _outer_shell_forces(positions, radii, box_length, outer_ratio, rng, pair_cache)
        outer_ratio = _updated_outer_ratio(
            outer_ratio=outer_ratio,
            initial_outer_ratio=initial_outer_ratio,
            inner_ratio=float(last_diag["inner_diameter_ratio"]),
            packing_fraction=active_phi,
            contraction_rate=contraction_rate,
        )

    return positions, {
        "iterations_used": iterations,
        "max_iterations": int(max_iterations),
        "outer_diameter_ratio_final": float(outer_ratio),
        "packing_fraction_active": active_phi,
        "stop_reason": stop_reason,
        "verlet_skin": float(verlet_skin),
        "pair_cache_rebuilds": int(pair_cache.rebuild_count),
        "pair_search_backend": pair_cache.pair_search_backend,
        "pair_cutoff_outer_ratio_final": float(pair_cache.outer_ratio_cutoff),
        **last_diag,
    }


class ForceBiasedGenerator:
    name = "force_biased"

    def generate(self, context: GeneratorContext) -> PackingResult:
        start = time.perf_counter()
        params = context.config.algorithm.parameters
        target_radii = context.resolved.radii.copy()
        target_phi = float(context.resolved.target_phi)
        high_density_defaults = target_phi >= HIGH_DENSITY_PHI_THRESHOLD
        default_initial_fraction = HIGH_DENSITY_INITIAL_RADIUS_FRACTION if high_density_defaults else 0.35
        default_outer_ratio = HIGH_DENSITY_OUTER_DIAMETER_RATIO if high_density_defaults else DEFAULT_OUTER_DIAMETER_RATIO
        default_max_iterations = HIGH_DENSITY_RELAXATION_STEPS_PER_STAGE if high_density_defaults else 400
        default_final_cleanup_steps = HIGH_DENSITY_FINAL_CLEANUP_STEPS if high_density_defaults else default_max_iterations

        initial_fraction = float(params.get("initial_radius_fraction", default_initial_fraction))
        stages = int(params.get("stages", 24))
        max_iterations = int(
            params.get(
                "max_iterations_per_stage",
                params.get("relaxation_steps_per_stage", default_max_iterations),
            )
        )
        force_scaling_factor = float(
            params.get("force_scaling_factor", params.get("force_step_scale", params.get("step_scale", 0.5)))
        )
        outer_ratio_initial = float(
            params.get("outer_diameter_ratio", params.get("outer_diameter_ratio_initial", default_outer_ratio))
        )
        contraction_rate = float(params.get("contraction_rate", 1.0e-3))
        final_cleanup_steps = int(params.get("final_cleanup_steps", default_final_cleanup_steps))
        max_displacement_fraction = float(params.get("max_displacement_fraction", 0.35))
        max_attempts = int(params.get("max_attempts_per_particle", 30000))
        candidates = int(params.get("candidates_per_particle", 30))
        tolerance_overlap = float(context.config.validation.tolerance_overlap)
        intermediate_tolerance_overlap = float(
            params.get("intermediate_tolerance_overlap", max(tolerance_overlap, 1.0e-6))
        )
        mean_target_diameter = max(2.0 * float(np.mean(target_radii)), 1.0e-12)
        verlet_skin = float(
            params.get("verlet_skin", params.get("verlet_skin_fraction", 0.5) * mean_target_diameter)
        )

        if not 0.0 < initial_fraction <= 1.0:
            raise ValueError("force_biased initial_radius_fraction must be in (0, 1].")
        if stages < 1:
            raise ValueError("force_biased stages must be >= 1.")
        if outer_ratio_initial < 1.0:
            raise ValueError("force_biased outer_diameter_ratio must be >= 1.")
        if contraction_rate <= 0.0:
            raise ValueError("force_biased contraction_rate must be > 0.")
        if intermediate_tolerance_overlap < tolerance_overlap:
            intermediate_tolerance_overlap = tolerance_overlap
        if verlet_skin <= 0.0:
            raise ValueError("force_biased verlet_skin or verlet_skin_fraction must be > 0.")

        start_radii = target_radii * initial_fraction
        positions, init_diag = poisson_disk_place(
            start_radii,
            context.domain.box_length,
            context.rng,
            max_attempts_per_particle=max_attempts,
            candidates_per_particle=candidates,
        )

        stage_diagnostics = []
        max_overlap = 0.0
        overlap_count = 0
        for stage in range(1, stages + 1):
            frac = stage / stages
            radii = start_radii + frac * (target_radii - start_radii)
            positions, diag = _force_biased_relax(
                positions,
                radii,
                context.domain.box_length,
                context.rng,
                max_iterations=max_iterations,
                initial_outer_ratio=outer_ratio_initial,
                contraction_rate=contraction_rate,
                force_scaling_factor=force_scaling_factor,
                max_displacement_fraction=max_displacement_fraction,
                tolerance_overlap=intermediate_tolerance_overlap,
                verlet_skin=verlet_skin,
            )
            max_overlap = float(diag["max_overlap"])
            overlap_count = int(diag["overlap_count"])
            stage_diagnostics.append(
                {
                    "stage": stage,
                    "radius_fraction": float(initial_fraction + frac * (1.0 - initial_fraction)),
                    **diag,
                }
            )

        positions, final_diag = _force_biased_relax(
            positions,
            target_radii,
            context.domain.box_length,
            context.rng,
            max_iterations=final_cleanup_steps,
            initial_outer_ratio=1.0,
            contraction_rate=contraction_rate,
            force_scaling_factor=force_scaling_factor,
            max_displacement_fraction=max_displacement_fraction,
            tolerance_overlap=tolerance_overlap,
            verlet_skin=verlet_skin,
        )
        max_overlap = float(final_diag["max_overlap"])
        overlap_count = int(final_diag["overlap_count"])

        ps = ParticleSet(
            positions=positions,
            radii=target_radii,
            box_length=context.domain.box_length,
        )
        status = "success" if max_overlap <= tolerance_overlap else "warning"
        return PackingResult(
            particle_set=ps.wrapped(),
            generator_name=self.name,
            status=status,
            diagnostics={
                "note": (
                    "P0 force-biased dense generator: variable-radius Poisson-disk initialization, "
                    "staged radius growth, outer-shell repulsive forces, and outer diameter ratio contraction. "
                    "Not a rigorous LS or RCP proof."
                ),
                "reference_model": "simplified Bezrukov/Jodrey-Tory-style force-biased relaxation",
                "initialization": "poisson_disk_small_radii",
                "initial_radius_fraction": initial_fraction,
                "high_density_defaults": bool(high_density_defaults),
                "high_density_phi_threshold": HIGH_DENSITY_PHI_THRESHOLD,
                "initialization_diagnostics": init_diag,
                "stages": stages,
                "max_iterations_per_stage": max_iterations,
                "force_scaling_factor": force_scaling_factor,
                "outer_diameter_ratio_initial": outer_ratio_initial,
                "nominal_density_ratio": NOMINAL_DENSITY_RATIO,
                "contraction_rate": contraction_rate,
                "max_displacement_fraction": max_displacement_fraction,
                "intermediate_tolerance_overlap": intermediate_tolerance_overlap,
                "final_tolerance_overlap": tolerance_overlap,
                "verlet_skin": verlet_skin,
                "force_backend": "vectorized_verlet_pair_cache",
                "final_max_overlap_estimate": max_overlap,
                "final_overlap_count_estimate": overlap_count,
                "final_cleanup": final_diag,
                "stage_diagnostics": stage_diagnostics,
            },
            elapsed_time_s=time.perf_counter() - start,
        )
