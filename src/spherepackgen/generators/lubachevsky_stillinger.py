"""Lubachevsky-Stillinger hard-sphere growth generator."""

from __future__ import annotations

import copy
import heapq
import time
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.domain import as_box_lengths, box_volume
from spherepackgen.domain.result import PackingResult
from spherepackgen.generators.base import GeneratorContext
from spherepackgen.generators.force_biased import ForceBiasedGenerator
from spherepackgen.generators.rsa import poisson_disk_place
from spherepackgen.utils.cell_list import PeriodicCellList, find_overlap_pairs, minimum_image_delta
from spherepackgen.utils.exceptions import GenerationError


AUTO_EXHAUSTIVE_PARTICLE_THRESHOLD = 2
DEFAULT_NEIGHBOR_REBUILD_FRACTION = 0.8
FORCE_BIASED_WARM_START_PHI_THRESHOLD = 0.60
DEFAULT_FORCE_BIASED_WARM_START_PHI = 0.50
LS_HIGH_DENSITY_VALIDATED_PHI_LIMIT = 0.62
_PERIODIC_IMAGE_OFFSETS = np.array(
    [(i, j, k) for i in (-1.0, 0.0, 1.0) for j in (-1.0, 0.0, 1.0) for k in (-1.0, 0.0, 1.0)],
    dtype=float,
)


def _packing_fraction(radii: np.ndarray, box_length: float) -> float:
    return float(np.sum(4.0 * np.pi * radii**3 / 3.0) / box_volume(box_length))


def _initial_radius_scale(params: dict, target_phi: float) -> float:
    if "initial_packing_fraction" in params:
        initial_phi = float(params["initial_packing_fraction"])
        if initial_phi <= 0.0:
            raise ValueError("initial_packing_fraction must be positive.")
        if initial_phi > target_phi:
            raise ValueError("initial_packing_fraction cannot exceed the target packing_fraction.")
        return float((initial_phi / target_phi) ** (1.0 / 3.0))
    return float(params.get("initial_radius_scale", 0.70))


def _select_initialization(params: dict, target_phi: float) -> tuple[str, str]:
    requested = str(params.get("initialization", params.get("initializer", "auto"))).strip().lower()
    aliases = {
        "auto": "auto",
        "poisson_disk": "poisson_disk_reduced_radii",
        "poisson_disk_reduced_radius": "poisson_disk_reduced_radii",
        "poisson_disk_reduced_radii": "poisson_disk_reduced_radii",
        "reduced_radii": "poisson_disk_reduced_radii",
        "reduced_radius": "poisson_disk_reduced_radii",
        "force_biased": "force_biased_warm_start",
        "force_biased_warm": "force_biased_warm_start",
        "force_biased_warm_start": "force_biased_warm_start",
        "warm_start": "force_biased_warm_start",
    }
    if requested not in aliases:
        raise ValueError(
            "initialization must be 'auto', 'poisson_disk', or 'force_biased_warm_start'."
        )

    normalized = aliases[requested]
    if normalized != "auto":
        return requested, normalized
    selected = (
        "force_biased_warm_start"
        if target_phi >= FORCE_BIASED_WARM_START_PHI_THRESHOLD
        else "poisson_disk_reduced_radii"
    )
    return requested, selected


def _first_parameter(params: dict, names: tuple[str, ...], default):
    for name in names:
        if name in params:
            return params[name]
    return default


def _warm_parameter(params: dict, name: str, default):
    return _first_parameter(
        params,
        (
            f"warm_start_{name}",
            f"force_biased_warm_start_{name}",
            f"warm_{name}",
        ),
        default,
    )


def _warm_start_packing_fraction(params: dict, target_phi: float) -> float:
    default_phi = min(DEFAULT_FORCE_BIASED_WARM_START_PHI, 0.95 * target_phi)
    warm_phi = float(
        _first_parameter(
            params,
            (
                "warm_start_packing_fraction",
                "force_biased_warm_start_phi",
                "warm_start_phi",
                "initial_packing_fraction",
            ),
            default_phi,
        )
    )
    if warm_phi <= 0.0:
        raise ValueError("warm_start_packing_fraction must be positive.")
    if warm_phi >= target_phi:
        raise ValueError("warm_start_packing_fraction must be lower than packing_fraction.")
    return warm_phi


def _force_biased_warm_start(
    context: GeneratorContext,
    target_radii: np.ndarray,
    target_phi: float,
    warm_phi: float,
    tolerance_overlap: float,
) -> tuple[np.ndarray, dict]:
    warm_scale = float((warm_phi / target_phi) ** (1.0 / 3.0))
    warm_radii = target_radii * warm_scale
    radius_mean = float(np.mean(warm_radii))
    warm_resolved = replace(
        context.resolved,
        target_phi=warm_phi,
        radii=warm_radii,
        radius_mean=radius_mean,
        radius_cv=float(np.std(warm_radii) / radius_mean) if radius_mean else 0.0,
    )
    warm_config = copy.deepcopy(context.config)
    params = context.config.algorithm.parameters
    warm_config.physical.packing_fraction = warm_phi
    warm_config.algorithm.name = "force_biased"
    warm_config.algorithm.parameters = {
        "initial_radius_fraction": float(_warm_parameter(params, "initial_radius_fraction", 0.65)),
        "stages": int(_warm_parameter(params, "stages", 16)),
        "relaxation_steps_per_stage": int(
            _first_parameter(
                params,
                (
                    "warm_start_relaxation_steps_per_stage",
                    "force_biased_warm_start_relaxation_steps_per_stage",
                    "warm_start_max_iterations_per_stage",
                    "force_biased_warm_start_max_iterations_per_stage",
                ),
                450,
            )
        ),
        "final_cleanup_steps": int(_warm_parameter(params, "final_cleanup_steps", 1200)),
        "outer_diameter_ratio": float(_warm_parameter(params, "outer_diameter_ratio", 1.08)),
        "contraction_rate": float(_warm_parameter(params, "contraction_rate", 1.0e-3)),
        "verlet_skin_fraction": float(_warm_parameter(params, "verlet_skin_fraction", 0.25)),
        "max_attempts_per_particle": int(
            _warm_parameter(params, "max_attempts_per_particle", params.get("max_attempts_per_particle", 30000))
        ),
        "candidates_per_particle": int(
            _warm_parameter(params, "candidates_per_particle", params.get("candidates_per_particle", 30))
        ),
    }
    optional_warm_parameters = (
        "force_scaling_factor",
        "force_step_scale",
        "step_scale",
        "max_displacement_fraction",
        "intermediate_tolerance_overlap",
        "verlet_skin",
    )
    for name in optional_warm_parameters:
        for candidate in (f"warm_start_{name}", f"force_biased_warm_start_{name}", f"warm_{name}"):
            if candidate in params:
                warm_config.algorithm.parameters[name] = params[candidate]
                break

    warm_start = time.perf_counter()
    warm_result = ForceBiasedGenerator().generate(
        GeneratorContext(warm_config, warm_resolved, context.domain, context.rng)
    )
    positions = warm_result.particle_set.positions.copy()
    overlap_pairs, min_gap = find_overlap_pairs(
        positions,
        warm_radii,
        context.domain.box_length,
        tolerance=tolerance_overlap,
    )
    max_overlap = max(0.0, -float(min_gap)) if np.isfinite(min_gap) else 0.0
    if overlap_pairs:
        raise GenerationError(
            "force_biased_warm_start produced overlapping reduced-radius spheres "
            f"(pairs={len(overlap_pairs)}, max_overlap={max_overlap:.6g}). "
            "Increase warm_start_final_cleanup_steps, lower warm_start_packing_fraction, "
            "or use poisson_disk initialization."
        )

    return positions, {
        "warm_start_generator": warm_result.generator_name,
        "warm_start_status": warm_result.status,
        "warm_start_packing_fraction": warm_phi,
        "warm_start_radius_scale": warm_scale,
        "warm_start_elapsed_time_s": time.perf_counter() - warm_start,
        "warm_start_overlap_pair_count": int(len(overlap_pairs)),
        "warm_start_min_gap": float(min_gap),
        "warm_start_max_overlap": float(max_overlap),
        "warm_start_parameters": dict(warm_config.algorithm.parameters),
        "warm_start_diagnostics": warm_result.diagnostics,
    }


def _random_velocities(
    n_particles: int,
    rng: np.random.Generator,
    velocity_scale: float,
) -> np.ndarray:
    velocities = rng.normal(size=(n_particles, 3))
    velocities -= np.mean(velocities, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(velocities**2, axis=1))))
    if rms <= 1.0e-14:
        velocities = rng.normal(size=(n_particles, 3))
        rms = float(np.sqrt(np.mean(np.sum(velocities**2, axis=1))))
    return velocities * (float(velocity_scale) / max(rms, 1.0e-14))


def _pair_collision_time(
    delta: np.ndarray,
    relative_velocity: np.ndarray,
    contact: float,
    contact_rate: float,
    event_epsilon: float,
) -> float | None:
    distance_squared = float(np.dot(delta, delta))
    c = distance_squared - contact * contact
    b_half = float(np.dot(delta, relative_velocity) - contact * contact_rate)
    if c <= 1.0e-12:
        return event_epsilon if b_half < 0.0 else None

    a = float(np.dot(relative_velocity, relative_velocity) - contact_rate * contact_rate)
    b = 2.0 * b_half
    if abs(a) <= 1.0e-14:
        if b >= 0.0:
            return None
        t = -c / b
        return float(t) if t > event_epsilon else None

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = float(np.sqrt(max(discriminant, 0.0)))
    roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    positive_roots = [float(value) for value in roots if value > event_epsilon and np.isfinite(value)]
    if not positive_roots:
        return None
    return min(positive_roots)


def _next_collision(
    positions: np.ndarray,
    velocities: np.ndarray,
    base_radii: np.ndarray,
    box_length: float,
    scale: float,
    growth_rate: float,
    event_epsilon: float,
    max_time: float | None = None,
):
    best_time = float("inf")
    best_pair: tuple[int, int] | None = None
    n_particles = len(base_radii)
    image_offsets = box_length * _PERIODIC_IMAGE_OFFSETS
    for i in range(n_particles - 1):
        for j in range(i + 1, n_particles):
            contact_base = float(base_radii[i] + base_radii[j])
            base_delta = positions[i] - positions[j]
            relative_velocity = velocities[i] - velocities[j]
            for image_offset in image_offsets:
                t_collision = _pair_collision_time(
                    base_delta + image_offset,
                    relative_velocity,
                    scale * contact_base,
                    growth_rate * contact_base,
                    event_epsilon,
                )
                if t_collision is None:
                    continue
                if max_time is not None and t_collision > max_time + event_epsilon:
                    continue
                if t_collision < best_time:
                    best_time = t_collision
                    best_pair = (i, j)
    if best_pair is None:
        return None, None
    return best_time, best_pair


def _periodic_recheck_interval(
    velocities: np.ndarray,
    base_radii: np.ndarray,
    box_length: float,
    growth_rate: float,
    event_epsilon: float,
) -> float:
    if len(base_radii) < 2:
        return float("inf")
    max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
    max_radius = float(np.max(base_radii))
    relative_speed_bound = 2.0 * max_speed + 2.0 * growth_rate * max_radius
    if relative_speed_bound <= 1.0e-12:
        return float("inf")
    return max(0.2 * float(np.min(as_box_lengths(box_length))) / relative_speed_bound, 10.0 * event_epsilon)


def _scale_at(initial_scale: float, growth_rate: float, current_time: float) -> float:
    return float(initial_scale + growth_rate * current_time)


def _position_at(
    position: np.ndarray,
    velocity: np.ndarray,
    last_update_time: float,
    current_time: float,
    box_length: float,
) -> np.ndarray:
    return np.mod(position + velocity * (current_time - last_update_time), box_length)


def _materialize_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    last_update_times: np.ndarray,
    current_time: float,
    box_length: float,
) -> None:
    positions[:] = np.mod(positions + velocities * (current_time - last_update_times)[:, None], box_length)
    last_update_times[:] = current_time


def _build_verlet_pairs(
    positions: np.ndarray,
    radii: np.ndarray,
    box_length: float,
    radius_scale: float,
    skin: float,
):
    n_particles = len(radii)
    adjacency: list[list[int]] = [[] for _ in range(n_particles)]
    pair_i = np.empty(0, dtype=np.int64)
    pair_j = np.empty(0, dtype=np.int64)
    if n_particles < 2:
        return adjacency, pair_i, pair_j

    max_radius = float(np.max(radii))
    max_cutoff = max(2.0 * radius_scale * max_radius + skin, 1.0e-12)
    used_tree = True
    try:
        tree = cKDTree(np.mod(positions, box_length), boxsize=box_length)
        pairs = tree.query_pairs(max_cutoff, output_type="ndarray")
    except Exception:
        used_tree = False
        pairs = np.empty((0, 2), dtype=np.int64)

    if len(pairs):
        pairs = np.asarray(pairs, dtype=np.int64)
        deltas = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        deltas -= box_length * np.rint(deltas / box_length)
        distances_squared = np.einsum("ij,ij->i", deltas, deltas)
        cutoffs = radius_scale * (radii[pairs[:, 0]] + radii[pairs[:, 1]]) + skin
        pairs = pairs[distances_squared <= cutoffs * cutoffs]
        if len(pairs):
            pair_i = pairs[:, 0].astype(np.int64, copy=True)
            pair_j = pairs[:, 1].astype(np.int64, copy=True)
            for i, j in zip(pair_i, pair_j):
                i = int(i)
                j = int(j)
                adjacency[i].append(j)
                adjacency[j].append(i)
        return adjacency, pair_i, pair_j
    if used_tree:
        return adjacency, pair_i, pair_j

    cell_cutoff = max(max_cutoff, 1.0e-12)
    cell_list = PeriodicCellList(box_length, cutoff=cell_cutoff)
    cell_list.build(positions)
    pair_i_list: list[int] = []
    pair_j_list: list[int] = []
    for i, position in enumerate(positions):
        search_radius = radius_scale * (float(radii[i]) + max_radius) + skin
        for j in cell_list.nearby_indices(position, search_radius):
            if j <= i:
                continue
            delta = minimum_image_delta(position, positions[j], box_length)
            distance_squared = float(np.dot(delta, delta))
            cutoff = radius_scale * float(radii[i] + radii[j]) + skin
            if distance_squared <= cutoff * cutoff:
                pair_i_list.append(i)
                pair_j_list.append(j)
                adjacency[i].append(j)
                adjacency[j].append(i)
    pair_i = np.asarray(pair_i_list, dtype=np.int64)
    pair_j = np.asarray(pair_j_list, dtype=np.int64)
    return adjacency, pair_i, pair_j


def _schedule_pair_event(
    heap: list[tuple[float, int, int, int, int, int]],
    positions: np.ndarray,
    velocities: np.ndarray,
    last_update_times: np.ndarray,
    particle_versions: np.ndarray,
    base_radii: np.ndarray,
    box_length: float,
    initial_scale: float,
    growth_rate: float,
    current_time: float,
    event_epsilon: float,
    pair: tuple[int, int],
    schedule_generation: int,
) -> bool:
    i, j = pair
    if i == j:
        return False
    pos_i = _position_at(positions[i], velocities[i], float(last_update_times[i]), current_time, box_length)
    pos_j = _position_at(positions[j], velocities[j], float(last_update_times[j]), current_time, box_length)
    contact_base = float(base_radii[i] + base_radii[j])
    delta = minimum_image_delta(pos_i, pos_j, box_length)
    offsets = np.zeros((1, 3)) if np.ndim(box_length) == 0 else _PERIODIC_IMAGE_OFFSETS * box_length
    predictions = [
        _pair_collision_time(delta + offset, velocities[i] - velocities[j],
                             _scale_at(initial_scale, growth_rate, current_time) * contact_base,
                             growth_rate * contact_base, event_epsilon)
        for offset in offsets
    ]
    t_collision = min((time for time in predictions if time is not None), default=None)
    if t_collision is None:
        return False
    heapq.heappush(
        heap,
        (
            float(current_time + t_collision),
            int(i),
            int(j),
            int(particle_versions[i]),
            int(particle_versions[j]),
            int(schedule_generation),
        ),
    )
    return True


def _schedule_pair_events_vectorized(
    heap: list[tuple[float, int, int, int, int, int]],
    positions: np.ndarray,
    velocities: np.ndarray,
    particle_versions: np.ndarray,
    base_radii: np.ndarray,
    box_length: float,
    initial_scale: float,
    growth_rate: float,
    current_time: float,
    event_epsilon: float,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    schedule_generation: int,
) -> int:
    if len(pair_i) == 0:
        return 0
    pair_i_array = pair_i
    pair_j_array = pair_j
    deltas = positions[pair_i_array] - positions[pair_j_array]
    deltas -= box_length * np.rint(deltas / box_length)
    relative_velocities = velocities[pair_i_array] - velocities[pair_j_array]
    contact_base = base_radii[pair_i_array] + base_radii[pair_j_array]
    contact = _scale_at(initial_scale, growth_rate, current_time) * contact_base
    contact_rate = growth_rate * contact_base
    rectangular = np.ndim(box_length) != 0
    if rectangular:
        # A non-nearest image can collide first in a short axis. Predict each
        # adjacent image, then retain the earliest event for each neighbor pair.
        offsets = _PERIODIC_IMAGE_OFFSETS * box_length
        deltas = (deltas[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        relative_velocities = np.repeat(relative_velocities, len(offsets), axis=0)
        contact = np.repeat(contact, len(offsets))
        contact_rate = np.repeat(contact_rate, len(offsets))
    c = np.einsum("ij,ij->i", deltas, deltas) - contact * contact
    b_half = np.einsum("ij,ij->i", deltas, relative_velocities) - contact * contact_rate
    a = np.einsum("ij,ij->i", relative_velocities, relative_velocities) - contact_rate * contact_rate
    b = 2.0 * b_half
    times = np.full(len(deltas), np.inf, dtype=float)

    overlap_mask = (c <= 1.0e-12) & (b_half < 0.0)
    times[overlap_mask] = event_epsilon

    linear_mask = (~overlap_mask) & (np.abs(a) <= 1.0e-14) & (b < 0.0)
    linear_times = np.full(len(deltas), np.inf, dtype=float)
    linear_times[linear_mask] = -c[linear_mask] / b[linear_mask]
    linear_mask &= linear_times > event_epsilon
    times[linear_mask] = linear_times[linear_mask]

    quadratic_mask = (~overlap_mask) & (np.abs(a) > 1.0e-14)
    discriminant = b * b - 4.0 * a * c
    quadratic_mask &= discriminant >= 0.0
    if np.any(quadratic_mask):
        sqrt_disc = np.sqrt(np.maximum(discriminant[quadratic_mask], 0.0))
        roots_1 = (-b[quadratic_mask] - sqrt_disc) / (2.0 * a[quadratic_mask])
        roots_2 = (-b[quadratic_mask] + sqrt_disc) / (2.0 * a[quadratic_mask])
        roots_1 = np.where(roots_1 > event_epsilon, roots_1, np.inf)
        roots_2 = np.where(roots_2 > event_epsilon, roots_2, np.inf)
        times[quadratic_mask] = np.minimum(roots_1, roots_2)

    if rectangular:
        times = times.reshape(len(pair_i_array), len(offsets)).min(axis=1)
    valid = np.isfinite(times)
    if not np.any(valid):
        return 0
    absolute_times = current_time + times[valid]
    valid_i = pair_i_array[valid]
    valid_j = pair_j_array[valid]
    valid_versions_i = particle_versions[valid_i]
    valid_versions_j = particle_versions[valid_j]
    for event_time, i, j, version_i, version_j in zip(
        absolute_times,
        valid_i,
        valid_j,
        valid_versions_i,
        valid_versions_j,
    ):
        heapq.heappush(
            heap,
            (
                float(event_time),
                int(i),
                int(j),
                int(version_i),
                int(version_j),
                int(schedule_generation),
            ),
        )
    return int(np.count_nonzero(valid))


def _run_exhaustive_events(
    positions: np.ndarray,
    velocities: np.ndarray,
    target_radii: np.ndarray,
    box_length: float,
    initial_scale: float,
    compression_rate: float,
    max_events: int,
    event_epsilon: float,
    masses: np.ndarray,
    rng: np.random.Generator,
):
    scale = float(initial_scale)
    events = 0
    simulated_time = 0.0
    last_gap_rate = 0.0
    periodic_rechecks = 0
    stop_reason = "target_radius_scale_reached"
    recheck_interval = _periodic_recheck_interval(
        velocities,
        target_radii,
        box_length,
        compression_rate,
        event_epsilon,
    )

    while scale < 1.0 - 1.0e-14:
        if events >= max_events:
            stop_reason = "max_events_reached"
            break
        time_to_target = (1.0 - scale) / compression_rate
        collision_window = min(time_to_target, recheck_interval)
        t_collision, pair = _next_collision(
            positions,
            velocities,
            target_radii,
            box_length,
            scale,
            compression_rate,
            event_epsilon,
            max_time=collision_window,
        )
        if pair is None or t_collision is None:
            positions = np.mod(positions + velocities * collision_window, box_length)
            simulated_time += float(collision_window)
            scale = min(1.0, scale + compression_rate * float(collision_window))
            if scale < 1.0 - 1.0e-14:
                periodic_rechecks += 1
                continue
            break
        if t_collision >= time_to_target:
            positions = np.mod(positions + velocities * time_to_target, box_length)
            simulated_time += float(time_to_target)
            scale = 1.0
            break

        positions = np.mod(positions + velocities * t_collision, box_length)
        simulated_time += float(t_collision)
        scale = min(1.0, scale + compression_rate * float(t_collision))
        last_gap_rate = _resolve_collision(
            positions,
            velocities,
            target_radii,
            box_length,
            scale,
            compression_rate,
            pair,
            masses,
            rng,
        )
        events += 1

    return positions, {
        "events": int(events),
        "simulated_time": float(simulated_time),
        "final_radius_scale": float(scale),
        "stop_reason": stop_reason,
        "last_pre_collision_gap_rate": float(last_gap_rate),
        "event_search_backend": "exhaustive_pair_scan",
        "neighbor_rebuilds": 0,
        "event_invalidations": 0,
        "scheduled_event_count": 0,
        "max_candidate_pair_count": int(len(target_radii) * max(0, len(target_radii) - 1) // 2),
        "exhaustive_periodic_rechecks": int(periodic_rechecks),
        "exhaustive_periodic_recheck_interval": float(recheck_interval),
    }


def _run_verlet_heap_events(
    positions: np.ndarray,
    velocities: np.ndarray,
    target_radii: np.ndarray,
    box_length: float,
    initial_scale: float,
    compression_rate: float,
    max_events: int,
    event_epsilon: float,
    masses: np.ndarray,
    rng: np.random.Generator,
    *,
    neighbor_skin: float,
    neighbor_velocity_safety_factor: float,
    neighbor_rebuild_fraction: float,
    max_heap_factor: float,
):
    n_particles = len(target_radii)
    current_time = 0.0
    target_time = (1.0 - float(initial_scale)) / compression_rate
    last_update_times = np.zeros(n_particles, dtype=float)
    particle_versions = np.zeros(n_particles, dtype=np.int64)
    heap: list[tuple[float, int, int, int, int, int]] = []
    adjacency: list[list[int]] = [[] for _ in range(n_particles)]
    schedule_generation = 0
    rebuild_deadline = 0.0
    rebuilds = 0
    invalidations = 0
    scheduled_events = 0
    max_candidate_pair_count = 0
    max_heap_size = 0
    last_gap_rate = 0.0
    stop_reason = "target_radius_scale_reached"

    def rebuild_events() -> None:
        nonlocal adjacency
        nonlocal heap
        nonlocal schedule_generation
        nonlocal rebuild_deadline
        nonlocal rebuilds
        nonlocal scheduled_events
        nonlocal max_candidate_pair_count
        nonlocal max_heap_size

        _materialize_positions(positions, velocities, last_update_times, current_time, box_length)
        schedule_generation += 1
        heap = []
        scale = _scale_at(initial_scale, compression_rate, current_time)
        adjacency, pair_i, pair_j = _build_verlet_pairs(positions, target_radii, box_length, scale, neighbor_skin)
        max_candidate_pair_count = max(max_candidate_pair_count, len(pair_i))
        scheduled_events += _schedule_pair_events_vectorized(
            heap,
            positions,
            velocities,
            particle_versions,
            target_radii,
            box_length,
            initial_scale,
            compression_rate,
            current_time,
            event_epsilon,
            pair_i,
            pair_j,
            schedule_generation,
        )
        max_heap_size = max(max_heap_size, len(heap))
        rebuilds += 1

        max_speed = float(np.max(np.linalg.norm(velocities, axis=1))) if n_particles else 0.0
        max_radius = float(np.max(target_radii)) if n_particles else 0.0
        relative_speed_bound = (
            2.0 * max(1.0, neighbor_velocity_safety_factor) * max_speed
            + 2.0 * compression_rate * max_radius
        )
        horizon = float(neighbor_rebuild_fraction) * neighbor_skin / max(relative_speed_bound, 1.0e-12)
        if np.ndim(box_length) != 0:
            horizon = min(horizon, _periodic_recheck_interval(velocities, target_radii, box_length, compression_rate, event_epsilon))
        rebuild_deadline = current_time + max(horizon, 10.0 * event_epsilon)

    rebuild_events()
    events = 0
    while current_time < target_time - 1.0e-14:
        if events >= max_events:
            stop_reason = "max_events_reached"
            break

        if len(heap) > max_heap_factor * max(1, max_candidate_pair_count + 2 * n_particles):
            rebuild_events()
            continue

        next_deadline = min(target_time, rebuild_deadline)
        next_event = None
        while heap:
            event_time, i, j, version_i, version_j, generation = heapq.heappop(heap)
            if generation != schedule_generation:
                invalidations += 1
                continue
            if version_i != int(particle_versions[i]) or version_j != int(particle_versions[j]):
                invalidations += 1
                continue
            next_event = (event_time, i, j)
            break

        if next_event is None:
            current_time = next_deadline
            if current_time >= target_time - 1.0e-14:
                break
            rebuild_events()
            continue

        event_time, i, j = next_event
        if event_time > next_deadline:
            heapq.heappush(
                heap,
                (
                    float(event_time),
                    int(i),
                    int(j),
                    int(particle_versions[i]),
                    int(particle_versions[j]),
                    int(schedule_generation),
                ),
            )
            current_time = next_deadline
            if current_time >= target_time - 1.0e-14:
                break
            rebuild_events()
            continue

        current_time = max(float(event_time), current_time)
        positions[i] = _position_at(positions[i], velocities[i], float(last_update_times[i]), current_time, box_length)
        positions[j] = _position_at(positions[j], velocities[j], float(last_update_times[j]), current_time, box_length)
        last_update_times[i] = current_time
        last_update_times[j] = current_time
        scale = min(1.0, _scale_at(initial_scale, compression_rate, current_time))
        last_gap_rate = _resolve_collision(
            positions,
            velocities,
            target_radii,
            box_length,
            scale,
            compression_rate,
            (i, j),
            masses,
            rng,
        )
        particle_versions[i] += 1
        particle_versions[j] += 1
        events += 1

        touched_pairs: set[tuple[int, int]] = set()
        for particle_index in (i, j):
            for neighbor_index in adjacency[particle_index]:
                a, b = sorted((particle_index, neighbor_index))
                touched_pairs.add((a, b))
        for pair in touched_pairs:
            if _schedule_pair_event(
                heap,
                positions,
                velocities,
                last_update_times,
                particle_versions,
                target_radii,
                box_length,
                initial_scale,
                compression_rate,
                current_time,
                event_epsilon,
                pair,
                schedule_generation,
            ):
                scheduled_events += 1
        max_heap_size = max(max_heap_size, len(heap))

    _materialize_positions(positions, velocities, last_update_times, min(current_time, target_time), box_length)
    scale = min(1.0, _scale_at(initial_scale, compression_rate, min(current_time, target_time)))
    return positions, {
        "events": int(events),
        "simulated_time": float(min(current_time, target_time)),
        "final_radius_scale": float(scale),
        "stop_reason": stop_reason,
        "last_pre_collision_gap_rate": float(last_gap_rate),
        "event_search_backend": "verlet_heap",
        "neighbor_skin": float(neighbor_skin),
        "neighbor_velocity_safety_factor": float(neighbor_velocity_safety_factor),
        "neighbor_rebuild_fraction": float(neighbor_rebuild_fraction),
        "neighbor_rebuilds": int(rebuilds),
        "event_invalidations": int(invalidations),
        "scheduled_event_count": int(scheduled_events),
        "max_candidate_pair_count": int(max_candidate_pair_count),
        "max_heap_size": int(max_heap_size),
    }


def _resolve_collision(
    positions: np.ndarray,
    velocities: np.ndarray,
    base_radii: np.ndarray,
    box_length: float,
    scale: float,
    growth_rate: float,
    pair: tuple[int, int],
    masses: np.ndarray,
    rng: np.random.Generator,
) -> float:
    i, j = pair
    delta = minimum_image_delta(positions[i], positions[j], box_length)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-14:
        normal = rng.normal(size=3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-14:
            normal = np.array([1.0, 0.0, 0.0])
            normal_norm = 1.0
        normal = normal / normal_norm
    else:
        normal = delta / distance

    contact_rate = growth_rate * float(base_radii[i] + base_radii[j])
    relative_normal_velocity = float(np.dot(velocities[i] - velocities[j], normal))
    gap_rate = relative_normal_velocity - contact_rate
    if gap_rate >= 0.0:
        return gap_rate

    inv_mass_sum = 1.0 / masses[i] + 1.0 / masses[j]
    impulse = -2.0 * gap_rate / inv_mass_sum
    velocities[i] += (impulse / masses[i]) * normal
    velocities[j] -= (impulse / masses[j]) * normal
    return gap_rate


def _relax_overlaps(
    positions: np.ndarray,
    radii: np.ndarray,
    box_length: float,
    rng: np.random.Generator,
    *,
    max_steps: int,
    tolerance: float,
):
    positions = np.asarray(positions, dtype=float).copy()
    max_overlap = 0.0
    overlap_count = 0
    for step in range(max(0, int(max_steps))):
        overlap_pairs, min_gap = find_overlap_pairs(positions, radii, box_length, tolerance=tolerance)
        overlap_count = len(overlap_pairs)
        max_overlap = max(0.0, -float(min_gap)) if np.isfinite(min_gap) else 0.0
        if not overlap_pairs:
            return positions, {
                "steps_used": step,
                "overlap_count": 0,
                "max_overlap": max_overlap,
                "min_gap": float(min_gap),
            }
        for i, j, gap in overlap_pairs:
            delta = minimum_image_delta(positions[i], positions[j], box_length)
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-14:
                direction = rng.normal(size=3)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1.0e-14:
                    direction = np.array([1.0, 0.0, 0.0])
                    direction_norm = 1.0
                direction = direction / direction_norm
            else:
                direction = delta / distance
            correction = 0.5 * (-float(gap) + tolerance) * direction
            positions[i] = np.mod(positions[i] + correction, box_length)
            positions[j] = np.mod(positions[j] - correction, box_length)

    overlap_pairs, min_gap = find_overlap_pairs(positions, radii, box_length, tolerance=tolerance)
    if np.isfinite(min_gap):
        max_overlap = max(0.0, -float(min_gap))
    overlap_count = len(overlap_pairs)
    return positions, {
        "steps_used": int(max_steps),
        "overlap_count": overlap_count,
        "max_overlap": max_overlap,
        "min_gap": float(min_gap),
    }


class LubachevskyStillingerGenerator:
    name = "lubachevsky_stillinger"

    def generate(self, context: GeneratorContext) -> PackingResult:
        start = time.perf_counter()
        params = context.config.algorithm.parameters
        target_radii = context.resolved.radii.copy()
        box_length = context.domain.box_lengths if context.resolved.fixed_dimensions else context.domain.box_length
        target_phi = float(context.resolved.resolved_phi if context.resolved.fixed_dimensions else context.resolved.target_phi)

        initialization_requested, initialization_selected = _select_initialization(params, target_phi)
        if initialization_selected == "force_biased_warm_start":
            warm_start_phi = _warm_start_packing_fraction(params, target_phi)
            initial_scale = float((warm_start_phi / target_phi) ** (1.0 / 3.0))
        else:
            warm_start_phi = None
            initial_scale = _initial_radius_scale(params, target_phi)
        compression_rate = float(params.get("compression_rate", params.get("contraction_rate", 1.0e-3)))
        velocity_scale = float(params.get("velocity_scale", 1.0))
        max_events = int(params.get("max_events", max(1000, 250 * len(target_radii))))
        max_attempts = int(params.get("max_attempts_per_particle", 30000))
        candidates = int(params.get("candidates_per_particle", 30))
        final_cleanup_steps = int(params.get("final_cleanup_steps", 200))
        event_epsilon = float(params.get("event_epsilon", 1.0e-10))
        mass_mode = str(params.get("mass_mode", "equal")).strip().lower()
        event_backend_requested = str(params.get("event_backend", params.get("event_search_backend", "auto"))).strip().lower()
        event_backend = event_backend_requested
        auto_exhaustive_particle_threshold = int(
            params.get("auto_exhaustive_particle_threshold", AUTO_EXHAUSTIVE_PARTICLE_THRESHOLD)
        )
        mean_target_diameter = max(2.0 * float(np.mean(target_radii)), 1.0e-12)
        neighbor_skin = float(
            params.get("neighbor_skin", params.get("neighbor_skin_fraction", 1.0) * mean_target_diameter)
        )
        neighbor_velocity_safety_factor = float(params.get("neighbor_velocity_safety_factor", 2.0))
        neighbor_rebuild_fraction = float(
            params.get("neighbor_rebuild_fraction", DEFAULT_NEIGHBOR_REBUILD_FRACTION)
        )
        max_heap_factor = float(params.get("max_heap_factor", 8.0))
        tolerance_overlap = float(context.config.validation.tolerance_overlap)

        if not 0.0 < initial_scale <= 1.0:
            raise ValueError("initial_radius_scale must be in (0, 1].")
        if compression_rate < 0.0:
            raise ValueError("compression_rate must be >= 0.")
        if initial_scale < 1.0 and compression_rate <= 0.0:
            raise ValueError("compression_rate must be > 0 when initial_radius_scale < 1.")
        if velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be > 0.")
        if max_events < 0:
            raise ValueError("max_events must be >= 0.")
        if mass_mode == "equal":
            masses = np.ones_like(target_radii)
        elif mass_mode in {"volume", "radius_cubed", "diameter_cubed"}:
            masses = np.maximum(target_radii**3, 1.0e-18)
        else:
            raise ValueError("mass_mode must be 'equal' or 'volume'.")
        if event_backend in {"verlet", "neighbor_heap", "cell_verlet", "cell_list"}:
            event_backend = "verlet_heap"
        if event_backend == "auto":
            event_backend = (
                "exhaustive_pair_scan"
                if len(target_radii) <= auto_exhaustive_particle_threshold
                else "verlet_heap"
            )
        if event_backend not in {"verlet_heap", "exhaustive", "exhaustive_pair_scan"}:
            raise ValueError("event_backend must be 'auto', 'verlet_heap', or 'exhaustive'.")
        if auto_exhaustive_particle_threshold < 0:
            raise ValueError("auto_exhaustive_particle_threshold must be >= 0.")
        if neighbor_skin <= 0.0:
            raise ValueError("neighbor_skin or neighbor_skin_fraction must be > 0.")
        if neighbor_velocity_safety_factor <= 0.0:
            raise ValueError("neighbor_velocity_safety_factor must be > 0.")
        if not 0.0 < neighbor_rebuild_fraction <= 1.0:
            raise ValueError("neighbor_rebuild_fraction must be in (0, 1].")
        if max_heap_factor < 1.0:
            raise ValueError("max_heap_factor must be >= 1.")

        start_radii = target_radii * initial_scale
        if initialization_selected == "force_biased_warm_start":
            positions, init_diag = _force_biased_warm_start(
                context,
                target_radii,
                target_phi,
                float(warm_start_phi),
                tolerance_overlap,
            )
        else:
            positions, init_diag = poisson_disk_place(
                start_radii,
                box_length,
                context.rng,
                max_attempts_per_particle=max_attempts,
                candidates_per_particle=candidates,
            )
        velocities = _random_velocities(len(target_radii), context.rng, velocity_scale)
        if initial_scale >= 1.0 - 1.0e-14:
            event_diag = {
                "events": 0,
                "simulated_time": 0.0,
                "final_radius_scale": 1.0,
                "stop_reason": "initial_radius_scale_already_target",
                "last_pre_collision_gap_rate": 0.0,
                "event_search_backend": event_backend,
                "neighbor_rebuilds": 0,
                "event_invalidations": 0,
                "scheduled_event_count": 0,
                "max_candidate_pair_count": 0,
            }
        elif event_backend in {"exhaustive", "exhaustive_pair_scan"}:
            positions, event_diag = _run_exhaustive_events(
                positions,
                velocities,
                target_radii,
                box_length,
                initial_scale,
                compression_rate,
                max_events,
                event_epsilon,
                masses,
                context.rng,
            )
        else:
            positions, event_diag = _run_verlet_heap_events(
                positions,
                velocities,
                target_radii,
                box_length,
                initial_scale,
                compression_rate,
                max_events,
                event_epsilon,
                masses,
                context.rng,
                neighbor_skin=neighbor_skin,
                neighbor_velocity_safety_factor=neighbor_velocity_safety_factor,
                neighbor_rebuild_fraction=neighbor_rebuild_fraction,
                max_heap_factor=max_heap_factor,
            )

        if event_diag["final_radius_scale"] < 1.0 - 1.0e-10:
            if initialization_selected == "force_biased_warm_start":
                next_step = (
                    "Increase max_events, increase compression_rate, tune "
                    "warm_start_packing_fraction closer to the target, or use force_biased "
                    "for practical high-density generation."
                )
            else:
                next_step = (
                    "Increase max_events, increase compression_rate, lower initial_radius_scale, "
                    "or choose force_biased_warm_start initialization for high-density LS runs."
                )
            if target_phi > LS_HIGH_DENSITY_VALIDATED_PHI_LIMIT:
                next_step += (
                    " Target packing_fraction above 0.62 remains experimental for this Python "
                    "LS event stage."
                )
            raise GenerationError(
                "Lubachevsky-Stillinger stopped before reaching the target radii "
                f"(scale={event_diag['final_radius_scale']:.6g}, events={event_diag['events']}, "
                f"max_events={max_events}). "
                f"{next_step}"
            )

        pre_cleanup_overlap_pairs, pre_cleanup_min_gap = find_overlap_pairs(
            positions,
            target_radii,
            box_length,
            tolerance=tolerance_overlap,
        )
        pre_cleanup_max_overlap = (
            max(0.0, -float(pre_cleanup_min_gap)) if np.isfinite(pre_cleanup_min_gap) else 0.0
        )

        positions, cleanup_diag = _relax_overlaps(
            positions,
            target_radii,
            box_length,
            context.rng,
            max_steps=final_cleanup_steps,
            tolerance=tolerance_overlap,
        )
        overlap_pairs, min_gap = find_overlap_pairs(positions, target_radii, box_length, tolerance=tolerance_overlap)
        max_overlap = max(0.0, -float(min_gap)) if np.isfinite(min_gap) else 0.0

        ps = ParticleSet(
            positions=positions,
            radii=target_radii,
            box_length=box_length,
        )
        status = "success" if not overlap_pairs else "warning"
        high_density_warning = None
        if target_phi > LS_HIGH_DENSITY_VALIDATED_PHI_LIMIT:
            high_density_warning = (
                "target_packing_fraction above 0.62 remains experimental for the Python LS "
                "event stage; local N=500 tests at packing_fraction=0.64 did not reach the "
                "target within 120000 events."
            )
        return PackingResult(
            particle_set=ps.wrapped(),
            generator_name=self.name,
            status=status,
            diagnostics={
                "note": (
                    "Native Lubachevsky-Stillinger dynamic-compression generator: reduced-radius "
                    "non-overlapping initialization, event-driven elastic hard-sphere collisions, and "
                    "linear particle growth in a fixed periodic box until the configured target "
                    "packing fraction is reached."
                ),
                "reference_model": (
                    "Based on the Lubachevsky-Stillinger fixed-volume growth protocol and "
                    "event-driven billiards scheduling. The Python implementation selects an "
                    "exhaustive pair scan for small systems and a Verlet-neighbor priority-queue "
                    "scheduler for larger systems unless the user explicitly chooses a backend."
                ),
                "protocol": "fixed_periodic_box_linear_radius_growth",
                "collision_model": "elastic_hard_sphere_collision_with_expanding_contact_surface",
                "postprocess_role": "final overlap cleanup is numerical polishing, not the LS compression mechanism",
                "initialization_requested": initialization_requested,
                "initialization": initialization_selected,
                "initial_radius_scale": float(initial_scale),
                "initial_packing_fraction": _packing_fraction(start_radii, box_length),
                "warm_start_packing_fraction": float(warm_start_phi) if warm_start_phi is not None else None,
                "target_packing_fraction": target_phi,
                "high_density_warning": high_density_warning,
                "compression_rate": float(compression_rate),
                "velocity_scale": float(velocity_scale),
                "mass_mode": mass_mode,
                "max_events": int(max_events),
                "event_backend_requested": event_backend_requested,
                "event_backend_selected": event_backend,
                "auto_exhaustive_particle_threshold": int(auto_exhaustive_particle_threshold),
                "neighbor_rebuild_fraction": float(neighbor_rebuild_fraction),
                **event_diag,
                "pre_cleanup_min_gap": float(pre_cleanup_min_gap),
                "pre_cleanup_max_overlap": float(pre_cleanup_max_overlap),
                "pre_cleanup_overlap_pair_count": int(len(pre_cleanup_overlap_pairs)),
                "max_attempts_per_particle": int(max_attempts),
                "candidates_per_particle": int(candidates),
                "initialization_diagnostics": init_diag,
                "final_cleanup": cleanup_diag,
                "final_min_gap": float(min_gap),
                "final_max_overlap": float(max_overlap),
                "final_overlap_pair_count": int(len(overlap_pairs)),
            },
            elapsed_time_s=time.perf_counter() - start,
        )
