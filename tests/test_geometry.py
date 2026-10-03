import numpy as np
import pytest

from spherepackgen.domain.domain import PeriodicCube
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.generators.force_biased import _VerletPairCache, _accumulate_pair_forces
from spherepackgen.generators.lubachevsky_stillinger import (
    _build_verlet_pairs,
    _pair_collision_time,
    _resolve_collision,
    _run_exhaustive_events,
    _run_verlet_heap_events,
)
from spherepackgen.generators.rsa import _candidate_min_gap, _candidate_min_gaps, _candidate_min_gaps_cell_list
from spherepackgen.utils.cell_list import PeriodicCellList, find_overlap_pairs, minimum_image_delta


def test_minimum_image_distance_wraps():
    domain = PeriodicCube(10.0)
    a = np.array([9.8, 0.0, 0.0])
    b = np.array([0.2, 0.0, 0.0])
    assert abs(domain.distance(a, b) - 0.4) < 1.0e-12


def test_particle_packing_fraction():
    particles = ParticleSet(
        positions=np.array([[0, 0, 0], [2, 2, 2]], dtype=float),
        radii=np.array([0.5, 0.5]),
        box_length=4.0,
    )
    expected = 2 * 4.0 * np.pi * 0.5**3 / 3.0 / 4.0**3
    assert abs(particles.packing_fraction - expected) < 1.0e-12


def test_overlap_detection_periodic():
    positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], dtype=float)
    radii = np.array([0.2, 0.2])
    pairs, min_gap = find_overlap_pairs(positions, radii, 10.0, tolerance=0.0)
    assert len(pairs) == 1
    assert min_gap < 0.0


def test_poisson_disk_batch_candidate_gaps_match_scalar_periodic():
    positions = np.array(
        [
            [0.2, 0.1, 0.1],
            [9.7, 0.1, 0.1],
            [5.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    radii = np.array([0.3, 0.4, 0.2], dtype=float)
    candidates = np.array(
        [
            [9.95, 0.1, 0.1],
            [5.6, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    scalar = np.array(
        [
            _candidate_min_gap(candidate, 0.25, positions, radii, len(radii), 10.0)
            for candidate in candidates
        ]
    )
    batched = _candidate_min_gaps(candidates, 0.25, positions, radii, len(radii), 10.0)
    assert np.allclose(batched, scalar)


def test_poisson_disk_cell_list_candidate_gaps_detect_periodic_overlaps():
    positions = np.array(
        [
            [0.2, 0.1, 0.1],
            [9.7, 0.1, 0.1],
            [5.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    radii = np.array([0.3, 0.4, 0.2], dtype=float)
    candidates = np.array(
        [
            [9.95, 0.1, 0.1],
            [5.6, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    cell_list = PeriodicCellList(10.0, cutoff=2.0 * float(np.max(radii)))
    for index, position in enumerate(positions):
        cell_list.add(index, position)

    scalar = np.array(
        [
            _candidate_min_gap(candidate, 0.25, positions, radii, len(radii), 10.0)
            for candidate in candidates
        ]
    )
    local = _candidate_min_gaps_cell_list(
        candidates,
        0.25,
        positions,
        radii,
        len(radii),
        10.0,
        cell_list,
        search_radius=0.25 + float(np.max(radii)),
    )

    assert np.array_equal(local < 0.0, scalar < 0.0)
    assert local[0] < 0.0


def test_force_biased_verlet_pair_cache_matches_periodic_exact_pairs():
    positions = np.array(
        [
            [0.05, 0.0, 0.0],
            [9.92, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            [5.45, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    radii = np.array([0.20, 0.25, 0.30, 0.15, 0.20], dtype=float)
    outer_ratio = 1.1
    skin = 0.02
    box_length = 10.0
    cache = _VerletPairCache(radii, box_length, outer_ratio, skin)
    cache.ensure_current(positions)

    domain = PeriodicCube(box_length)
    exact_pairs = set()
    for i in range(len(radii) - 1):
        for j in range(i + 1, len(radii)):
            cutoff = outer_ratio * (radii[i] + radii[j]) + skin
            if domain.distance(positions[i], positions[j]) <= cutoff:
                exact_pairs.add((i, j))

    cached_pairs = set(zip(cache.pair_i.tolist(), cache.pair_j.tolist()))
    assert cached_pairs == exact_pairs
    assert cache.pair_search_backend in {"ckdtree", "cell_list"}


def test_force_biased_force_accumulation_matches_add_at():
    active_i = np.array([0, 0, 2, 3, 3], dtype=np.int64)
    active_j = np.array([1, 2, 3, 0, 1], dtype=np.int64)
    pair_forces = np.array(
        [
            [1.0, 0.0, -0.5],
            [0.2, 0.3, 0.4],
            [-0.7, 0.1, 0.0],
            [0.5, -0.2, 0.8],
            [0.0, 0.4, -0.1],
        ],
        dtype=float,
    )
    expected = np.zeros((4, 3), dtype=float)
    np.add.at(expected, active_i, pair_forces)
    np.add.at(expected, active_j, -pair_forces)

    actual = _accumulate_pair_forces(4, active_i, active_j, pair_forces)

    assert np.allclose(actual, expected)


def test_force_biased_verlet_pair_cache_shrinks_cutoff_on_rebuild():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [1.25, 0.0, 0.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    radii = np.full(4, 0.5, dtype=float)
    box_length = 8.0
    skin = 0.02
    cache = _VerletPairCache(radii, box_length, outer_ratio_cutoff=1.30, skin=skin)
    cache.ensure_current(positions)
    assert (0, 2) in set(zip(cache.pair_i.tolist(), cache.pair_j.tolist()))

    cache.set_outer_ratio_cutoff(1.05)
    moved = positions.copy()
    moved[3] += np.array([0.02, 0.0, 0.0])
    cache.ensure_current(moved)

    cached_pairs = set(zip(cache.pair_i.tolist(), cache.pair_j.tolist()))
    assert (0, 1) in cached_pairs
    assert (0, 2) not in cached_pairs
    assert cache.outer_ratio_cutoff == 1.05


def test_lubachevsky_stillinger_collision_time_includes_contact_growth():
    t_collision = _pair_collision_time(
        delta=np.array([1.0, 0.0, 0.0], dtype=float),
        relative_velocity=np.zeros(3, dtype=float),
        contact=0.8,
        contact_rate=0.1,
        event_epsilon=1.0e-12,
    )

    assert t_collision == pytest.approx(2.0)


def test_lubachevsky_stillinger_collision_reflects_expanding_surface_gap_rate():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    velocities = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float)
    base_radii = np.array([0.5, 0.5], dtype=float)
    masses = np.ones(2, dtype=float)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    contact_rate = 0.05 * float(base_radii[0] + base_radii[1])
    initial_momentum = np.sum(masses[:, None] * velocities, axis=0)

    incoming_gap_rate = _resolve_collision(
        positions,
        velocities,
        base_radii,
        box_length=4.0,
        scale=1.0,
        growth_rate=0.05,
        pair=(0, 1),
        masses=masses,
        rng=np.random.default_rng(123),
    )
    outgoing_gap_rate = float(np.dot(velocities[0] - velocities[1], normal) - contact_rate)
    final_momentum = np.sum(masses[:, None] * velocities, axis=0)

    assert incoming_gap_rate < 0.0
    assert outgoing_gap_rate == pytest.approx(-incoming_gap_rate)
    assert np.allclose(final_momentum, initial_momentum)


def test_lubachevsky_stillinger_verlet_pairs_match_exact_periodic_neighbors():
    positions = np.array(
        [
            [0.05, 0.0, 0.0],
            [9.90, 0.0, 0.0],
            [4.0, 4.0, 4.0],
            [4.62, 4.0, 4.0],
            [7.0, 7.0, 7.0],
        ],
        dtype=float,
    )
    radii = np.array([0.20, 0.28, 0.30, 0.22, 0.25], dtype=float)
    box_length = 10.0
    radius_scale = 0.8
    skin = 0.15

    adjacency, pair_i, pair_j = _build_verlet_pairs(positions, radii, box_length, radius_scale, skin)

    exact_pairs = set()
    for i in range(len(radii) - 1):
        for j in range(i + 1, len(radii)):
            cutoff = radius_scale * float(radii[i] + radii[j]) + skin
            delta = minimum_image_delta(positions[i], positions[j], box_length)
            if float(np.linalg.norm(delta)) <= cutoff:
                exact_pairs.add((i, j))

    cached_pairs = set(zip(pair_i.tolist(), pair_j.tolist()))
    assert cached_pairs == exact_pairs
    for i, j in exact_pairs:
        assert j in adjacency[i]
        assert i in adjacency[j]


def test_lubachevsky_stillinger_verlet_heap_matches_exhaustive_small_system():
    positions = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.6, 0.5, 0.5],
            [3.2, 0.5, 0.5],
            [4.3, 0.5, 0.5],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [0.12, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [-0.10, 0.0, 0.0],
        ],
        dtype=float,
    )
    radii = np.full(4, 0.45, dtype=float)
    box_length = 5.0
    common = {
        "target_radii": radii,
        "box_length": box_length,
        "initial_scale": 0.7,
        "compression_rate": 1.0e-2,
        "max_events": 200,
        "event_epsilon": 1.0e-10,
        "masses": np.ones(4, dtype=float),
        "rng": np.random.default_rng(123),
    }

    exhaustive_positions, exhaustive_diag = _run_exhaustive_events(
        positions.copy(),
        velocities.copy(),
        **common,
    )
    heap_positions, heap_diag = _run_verlet_heap_events(
        positions.copy(),
        velocities.copy(),
        **common,
        neighbor_skin=2.0,
        neighbor_velocity_safety_factor=2.0,
        neighbor_rebuild_fraction=0.8,
        max_heap_factor=8.0,
    )

    assert exhaustive_diag["final_radius_scale"] == pytest.approx(1.0)
    assert heap_diag["final_radius_scale"] == pytest.approx(1.0)
    assert exhaustive_diag["events"] == heap_diag["events"] == 18
    assert heap_diag["neighbor_rebuilds"] > 0
    assert np.allclose(heap_positions, exhaustive_positions, atol=1.0e-9)
    for candidate_positions in (exhaustive_positions, heap_positions):
        overlap_pairs, min_gap = find_overlap_pairs(candidate_positions, radii, box_length, tolerance=0.0)
        assert overlap_pairs == []
        assert min_gap > 0.0
