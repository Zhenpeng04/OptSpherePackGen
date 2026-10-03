from itertools import product

import numpy as np
import pytest

from box_reference import image_distances
from test_rectangular_workflow import fixed_config
from spherepackgen.api import run_generation
from spherepackgen.analysis.descriptors import nearest_neighbor_distribution, pair_correlation, structure_factor
from spherepackgen.domain.enums import SpatialOrderClass
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.generators.force_biased import _VerletPairCache
from spherepackgen.generators.lubachevsky_stillinger import _periodic_recheck_interval
from spherepackgen.workflows.resolve import resolve_parameters


@pytest.mark.parametrize("lengths", [[5, 5, 2], [3, 3, 12]])
def test_neighbors_and_verlet_cache_against_images(lengths):
    rng = np.random.default_rng(16)
    positions = rng.random((50, 3)) * lengths
    radii = rng.uniform(0.1, 0.3, 50)
    ps = ParticleSet(positions, radii, lengths)
    distances = image_distances(positions, lengths)
    assert nearest_neighbor_distribution(ps)["distances"] == pytest.approx(np.min(distances, axis=1))
    cache = _VerletPairCache(radii, lengths, 1.1, 0.1)
    cache.ensure_current(positions)
    expected = {(i, j) for i in range(50) for j in range(i + 1, 50)
                if distances[i, j] <= 1.1 * (radii[i] + radii[j]) + 0.1}
    assert set(zip(cache.pair_i, cache.pair_j)) == expected
    reference_histogram = np.histogram(distances[np.triu_indices(50, 1)], bins=np.linspace(0, min(lengths)/2, 11))[0]
    g2 = pair_correlation(ps, bins=10)
    assert g2["counts"] == pytest.approx(reference_histogram)


def test_rectangular_structure_factor_matches_direct_sum():
    positions = np.array([[0.2, 0.4, 0.6], [2, 1, 3], [1.5, 1.8, 4.2]])
    ps = ParticleSet(positions, np.full(3, 0.1), [3, 3, 6])
    shells = {}
    for indices in product(range(-1, 2), repeat=3):
        if indices == (0, 0, 0):
            continue
        wave = 2 * np.pi * np.array(indices) / ps.box_lengths
        key = round(float(np.linalg.norm(wave)), 10)
        shells.setdefault(key, []).append(abs(np.sum(np.exp(-1j * positions @ wave)))**2 / 3)
    result = structure_factor(ps, k_max_index=1)
    assert result["Sk"] == pytest.approx([np.mean(shells[k]) for k in sorted(shells)])


@pytest.mark.parametrize("backend", ["exhaustive", "verlet_heap"])
@pytest.mark.parametrize("lengths", [(4, 4, 2), (3, 3, 6)])
def test_rectangular_ls_collision_backends(tmp_path, backend, lengths):
    config = fixed_config(tmp_path, "lubachevsky_stillinger", lengths)
    config.algorithm.parameters = {"initial_radius_scale": 0.75, "compression_rate": 0.02,
        "max_events": 6000, "event_backend": backend, "final_cleanup_steps": 50}
    bundle = run_generation(config)
    assert bundle.validation.passed, bundle.validation.errors
    assert bundle.result.diagnostics["pre_cleanup_overlap_pair_count"] == 0
    ps = bundle.result.particle_set
    distances = image_distances(ps.positions, lengths)
    assert np.min(distances) >= 1 - config.validation.tolerance_overlap


def test_ls_periodic_recheck_uses_shortest_dimension():
    velocities = np.array([[1, 0, 0], [-1, 0, 0]])
    radii = np.array([0.5, 0.5])
    interval = _periodic_recheck_interval(velocities, radii, np.array([8, 8, 2]), 0.01, 1e-10)
    assert interval == pytest.approx(_periodic_recheck_interval(velocities, radii, 2, 0.01, 1e-10))


def test_fixed_crystal_retains_equal_lattice_spacing(tmp_path):
    config = fixed_config(tmp_path, "crystal")
    config.structure.spatial_order = SpatialOrderClass.PERIODIC_CRYSTAL
    spacing = (4 * np.pi / 6 / 0.2)**(1/3)
    config.domain.length_m = 2 * spacing * 1e-6
    config.domain.depth_m = spacing * 1e-6
    config.particles.num_particles = None
    config.physical.packing_fraction = 0.2
    config.algorithm.parameters = {"lattice_type": "FCC", "unit_cells": [2, 2, 1]}
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.particle_set.n_particles == 16
    assert bundle.result.diagnostics["unit_cells"] == [2, 2, 1]
    assert np.min(image_distances(bundle.result.particle_set.positions, bundle.resolved.box_lengths)) == pytest.approx(spacing / np.sqrt(2))


def test_incommensurate_crystal_dimensions_are_rejected(tmp_path):
    config = fixed_config(tmp_path, "crystal", lengths=(4, 4, 2.3))
    config.structure.spatial_order = SpatialOrderClass.PERIODIC_CRYSTAL
    config.particles.num_particles = None
    with pytest.raises(ValueError, match="unstrained"):
        resolve_parameters(config)


def test_fixed_hcp_is_explicitly_rejected(tmp_path):
    config = fixed_config(tmp_path, "crystal")
    config.structure.spatial_order = SpatialOrderClass.PERIODIC_CRYSTAL
    config.algorithm.parameters["lattice_type"] = "HCP"
    with pytest.raises(ValueError, match="HCP"):
        resolve_parameters(config)


def test_ls_backends_agree_on_rectangular_small_system(tmp_path):
    sets = []
    for backend in ["exhaustive", "verlet_heap"]:
        config = fixed_config(tmp_path / backend, "lubachevsky_stillinger", lengths=(4, 4, 2))
        config.algorithm.parameters = {"initial_radius_scale": 0.75, "compression_rate": 0.2,
            "max_events": 6000, "event_backend": backend, "final_cleanup_steps": 0}
        bundle = run_generation(config)
        assert bundle.validation.passed
        sets.append(bundle.result.particle_set.positions)
    assert sets[0] == pytest.approx(sets[1], abs=1e-7)


def test_rectangular_heap_predicts_non_nearest_image_collision():
    from spherepackgen.generators.lubachevsky_stillinger import (
        _next_collision, _schedule_pair_event, _schedule_pair_events_vectorized)
    positions = np.array([[1., 1., 0.45], [1., 1., 1.5]])
    velocities = np.array([[0., 0., 1.], [0., 0., 0.]])
    radii = np.full(2, 0.4)
    lengths = np.array([4., 4., 2.])
    expected, pair = _next_collision(positions, velocities, radii, lengths, 0.9, 0.01, 1e-10)
    assert pair == (0, 1) and expected > 0
    scalar, vector = [], []
    versions = np.zeros(2, dtype=int)
    assert _schedule_pair_event(scalar, positions, velocities, np.zeros(2), versions,
        radii, lengths, 0.9, 0.01, 0.0, 1e-10, pair, 0)
    assert _schedule_pair_events_vectorized(vector, positions, velocities, versions,
        radii, lengths, 0.9, 0.01, 0.0, 1e-10, np.array([0]), np.array([1]), 0) == 1
    assert scalar[0][0] == pytest.approx(expected)
    assert vector[0][0] == pytest.approx(expected)
