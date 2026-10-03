import numpy as np
import pytest

from box_reference import image_distances
from spherepackgen.domain.domain import PeriodicBox, PeriodicCube
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.utils.cell_list import find_overlap_pairs


@pytest.mark.parametrize("lengths", [[4, 4, 4], [4, 4, 1], [2, 2, 8], [2, 3, 4]])
def test_periodic_box_matches_explicit_images(lengths):
    rng = np.random.default_rng(10)
    positions = rng.random((50, 3)) * lengths
    positions[:2] = [[0.01, 0.01, 0.01], np.array(lengths) - 0.01]
    radii = rng.uniform(0.08, 0.25, len(positions))
    domain = PeriodicBox(lengths)
    reference = image_distances(positions, lengths)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert domain.distance(positions[i], positions[j]) == pytest.approx(reference[i, j])
    expected = {(i, j) for i in range(len(positions)) for j in range(i + 1, len(positions))
                if reference[i, j] < radii[i] + radii[j] - 1e-8}
    actual, _ = find_overlap_pairs(positions, radii, lengths, tolerance=1e-8)
    assert {(i, j) for i, j, _ in actual} == expected
    assert domain.volume == pytest.approx(np.prod(lengths))
    assert np.allclose(domain.wrap(positions + np.array(lengths) * [2, -3, 1]), positions)


def test_legacy_cube_and_particle_scaling():
    assert PeriodicCube(box_length=3).box_length == 3
    particles = ParticleSet([[1, 1, 1]], [0.2], [2, 2, 4])
    scaled = particles.scaled(5)
    assert scaled.box_lengths == pytest.approx([10, 10, 20])
    assert scaled.packing_fraction == pytest.approx(particles.packing_fraction)


@pytest.mark.parametrize("lengths", [0, -1, [1, 1], [1, np.nan, 2], [1, 2, np.inf]])
def test_invalid_box_dimensions(lengths):
    with pytest.raises(ValueError, match="positive finite"):
        PeriodicBox(lengths)
