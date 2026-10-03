import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from box_reference import image_distances
from spherepackgen.api import load_config, run_generation
from spherepackgen.config.schema import DomainConfig
from spherepackgen.domain.enums import SizeDistributionClass
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.workflows.resolve import resolve_parameters
from spherepackgen.validation.quality_gate import validate_particle_set


def fixed_config(tmp_path, algorithm="poisson_disk", lengths=(4, 4, 2), count=12):
    config = load_config("configs/examples/hard_core_random.yaml")
    config.structure.size_distribution = SizeDistributionClass.MONODISPERSE
    config.size_distribution.type = "monodisperse"
    config.domain = DomainConfig(type="periodic_box", length_m=lengths[0] * 1e-6, depth_m=lengths[2] * 1e-6)
    config.physical.mean_diameter_m = 1e-6
    config.physical.medium_thickness_m = None
    config.physical.packing_fraction = count * np.pi / 6 / np.prod(lengths)
    config.particles.num_particles = count
    config.output.path = tmp_path / algorithm
    config.output.make_plots = False
    config.output.formats = ["csv", "json", "hdf5"]
    config.runtime.random_seed = 123
    config.algorithm.name = algorithm
    config.algorithm.parameters = {}
    return config


@pytest.mark.parametrize("lengths", [(4, 4, 2), (3, 3, 6)])
@pytest.mark.parametrize("algorithm", ["poisson_disk", "rsa", "metropolis", "force_biased", "marked_poisson_boolean"])
def test_rectangular_generators_and_export(tmp_path, lengths, algorithm):
    config = fixed_config(tmp_path, algorithm, lengths)
    if algorithm == "marked_poisson_boolean":
        from spherepackgen.domain.enums import SpatialOrderClass
        config.structure.spatial_order = SpatialOrderClass.OVERLAPPING_RANDOM
    bundle = run_generation(config)
    assert bundle.validation.passed, bundle.validation.errors
    ps = bundle.result.particle_set
    assert ps.box_lengths == pytest.approx(lengths)
    assert ps.n_particles == 12
    assert np.all(ps.positions >= 0) and np.all(ps.positions < ps.box_lengths)
    if algorithm != "marked_poisson_boolean":
        distances = image_distances(ps.positions, lengths)
        assert np.min(distances - ps.radii[:, None] - ps.radii[None, :]) >= -config.validation.tolerance_overlap
    metadata = json.loads((config.output.path / "metadata.json").read_text())
    assert metadata["resolved_parameters"]["box_lengths"] == pytest.approx(lengths)
    import h5py
    with h5py.File(config.output.path / "particles.h5") as h5:
        assert h5.attrs["box_lengths"] == pytest.approx(lengths)
        assert "box_length" not in h5.attrs
    replay = load_config(config.output.path / "config_replay.yaml")
    replay.output.path = tmp_path / "replay"
    replay.output.formats = []
    rerun = run_generation(replay)
    assert rerun.result.particle_set.positions == pytest.approx(ps.positions)
    assert rerun.result.particle_set.radii == pytest.approx(ps.radii)


@pytest.mark.parametrize("kind", ["mono", "quasi", "poly", "mixture"])
def test_fixed_dimensions_auto_count_and_reproducibility(tmp_path, kind):
    config = fixed_config(tmp_path)
    config.particles.num_particles = None
    config.physical.packing_fraction = 0.1793
    if kind == "quasi":
        config.structure.size_distribution = SizeDistributionClass.QUASI_MONODISPERSE
    elif kind == "poly":
        config.structure.size_distribution = SizeDistributionClass.CONTINUOUS_POLYDISPERSE
        config.size_distribution.type = "lognormal"
        config.size_distribution.cv_radius = 0.2
    elif kind == "mixture":
        config.structure.size_distribution = SizeDistributionClass.DISCRETE_MIXTURE
        config.size_distribution.species = [{"radius": 0.8, "fraction": 0.5}, {"radius": 1.2, "fraction": 0.5}]
    first, second = resolve_parameters(config), resolve_parameters(config)
    assert first.box_lengths == pytest.approx([4, 4, 2])
    assert first.radii == pytest.approx(second.radii)
    assert np.mean(first.radii) == pytest.approx(0.5)
    assert abs(first.resolved_phi - first.target_phi) <= first.density_resolution_tolerance + 1e-6
    assert first.box_lengths_m == pytest.approx([4e-6, 4e-6, 2e-6])
    if kind == "mono":
        assert first.n_particles == round(first.target_phi * 32 / (np.pi / 6))
        assert first.density_resolution_tolerance == pytest.approx(np.pi / 12 / 32)


def test_fixed_count_conflict_is_rejected(tmp_path):
    config = fixed_config(tmp_path)
    config.particles.num_particles = 100
    with pytest.raises(ValueError, match="inconsistent"):
        resolve_parameters(config)


def test_custom_file_replay_uses_archived_input(tmp_path, monkeypatch):
    config = fixed_config(tmp_path)
    config.particles.num_particles = None
    config.structure.size_distribution = SizeDistributionClass.IMPORTED_DISTRIBUTION
    config.size_distribution.size_file = "sizes.txt"
    config.physical.packing_fraction = 0.12
    (tmp_path / "sizes.txt").write_text("0.8\n1.0\n1.2\n")
    monkeypatch.chdir(tmp_path)
    first = run_generation(config)
    assert first.validation.passed
    replay = load_config(config.output.path / "config_replay.yaml")
    assert Path(replay.size_distribution.size_file).is_relative_to(config.output.path.resolve())
    (tmp_path / "sizes.txt").unlink()
    replay.output.path = tmp_path / "replayed_custom"
    second = run_generation(replay)
    assert second.validation.passed
    assert second.result.particle_set.radii == pytest.approx(first.result.particle_set.radii)
    assert second.result.particle_set.positions == pytest.approx(first.result.particle_set.positions)


def test_self_image_overlap_detected_even_for_one_particle(tmp_path):
    config = fixed_config(tmp_path)
    ps = ParticleSet([[1, 1, 0.2]], [0.5], [4, 4, 0.5])
    report = validate_particle_set(ps, config)
    assert not report.passed
    assert any("own periodic image" in error for error in report.errors)


def test_density_allowance_does_not_hide_generator_volume_error(tmp_path):
    config = fixed_config(tmp_path)
    config.particles.num_particles = None
    resolved = resolve_parameters(config)
    ps = ParticleSet(np.zeros((resolved.n_particles, 3)), resolved.radii * 1.01, resolved.box_length)
    report = validate_particle_set(ps, config, resolved)
    assert not report.passed
    assert any("resolved particle volume" in error for error in report.errors)


@pytest.mark.parametrize("domain,extra", [({"type": "periodic_box", "length": "4 um"}, {}),
    ({"type": "periodic_box", "length": "4 um", "depth": "2 um"}, {"medium_thickness": "2 um"}),
    ({"type": "unknown"}, {})])
def test_invalid_domain_inputs(tmp_path, domain, extra):
    raw = yaml.safe_load(open("configs/examples/hard_core_random.yaml"))
    raw["domain"] = domain
    raw["physical"].pop("medium_thickness", None)
    raw["physical"].update(extra)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)
