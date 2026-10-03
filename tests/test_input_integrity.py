import numpy as np
import pytest

from spherepackgen.config.loader import config_from_mapping
from spherepackgen.config.schema import ValidationConfig
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.validation.quality_gate import validate_particle_set


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["positions", "radii", "box_length"])
def test_constructor_rejects_nonfinite_geometry(field, value):
    data = dict(positions=[[1., 1., 1.]], radii=[.5], box_length=10.)
    data[field] = [[value, 1., 1.]] if field == "positions" else [value] if field == "radii" else value
    with pytest.raises(ValueError, match="finite"):
        ParticleSet(**data)


@pytest.mark.parametrize("kwargs", [
    {"positions": np.empty((0, 3)), "radii": []},
    {"positions": [[1., 1.]], "radii": [.5]},
    {"positions": [[1., 1., 1.]], "radii": [[.5]]},
    {"positions": [[1., 1., 1.]], "radii": .5},
    {"positions": [[1., 1., 1.]], "radii": [.5], "species_id": [0, 1]},
    {"positions": [[1., 1., 1.]], "radii": [.5], "material_id": [np.nan]},
    {"positions": [[1., 1., 1.]], "radii": [.5], "material_id": [-1]},
    {"positions": [[1., 1., 1.]], "radii": [.5], "material_id": [.5]},
])
def test_constructor_rejects_empty_malformed_or_invalid_labels(kwargs):
    with pytest.raises(ValueError):
        ParticleSet(box_length=10., **kwargs)


@pytest.mark.parametrize("field", ["positions", "radii", "box_length", "species_id"])
def test_quality_gate_rejects_mutated_geometry(field):
    config = config_from_mapping({"physical": {"mean_diameter": "200 nm", "packing_fraction": .1}})
    particles = ParticleSet([[1., 1., 1.]], [.5], 10.)
    if field == "box_length":
        particles.box_length = np.nan
    elif field == "positions":
        particles.positions[0, 0] = np.nan
    elif field == "radii":
        particles.radii[0] = np.nan
    else:
        particles.species_id = np.array([0, 1])
    report = validate_particle_set(particles, config)
    assert not report.passed
    assert report.metrics["input_valid"] is False
    assert report.errors


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.])
@pytest.mark.parametrize("field", ["tolerance_phi", "tolerance_overlap"])
def test_invalid_validation_tolerance_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        ValidationConfig(**{field: value})


def test_valid_single_particle_and_zero_tolerances():
    particles = ParticleSet([[0., 0., 0.]], [.5], 10.)
    config = config_from_mapping({"physical": {"mean_diameter": "200 nm", "packing_fraction": particles.packing_fraction},
                                  "validation": {"tolerance_phi": 0., "tolerance_overlap": 0.}})
    assert validate_particle_set(particles, config).passed


def test_workflow_stops_before_analysis_and_export_for_invalid_generator_output(tmp_path, monkeypatch):
    from spherepackgen.domain.result import PackingResult
    from spherepackgen.workflows import generation
    from spherepackgen.utils.exceptions import GenerationError

    class CorruptGenerator:
        def generate(self, context):
            particles = ParticleSet([[0., 0., 0.]], context.resolved.radii, context.domain.box_length)
            particles.positions[0, 0] = np.nan
            return PackingResult(particles, "corrupt-test", "success")

    config = config_from_mapping({"physical": {"mean_diameter": "200 nm", "packing_fraction": .01},
                                  "particles": {"num_particles": 1},
                                  "output": {"path": str(tmp_path / "bad")}})
    monkeypatch.setattr(generation, "get_generator", lambda config: CorruptGenerator())
    with pytest.raises(GenerationError, match="Invalid generated geometry"):
        generation.run_generation(config)
    assert not (tmp_path / "bad/metadata.json").exists()
    assert not (tmp_path / "bad/particles_real_units.csv").exists()
