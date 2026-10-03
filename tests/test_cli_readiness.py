import json

import pytest
import yaml

from spherepackgen.cli import main


@pytest.mark.parametrize("options,expected", [([], 3), (["--strict"], 3), (["--allow-candidate"], 0)])
def test_cli_candidate_policy(tmp_path, capsys, monkeypatch, options, expected):
    import numpy as np
    from spherepackgen.domain.particles import ParticleSet
    from spherepackgen.domain.result import PackingResult
    from spherepackgen.workflows import generation

    class CandidateGenerator:
        def generate(self, context):
            positions = np.array([[x, y, z] for x in (.25, .75)
                                  for y in (.25, .75) for z in (.25, .75)])
            particles = ParticleSet(positions * context.domain.box_lengths,
                                    context.resolved.radii, context.domain.box_lengths)
            return PackingResult(particles, "candidate-test", "warning",
                                 {"structure_target_met": False})

    monkeypatch.setattr(generation, "get_generator", lambda config: CandidateGenerator())
    raw = {
        "physical": {"mean_diameter": "200 nm", "packing_fraction": .001},
        "structure": {"spatial_order": "hard_core_random"},
        "particles": {"num_particles": 8},
        "runtime": {"random_seed": 123},
        "analysis": {"compute_g2": False, "compute_Sk": False},
        "output": {"path": str(tmp_path / "candidate"), "make_plots": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert main(["run", str(path), *options]) == expected
    summary = json.loads(capsys.readouterr().out)
    assert summary["geometry_valid"] is True
    assert summary["structure_target_met"] is False
    assert summary["export_ready"] is False
    assert summary["status"] == "warning"
    metadata = json.loads((tmp_path / "candidate/metadata.json").read_text())
    manifest = json.loads((tmp_path / "candidate/run_manifest.json").read_text())
    assert metadata["readiness"] == manifest["readiness"]
    assert manifest["readiness"]["export_ready"] is False
    assert (tmp_path / "candidate/particles_real_units.csv").exists()


@pytest.mark.parametrize("options", [[], ["--strict"], ["--allow-candidate"]])
def test_cli_geometry_errors_always_fail(tmp_path, capsys, monkeypatch, options):
    import numpy as np
    from spherepackgen.domain.particles import ParticleSet
    from spherepackgen.domain.result import PackingResult
    from spherepackgen.workflows import generation

    class OverlappingGenerator:
        def generate(self, context):
            particles = ParticleSet(np.zeros((context.resolved.n_particles, 3)),
                                    context.resolved.radii, context.domain.box_length)
            return PackingResult(particles, "overlapping-test", "success")

    raw = {
        "physical": {"mean_diameter": "200 nm", "packing_fraction": .1},
        "particles": {"num_particles": 8},
        "validation": {"require_non_overlap": True},
        "analysis": {"compute_g2": False, "compute_Sk": False},
        "output": {"path": str(tmp_path / "bad"), "make_plots": False},
    }
    # The real quality gate must reject overlapping output regardless of CLI policy.
    monkeypatch.setattr(generation, "get_generator", lambda config: OverlappingGenerator())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert main(["run", str(path), *options]) == 2
    assert json.loads(capsys.readouterr().out)["export_ready"] is False
