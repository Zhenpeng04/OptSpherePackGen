import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import yaml

from spherepackgen import load_config, load_snapshot, run_generation
from spherepackgen.config.loader import config_from_mapping
from spherepackgen.provenance import sha256_file


def config_for(tmp_path, size_file=None):
    raw = {
        "physical": {"mean_diameter": "200 nm", "packing_fraction": .05},
        "structure": {"size_distribution": "imported_distribution" if size_file else "monodisperse",
                      "spatial_order": "hard_core_random"},
        "particles": {"num_particles": 40},
        "size_distribution": {"size_file": str(size_file)} if size_file else {},
        "analysis": {"compute_g2": False, "compute_Sk": False},
        "output": {"path": str(tmp_path / "original"), "make_plots": False},
    }
    return config_from_mapping(raw)


def test_unseeded_run_records_effective_seed_without_changing_caller(tmp_path):
    config = config_for(tmp_path)
    first = run_generation(config)
    assert config.runtime.random_seed is None
    assert isinstance(first.config.runtime.random_seed, int)
    metadata = json.loads((first.config.output.path / "metadata.json").read_text())
    assert metadata["provenance"]["requested_random_seed"] is None
    assert metadata["provenance"]["effective_random_seed"] == first.config.runtime.random_seed
    assert metadata["provenance"]["software"]["version"]
    assert metadata["provenance"]["environment"]["dependencies"]["numpy"]
    assert yaml.safe_load((first.config.output.path / "config_input.yaml").read_text()).get("runtime") is None
    second = run_generation(first.config.output.path / "config_replay.yaml")
    assert np.array_equal(first.result.particle_set.positions, second.result.particle_set.positions)
    assert np.array_equal(first.result.particle_set.radii, second.result.particle_set.radii)


def test_moved_bundle_replays_without_original_files_and_preserves_prior_runs(tmp_path, monkeypatch):
    source = tmp_path / "sizes.csv"
    source.write_text("diameter\n200\n300\n400\n")
    config = config_for(tmp_path, source)
    first = run_generation(config)
    saved = (first.config.output.path / "particles_real_units.csv").read_bytes()
    portable = tmp_path / "relocated"
    with ZipFile(first.config.output.path / "run_bundle.zip") as archive:
        archive.extractall(portable)
    source.unlink()
    # The original archived input is also unavailable now.
    Path(first.config.size_distribution.size_file).unlink()
    other = tmp_path / "other-working-directory"
    other.mkdir()
    monkeypatch.chdir(other)
    replay_path = portable / "config_replay.yaml"
    replay = load_config(replay_path)
    assert Path(replay.size_distribution.size_file).is_relative_to(portable)
    assert replay.output.path == portable / "replay"
    replayed = run_generation(replay)
    assert np.array_equal(first.result.particle_set.positions, replayed.result.particle_set.positions)
    assert np.array_equal(first.result.particle_set.radii, replayed.result.particle_set.radii)
    repeated = run_generation(replay_path)
    assert repeated.config.output.path != replayed.config.output.path
    assert repeated.config.output.path.parent == portable
    assert (portable / "particles_real_units.csv").read_bytes() == saved
    manifest = json.loads((portable / "run_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert sha256_file(portable / name) == digest


def test_snapshot_is_exact_and_detects_corruption_even_without_csv(tmp_path):
    config = config_for(tmp_path)
    config.output.save_dimensionless_coordinates = False
    config.output.save_real_coordinates = False
    bundle = run_generation(config)
    restored = load_snapshot(bundle.config.output.path)
    assert np.array_equal(restored.positions, bundle.result.particle_set.positions)
    assert np.array_equal(restored.radii, bundle.result.particle_set.radii)
    assert np.array_equal(restored.box_lengths, bundle.result.particle_set.box_lengths)
    assert not (bundle.config.output.path / "particles_real_units.csv").exists()
    snapshot = bundle.config.output.path / "particles_snapshot.npz"
    snapshot.write_bytes(snapshot.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(bundle.config.output.path)


def test_replay_detects_modified_input_file(tmp_path):
    source = tmp_path / "sizes.txt"
    source.write_text("200\n300\n400\n")
    bundle = run_generation(config_for(tmp_path, source))
    replay = load_config(bundle.config.output.path / "config_replay.yaml")
    Path(replay.size_distribution.size_file).write_text("200\n200\n200\n")
    with pytest.raises(ValueError, match="checksum"):
        run_generation(replay)


@pytest.mark.parametrize("seed", [-1, 1.5, True, "123"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="random_seed"):
        config_from_mapping({"physical": {"mean_diameter": "200 nm", "packing_fraction": .1},
                             "runtime": {"random_seed": seed}})
