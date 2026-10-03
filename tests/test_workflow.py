from math import pi, sqrt

import pytest

from spherepackgen.api import load_config, run_generation
from spherepackgen.domain.enums import (
    LatticeType,
    SpatialOrderClass,
    lattice_basis_count,
    lattice_max_packing_fraction,
)
from spherepackgen.generators.registry import GENERATORS, default_generator_name
from spherepackgen.utils.exceptions import GenerationError


def test_fcc_example_runs(tmp_path):
    config = load_config("configs/examples/fcc.yaml")
    config.output.path = tmp_path / "fcc"
    config.output.make_plots = False
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.particle_set.n_particles == 4 * 4**3
    assert (config.output.path / "particles_dimensionless.csv").exists()
    assert (config.output.path / "metadata.json").exists()


def test_diamond_cubic_crystal_runs(tmp_path):
    config = load_config("configs/examples/fcc.yaml")
    config.output.path = tmp_path / "diamond_cubic"
    config.output.make_plots = False
    config.output.formats = ["csv", "json"]
    config.analysis.compute_g2 = False
    config.analysis.compute_Sk = False
    config.physical.packing_fraction = 0.20
    config.algorithm.parameters["lattice_type"] = "DIAMOND_CUBIC"
    config.algorithm.parameters["unit_cells"] = 2

    bundle = run_generation(config)

    assert bundle.validation.passed
    assert bundle.result.particle_set.n_particles == 8 * 2**3
    assert bundle.result.diagnostics["lattice_type"] == "DIAMOND_CUBIC"


def test_lattice_helpers_include_diamond_cubic():
    assert LatticeType.from_value("diamond cubic") == LatticeType.DIAMOND_CUBIC
    assert lattice_basis_count("diamond") == 8
    assert lattice_max_packing_fraction("DIAMOND_CUBIC") == pytest.approx(sqrt(3.0) * pi / 16.0)


def test_hard_core_random_example_runs(tmp_path):
    config = load_config("configs/examples/hard_core_random.yaml")
    config.output.path = tmp_path / "poisson_disk"
    config.output.make_plots = False
    config.particles.num_particles = 40
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.generator_name == "poisson_disk"
    assert bundle.validation.metrics["overlap_pair_count"] == 0
    assert (config.output.path / "analysis" / "g2.csv").exists()


def test_hard_core_random_default_generator_thresholds():
    config = load_config("configs/examples/hard_core_random.yaml")
    assert "poisson_disk" in GENERATORS
    assert "force_biased" in GENERATORS
    assert "metropolis" in GENERATORS
    assert "soft_compression" not in GENERATORS
    assert "dense" not in GENERATORS
    for phi, expected in [
        (0.18, "poisson_disk"),
        (0.180001, "rsa"),
        (0.34, "rsa"),
        (0.340001, "force_biased"),
    ]:
        config.physical.packing_fraction = phi
        assert default_generator_name(config) == expected










def test_force_biased_generator_runs(tmp_path):
    config = load_config("configs/examples/force_biased.yaml")
    config.output.path = tmp_path / "force_biased"
    config.output.make_plots = False
    config.output.formats = ["csv", "json"]
    config.particles.num_particles = 32
    config.physical.packing_fraction = 0.36
    config.algorithm.parameters["stages"] = 10
    config.algorithm.parameters["relaxation_steps_per_stage"] = 160
    config.algorithm.parameters["final_cleanup_steps"] = 240
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.generator_name == "force_biased"
    assert bundle.validation.metrics["overlap_pair_count"] == 0
    assert abs(bundle.result.diagnostics["outer_diameter_ratio_initial"] - 1.2 ** (1.0 / 3.0)) < 1.0e-9
    assert bundle.result.diagnostics["force_backend"] == "vectorized_verlet_pair_cache"
    assert bundle.result.diagnostics["intermediate_tolerance_overlap"] >= bundle.result.diagnostics["final_tolerance_overlap"]


def test_lubachevsky_stillinger_generator_runs(tmp_path):
    config = load_config("configs/examples/lubachevsky_stillinger.yaml")
    config.output.path = tmp_path / "lubachevsky_stillinger"
    config.output.make_plots = False
    config.output.formats = ["csv", "json"]
    config.analysis.compute_g2 = False
    config.analysis.compute_Sk = False
    config.particles.num_particles = 12
    config.physical.packing_fraction = 0.18
    config.algorithm.parameters.update(
        {
            "initial_radius_scale": 0.75,
            "compression_rate": 1.0e-2,
            "max_events": 2000,
            "event_backend": "verlet_heap",
            "neighbor_rebuild_fraction": 0.8,
            "final_cleanup_steps": 50,
        }
    )
    assert "lubachevsky_stillinger" in GENERATORS
    assert "ls" in GENERATORS
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.generator_name == "lubachevsky_stillinger"
    assert bundle.result.diagnostics["events"] > 0
    assert bundle.result.diagnostics["event_search_backend"] == "verlet_heap"
    assert bundle.result.diagnostics["neighbor_rebuilds"] > 0
    assert bundle.result.diagnostics["neighbor_rebuild_fraction"] == pytest.approx(0.8)
    assert bundle.result.diagnostics["pre_cleanup_overlap_pair_count"] == 0
    assert bundle.validation.metrics["overlap_pair_count"] == 0


def test_lubachevsky_stillinger_auto_backend_uses_exhaustive_for_small_systems(tmp_path):
    config = load_config("configs/examples/lubachevsky_stillinger.yaml")
    config.output.path = tmp_path / "lubachevsky_stillinger_auto"
    config.output.make_plots = False
    config.output.formats = ["csv", "json"]
    config.analysis.compute_g2 = False
    config.analysis.compute_Sk = False
    config.particles.num_particles = 2
    config.physical.packing_fraction = 0.18
    config.algorithm.parameters.update(
        {
            "initial_radius_scale": 0.75,
            "compression_rate": 1.0e-2,
            "max_events": 1000,
            "event_backend": "auto",
            "final_cleanup_steps": 25,
        }
    )
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.diagnostics["event_backend_requested"] == "auto"
    assert bundle.result.diagnostics["auto_exhaustive_particle_threshold"] == 2
    assert bundle.result.diagnostics["event_backend_selected"] == "exhaustive_pair_scan"
    assert bundle.result.diagnostics["event_search_backend"] == "exhaustive_pair_scan"
    assert bundle.result.diagnostics["pre_cleanup_overlap_pair_count"] == 0


def test_lubachevsky_stillinger_auto_backend_uses_verlet_above_threshold(tmp_path):
    config = load_config("configs/examples/lubachevsky_stillinger.yaml")
    config.output.path = tmp_path / "lubachevsky_stillinger_auto_verlet"
    config.output.make_plots = False
    config.output.formats = ["csv", "json"]
    config.analysis.compute_g2 = False
    config.analysis.compute_Sk = False
    config.particles.num_particles = 12
    config.physical.packing_fraction = 0.18
    config.algorithm.parameters.update(
        {
            "initial_radius_scale": 0.75,
            "compression_rate": 1.0e-2,
            "max_events": 2000,
            "event_backend": "auto",
            "final_cleanup_steps": 25,
        }
    )
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.diagnostics["event_backend_requested"] == "auto"
    assert bundle.result.diagnostics["auto_exhaustive_particle_threshold"] == 2
    assert bundle.result.diagnostics["event_backend_selected"] == "verlet_heap"
    assert bundle.result.diagnostics["event_search_backend"] == "verlet_heap"
    assert bundle.result.diagnostics["pre_cleanup_overlap_pair_count"] == 0


def test_lubachevsky_stillinger_auto_initialization_uses_force_biased_warm_start(tmp_path):
    config = load_config("configs/examples/lubachevsky_stillinger.yaml")
    config.output.path = tmp_path / "lubachevsky_stillinger_warm_start"
    config.output.make_plots = False
    config.output.formats = ["csv", "json"]
    config.analysis.compute_g2 = False
    config.analysis.compute_Sk = False
    config.particles.num_particles = 4
    config.physical.packing_fraction = 0.60
    config.algorithm.parameters.update(
        {
            "initialization": "auto",
            "warm_start_packing_fraction": 0.50,
            "warm_start_stages": 4,
            "warm_start_relaxation_steps_per_stage": 80,
            "warm_start_final_cleanup_steps": 200,
            "compression_rate": 0.5,
            "max_events": 5000,
            "event_backend": "auto",
            "final_cleanup_steps": 25,
        }
    )
    bundle = run_generation(config)
    assert bundle.validation.passed
    diagnostics = bundle.result.diagnostics
    assert diagnostics["initialization_requested"] == "auto"
    assert diagnostics["initialization"] == "force_biased_warm_start"
    assert diagnostics["warm_start_packing_fraction"] == pytest.approx(0.50)
    assert diagnostics["initialization_diagnostics"]["warm_start_generator"] == "force_biased"
    assert diagnostics["event_backend_selected"] == "verlet_heap"
    assert diagnostics["pre_cleanup_overlap_pair_count"] == 0
    assert diagnostics["final_overlap_pair_count"] == 0
    assert bundle.validation.metrics["overlap_pair_count"] == 0










def test_low_phi_poisson_disk_spreads_across_periodic_box(tmp_path):
    config = load_config("configs/examples/hard_core_random.yaml")
    config.output.path = tmp_path / "poisson_disk_spread"
    config.output.make_plots = False
    config.output.formats = ["csv", "json"]
    config.particles.num_particles = 120
    config.physical.packing_fraction = 0.01
    config.runtime.random_seed = 12345
    bundle = run_generation(config)
    positions = bundle.result.particle_set.positions
    box = bundle.result.particle_set.box_length
    spans = positions.max(axis=0) - positions.min(axis=0)
    assert bundle.result.generator_name == "poisson_disk"
    assert bundle.validation.passed
    assert bundle.validation.metrics["overlap_pair_count"] == 0
    assert (spans / box > 0.85).all()


def test_hard_sphere_random_alias_maps_to_hard_core_random():
    assert SpatialOrderClass.from_value("hard_sphere_random") == SpatialOrderClass.HARD_CORE_RANDOM


def test_overlapping_random_boolean_example_allows_nominal_fraction_above_one(tmp_path):
    config = load_config("configs/examples/overlapping_random_boolean.yaml")
    config.output.path = tmp_path / "overlapping"
    config.output.make_plots = False
    bundle = run_generation(config)
    assert bundle.validation.passed
    assert bundle.result.generator_name == "marked_poisson_boolean"
    assert config.validation.require_non_overlap is False
    assert bundle.validation.metrics["packing_fraction_actual"] > 1.0
    assert abs(bundle.validation.metrics["target_covered_volume_fraction"] - 0.70) < 1.0e-12
    assert abs(bundle.validation.metrics["expected_boolean_covered_fraction"] - 0.70) < 1.0e-8
    assert bundle.validation.metrics["overlap_pair_count"] is None
    assert bundle.validation.metrics["overlap_check_skipped"] is True
    assert bundle.validation.metrics["overlap_check_reason"] == "overlaps_allowed_by_marked_poisson_boolean_model"


def test_cli_run(tmp_path, monkeypatch, capsys):
    config = load_config("configs/examples/fcc.yaml")
    config.output.path = tmp_path / "cli_fcc"
    # The CLI reads from disk, so create a compact temporary config.
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
project:
  name: cli_fcc
structure:
  size_distribution: monodisperse
  spatial_order: periodic_crystal
physical:
  mean_diameter: 300 nm
  packing_fraction: 0.50
  medium_thickness: 2.0 um
domain:
  type: periodic_cube
  box_size: auto
particles:
  num_particles: auto
size_distribution:
  type: monodisperse
algorithm:
  name: default
  parameters:
    lattice_type: FCC
    unit_cells: 2
analysis:
  compute_g2: true
  compute_Sk: true
  bins: 20
  k_max_index: 2
validation:
  tolerance_phi: 1.0e-8
  tolerance_overlap: 1.0e-5
output:
  path: PLACEHOLDER
  coordinate_unit: nm
  formats: [csv, json]
  make_plots: false
runtime:
  random_seed: 1
""".replace("PLACEHOLDER", str(tmp_path / "cli_fcc").replace("\\", "/")),
        encoding="utf-8",
    )
    from spherepackgen.cli import main

    code = main(["run", str(cfg_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "validation_passed" in out
    assert (tmp_path / "cli_fcc" / "metadata.json").exists()
