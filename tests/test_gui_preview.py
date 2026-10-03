import numpy as np
import pytest
from pathlib import Path

from spherepackgen.gui.streamlit_app import (
    _build_config,
    _force_biased_default_parameters,
    DEFAULT_G2_BINS,
    DEFAULT_PACKING_FRACTION_TOLERANCE,
    DEFAULT_SK_MAX_INDEX,
    CRYSTAL_LATTICE_OPTIONS,
    SPATIAL_ORDER_OPTIONS,
    _crystal_lattice_label,
    _preflight_checks,
    _preview_histogram_settings,
    _spatial_order_label,
    _timestamped_output_path,
)

GUI_SOURCE = Path(__file__).resolve().parents[1] / "src" / "spherepackgen" / "gui" / "streamlit_app.py"

def _base_gui_form_data(packing_fraction: float) -> dict:
    return {
        "project_name": "gui_case",
        "description": "",
        "output_path": "results/gui_case",
        "size_distribution": "monodisperse",
        "spatial_order": "hard_core_random",
        "algorithm_name": "default",
        "effective_algorithm_name": "force_biased",
        "mean_diameter_um": 0.2,
        "packing_fraction": packing_fraction,
        "covered_volume_fraction": 0.70,
        "medium_thickness_um": 1.0,
        "auto_particles": True,
        "num_particles": 24,
        "make_plots": False,
        "save_hdf5": False,
        "cv_radius": 0.03,
        "min_radius_factor": 0.90,
        "max_radius_factor": 1.10,
        "continuous_distribution": "lognormal",
        "gamma_shape": 2.5,
        "weibull_shape": 2.5,
        "custom_size_file_path": "",
        "custom_file_values": "diameter",
        "custom_file_column": "",
        "lattice_type": "FCC",
        "max_attempts_per_particle": 10000,
        "poisson_disk_candidates_per_particle": 30,
        "ls_initial_radius_scale": 0.70,
        "ls_compression_rate": 1.0e-3,
        "ls_velocity_scale": 1.0,
        "ls_max_events": 25000,
        "ls_mass_mode": "equal",
        "ls_event_backend": "verlet_heap",
        "ls_neighbor_skin_fraction": 1.0,
        "ls_neighbor_velocity_safety_factor": 2.0,
        "ls_max_heap_factor": 8.0,
        "ls_final_cleanup_steps": 200,
        "force_biased_final_cleanup_steps": 400,
        "compute_g2": False,
        "compute_sk": False,
        "bins": 80,
        "k_max_index": 5,
        "tolerance_phi": 1.0e-3,
        "tolerance_overlap": 1.0e-5,
        "random_seed": 12345,
    }

def test_particle_size_preview_histogram_uses_percent_weights():
    diameters = np.array([0.25, 0.30, 0.35, 0.40], dtype=float)
    bins, weights = _preview_histogram_settings(diameters)

    assert bins == 32
    assert weights.shape == diameters.shape
    assert np.sum(weights) == pytest.approx(100.0)

def test_particle_size_preview_monodisperse_histogram_is_single_percent_bin():
    diameters = np.ones(2000, dtype=float) * 0.30
    bins, weights = _preview_histogram_settings(diameters)

    assert bins == 1
    assert np.sum(weights) == pytest.approx(100.0)


def test_gui_spatial_order_options_use_simplified_modes():
    assert SPATIAL_ORDER_OPTIONS == [
        "periodic_crystal",
        "hard_core_random",
        "overlapping_random",
    ]


def test_gui_labels_hard_core_random_as_non_overlapping_random():
    assert _spatial_order_label("hard_core_random") == "Non-overlapping random"


def test_gui_crystal_lattice_options_include_diamond_cubic():
    assert CRYSTAL_LATTICE_OPTIONS == ["FCC", "BCC", "SC", "HCP", "DIAMOND_CUBIC"]
    assert _crystal_lattice_label("DIAMOND_CUBIC") == "diamond cubic"


def test_timestamped_output_path_keeps_project_slug():
    assert _timestamped_output_path("My Project", "20260705_150000") == "results/My_Project_20260705_150000"


def test_gui_analysis_defaults_are_fixed_for_simplified_mode():
    assert DEFAULT_G2_BINS == 100
    assert DEFAULT_SK_MAX_INDEX == 20
    assert DEFAULT_PACKING_FRACTION_TOLERANCE == pytest.approx(0.001)


def test_preflight_rejects_invalid_radius_range():
    data = _minimal_preflight_data()
    data["min_radius_factor"] = 2.0
    data["max_radius_factor"] = 1.0

    errors, warnings = _preflight_checks(data, estimated_particle_count=100, particle_estimate_error=None)

    assert any("Min radius factor" in item for item in errors)
    assert warnings == []




def test_preflight_warns_large_particle_count():
    data = _minimal_preflight_data()

    errors, warnings = _preflight_checks(data, estimated_particle_count=30000, particle_estimate_error=None)

    assert errors == []
    assert any("large" in item for item in warnings)


def test_preflight_rejects_crystal_above_lattice_packing_limit():
    data = _minimal_preflight_data()
    data["spatial_order"] = "periodic_crystal"
    data["effective_algorithm_name"] = "crystal"
    data["lattice_type"] = "DIAMOND_CUBIC"
    data["packing_fraction"] = 0.50

    errors, warnings = _preflight_checks(data, estimated_particle_count=100, particle_estimate_error=None)

    assert any("theoretical maximum" in item and "diamond cubic" in item for item in errors)
    assert warnings == []


def _minimal_preflight_data():
    return {
        "size_distribution": "monodisperse",
        "spatial_order": "hard_core_random",
        "effective_algorithm_name": "force_biased",
        "packing_fraction": 0.50,
        "covered_volume_fraction": None,
        "auto_particles": True,
        "num_particles": 100,
        "continuous_distribution": "lognormal",
        "custom_size_file_path": "",
        "cv_radius": 0.10,
        "min_radius_factor": 0.80,
        "max_radius_factor": 1.20,
        "lattice_type": "FCC",
    }
def test_force_biased_gui_defaults_switch_for_high_density():
    low = _force_biased_default_parameters(0.50)
    high = _force_biased_default_parameters(0.62)

    assert low["initial_radius_fraction"] == pytest.approx(0.35)
    assert low["relaxation_steps_per_stage"] == 400
    assert high["initial_radius_fraction"] == pytest.approx(0.65)
    assert high["outer_diameter_ratio"] == pytest.approx(1.08)
    assert high["relaxation_steps_per_stage"] == 800
    assert high["final_cleanup_steps"] == 2000


def test_gui_build_config_includes_force_biased_core_and_advanced_parameters():
    data = _base_gui_form_data(0.50)
    data.update(
        {
            "force_biased_initial_radius_fraction": 0.42,
            "force_biased_stages": 12,
            "force_biased_steps_per_stage": 180,
            "force_biased_contraction_rate": 2.0e-3,
            "force_biased_outer_diameter_ratio": 1.05,
            "force_biased_force_scaling_factor": 0.45,
            "force_biased_max_displacement_fraction": 0.30,
            "force_biased_intermediate_tolerance": 2.0e-6,
            "force_biased_verlet_skin_fraction": 0.40,
            "force_biased_final_cleanup_steps": 360,
        }
    )

    config = _build_config(data)
    params = config["algorithm"]["parameters"]

    assert params["initial_radius_fraction"] == pytest.approx(0.42)
    assert params["stages"] == 12
    assert params["relaxation_steps_per_stage"] == 180
    assert params["contraction_rate"] == pytest.approx(2.0e-3)
    assert params["outer_diameter_ratio"] == pytest.approx(1.05)
    assert params["force_scaling_factor"] == pytest.approx(0.45)
    assert params["max_displacement_fraction"] == pytest.approx(0.30)
    assert params["intermediate_tolerance_overlap"] == pytest.approx(2.0e-6)
    assert params["verlet_skin_fraction"] == pytest.approx(0.40)
    assert params["final_cleanup_steps"] == 360


def test_gui_build_config_outputs_hdf5_only_when_selected():
    default_config = _build_config(_base_gui_form_data(0.50))
    assert default_config["output"]["formats"] == ["csv", "json"]

    with_hdf5_data = _base_gui_form_data(0.50)
    with_hdf5_data["save_hdf5"] = True
    hdf5_config = _build_config(with_hdf5_data)
    assert hdf5_config["output"]["formats"] == ["csv", "json", "hdf5"]


def test_gui_build_config_uses_overlap_tolerance_default():
    config = _build_config(_base_gui_form_data(0.50))
    assert config["validation"]["tolerance_overlap"] == pytest.approx(1.0e-5)




def test_gui_widgets_use_versioned_fast_defaults():
    source = GUI_SOURCE.read_text(encoding="utf-8")

    assert 'key="make_plots_default_true_v1"' in source
    assert '"Generate analysis plots and 3D views"' in source
    assert 'key="save_hdf5_default_false_v3"' in source
    assert 'key="overlap_tolerance_default_1e_minus_5_v2"' in source
    assert 'st.checkbox("Save HDF5", value=False' in source
    assert 'DEFAULT_OVERLAP_TOLERANCE = 1.0e-5' in source
    assert 'with st.expander("Advanced force-biased controls")' in source
    assert '"Initial radius fraction"' in source
    assert '"Radius-growth stages"' in source
    assert '"Relaxation steps per stage"' in source
    assert '"Contraction rate"' in source


def test_gui_hides_developer_and_removed_dense_baselines():
    source = GUI_SOURCE.read_text(encoding="utf-8")

    assert '"soft_compression"' not in source
    assert '"Metropolis sweeps"' not in source
    assert '"metropolis"' not in source
