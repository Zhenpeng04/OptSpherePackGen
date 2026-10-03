import json
from pathlib import Path

import pytest
import yaml
import time
from streamlit.testing.v1 import AppTest

from spherepackgen.gui.streamlit_app import _build_config, _estimate_particle_count_from_preview
from test_gui_preview import _base_gui_form_data


def wait_for_gui(app, timeout=60):
    deadline = time.monotonic() + timeout
    while "gui_job" in app.session_state:
        if time.monotonic() >= deadline:
            raise AssertionError("GUI generation did not finish within the test deadline")
        time.sleep(.1)
        app.run()
    assert not app.exception
    return app


def test_gui_config_exposes_fixed_length_and_depth():
    data = _base_gui_form_data(0.2)
    data.update(lateral_length_um=2.0, medium_thickness_um=0.5)
    config = _build_config(data)
    assert config["domain"] == {"type": "periodic_box", "length": "2.0 um", "depth": "0.5 um"}
    assert "medium_thickness" not in config["physical"]


def test_preview_count_uses_rectangular_volume():
    inputs = dict(size_distribution="monodisperse", spatial_order="hard_core_random", lattice_type="FCC",
        mean_diameter_um=0.2, packing_fraction=0.15, medium_thickness_um=0.8, lateral_length_um=1.6,
        random_seed=123, cv_radius=0.0, min_radius_factor=None, max_radius_factor=None,
        continuous_distribution="lognormal", gamma_shape=2.5, weibull_shape=2.5,
        custom_size_file=None, custom_file_values="diameter", custom_file_column="")
    n, volume, error = _estimate_particle_count_from_preview(**inputs)
    assert error is None
    assert n == round(0.15 * (1.6/0.2)**2 * (0.8/0.2) / volume)


def test_gui_generates_and_displays_rectangular_results(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/spherepackgen/gui/streamlit_app.py"
    app = AppTest.from_file(str(source), default_timeout=45).run()
    assert not app.exception
    assert next(item for item in app.selectbox if item.label == "Spatial order / structure type").value == "hard_core_random"
    values = {"Mean particle diameter (um)": 0.2, "Target packing fraction": 0.15,
              "Lateral length Lx = Ly (um)": 1.2, "Depth Lz (um)": 0.6}
    for item in app.number_input:
        if item.label in values:
            item.set_value(values[item.label])
    app.session_state["gui_output_path"] = str(tmp_path / "gui_smoke")
    app.run()
    assert not app.exception
    generate = next(button for button in app.button if button.label == "Generate")
    assert not generate.disabled
    generate.click().run()
    wait_for_gui(app)
    assert not app.exception
    report = json.loads((tmp_path / "gui_smoke/validation_report.json").read_text())
    assert report["passed"], report["errors"]
    metadata = json.loads((tmp_path / "gui_smoke/metadata.json").read_text())
    assert metadata["resolved_parameters"]["box_lengths_m"] == pytest.approx([1.2e-6, 1.2e-6, 0.6e-6])
    config = yaml.safe_load((tmp_path / "gui_smoke/config_replay.yaml").read_text())
    assert config["domain"]["type"] == "periodic_box"
    assert any(metric.label == "Box Lx × Ly × Lz" for metric in app.metric)
    assert (tmp_path / "gui_smoke/figures/packing_perspective.png").exists()
    assert any(item.value == "3D Packing View" for item in app.get("subheader"))
    assert any("Preview shows" in item.value for item in app.caption)
    assert not (tmp_path / "gui_smoke/particles.h5").exists()
