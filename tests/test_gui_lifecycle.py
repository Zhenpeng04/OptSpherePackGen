import json
from pathlib import Path
import time

from streamlit.testing.v1 import AppTest

from test_rectangular_gui import wait_for_gui

SOURCE = Path(__file__).resolve().parents[1] / "src/spherepackgen/gui/streamlit_app.py"


def control(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def small_app(tmp_path, order="hard_core_random"):
    app = AppTest.from_file(str(SOURCE), default_timeout=30).run()
    control(app, "selectbox", "Particle-size distribution").set_value("monodisperse").run()
    control(app, "selectbox", "Spatial order / structure type").set_value(order).run()
    for label, value in {"Mean particle diameter (um)": .2, "Lateral length Lx = Ly (um)": 1.2,
                         "Depth Lz (um)": .6}.items():
        control(app, "number_input", label).set_value(value)
    if order != "overlapping_random":
        control(app, "number_input", "Target packing fraction").set_value(.15)
    for label in ("Compute g2(r)", "Compute S(k)", "Generate analysis plots and 3D views"):
        control(app, "checkbox", label).set_value(False)
    app.session_state["gui_output_path"] = str(tmp_path / "case")
    app.run()
    assert not app.exception
    return app


def test_exact_count_saved_result_and_run_isolation(tmp_path):
    app = small_app(tmp_path)
    control(app, "selectbox", "Particle-size distribution").set_value("continuous_polydisperse").run()
    control(app, "selectbox", "Continuous distribution").set_value("gamma").run()
    preview = control(app, "number_input", "Particle count").value
    control(app, "button", "Generate").click().run()
    wait_for_gui(app)
    first = Path(app.session_state["last_generated_output_path"])
    metadata = json.loads((first / "metadata.json").read_text())
    assert metadata["resolved_parameters"]["n_particles"] == preview
    csv = (first / "particles_real_units.csv").read_bytes()
    control(app, "number_input", "Random seed").set_value(98765).run()
    assert any(metric.label == "Validation" for metric in app.metric)
    assert len(app.get("download_button")) == 4
    assert any("Current settings have not been generated" in item.value for item in app.info)
    # An old unlisted image must never appear in this run's result display.
    (first / "figures").mkdir(exist_ok=True)
    (first / "figures/packing_perspective.png").write_bytes(b"stale image")
    app.run()
    assert not app.exception
    assert not any(item.value == "3D Packing View" for item in app.get("subheader"))
    control(app, "button", "Generate").click().run()
    wait_for_gui(app)
    second = Path(app.session_state["last_generated_output_path"])
    assert second != first
    assert (first / "particles_real_units.csv").read_bytes() == csv
    assert not (second / "figures/packing_perspective.png").exists()


def test_invalid_preview_and_thin_box_block_before_generation(tmp_path):
    app = small_app(tmp_path)
    control(app, "number_input", "Depth Lz (um)").set_value(.1).run()
    assert control(app, "button", "Generate").disabled
    assert any("own periodic image" in item.value for item in app.error)
    control(app, "number_input", "Depth Lz (um)").set_value(.6)
    control(app, "selectbox", "Particle-size distribution").set_value("quasi_monodisperse").run()
    assert "periodic_crystal" not in control(app, "selectbox", "Spatial order / structure type").options
    control(app, "number_input", "Min radius factor").set_value(1.2)
    control(app, "number_input", "Max radius factor").set_value(.8).run()
    assert not app.exception
    assert control(app, "button", "Generate").disabled


def test_manual_count_and_crystal_dimensions(tmp_path):
    app = small_app(tmp_path)
    control(app, "checkbox", "Automatically compute particle count").set_value(False).run()
    control(app, "number_input", "Particle count").set_value(1).run()
    assert control(app, "button", "Generate").disabled
    assert any("inconsistent" in item.value for item in app.error)
    control(app, "checkbox", "Automatically compute particle count").set_value(True).run()
    assert control(app, "selectbox", "Algorithm").options == [
        "Automatic", "variable-radius Poisson-disk", "RSA", "force-biased relaxation", "Lubachevsky-Stillinger"
    ]
    control(app, "selectbox", "Spatial order / structure type").set_value("periodic_crystal").run()
    assert "HCP" not in control(app, "selectbox", "Crystal lattice").options
    control(app, "number_input", "Crystal cells nx = ny").set_value(3)
    control(app, "number_input", "Crystal cells nz").set_value(2).run()
    control(app, "button", "Apply compatible crystal dimensions").click().run()
    assert not app.exception
    assert not control(app, "button", "Generate").disabled
    assert control(app, "number_input", "Particle count").value == 72




def test_cancel_and_timeout_keep_previous_results(tmp_path):
    app = small_app(tmp_path)
    control(app, "button", "Generate").click().run()
    wait_for_gui(app)
    previous = app.session_state["last_generated_output_path"]
    control(app, "selectbox", "Algorithm").set_value("lubachevsky_stillinger").run()
    control(app, "number_input", "LS max collision events").set_value(10_000_000).run()
    control(app, "button", "Generate").click().run()
    process = app.session_state["gui_job"]["process"]
    control(app, "button", "Cancel run").click().run()
    assert "gui_job" not in app.session_state
    assert process.poll() is not None
    assert app.session_state["last_generated_output_path"] == previous
    assert any("cancelled" in item.value for item in app.error)

    # Exercise the real worker timeout with the saved slow LS configuration.
    from spherepackgen.gui.jobs import start_job, read_progress, cancel_job
    configuration = next(path for path in tmp_path.glob("case_run_*/_gui_config.yaml"))
    job = start_job(configuration, timeout_s=1)
    try:
        job["process"].wait(timeout=20)
        progress = read_progress(configuration.parent)
        assert progress["stage"] == "failed"
        assert "time limit" in progress["message"]
    finally:
        if job["process"].poll() is None:
            cancel_job(job)
