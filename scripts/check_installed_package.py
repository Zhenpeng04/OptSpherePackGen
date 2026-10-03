"""Exercise an installed distribution outside the checkout (also used by CI)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import URLError
from urllib.request import urlopen


def check():
    import spherepackgen
    import numpy as np
    import yaml
    from spherepackgen.gui.jobs import terminate_tree
    from streamlit.testing.v1 import AppTest

    origin = Path(spherepackgen.__file__).resolve()
    source_package = Path(__file__).resolve().parents[1] / "src/spherepackgen"
    if origin.is_relative_to(source_package):
        raise RuntimeError("This check must import a built distribution, not the source checkout")
    # Streamlit's Windows file watcher may retain a directory handle briefly.
    # Cleanup failures must not replace a meaningful smoke-test diagnostic.
    with TemporaryDirectory(prefix="spherepackgen-installed-", ignore_cleanup_errors=True) as temporary:
        # macOS resolves /var to /private/var; Windows runners may use an 8.3
        # temporary-directory alias. Compare canonical workspace paths.
        root = Path(temporary).resolve()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["SPHEREPACKGEN_WORKSPACE"] = str(root)
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

        def cli(*arguments):
            completed = subprocess.run([sys.executable, "-m", "spherepackgen.cli", *arguments],
                                       cwd=root, env=environment, capture_output=True, text=True,
                                       timeout=120, **options)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            return completed.stdout

        cli("examples", "--output", "examples")
        summary = json.loads(cli("run", "examples/fcc.yaml"))
        assert summary["export_ready"]
        assert (root / "results/fcc_example/run_bundle.zip").exists()
        snapshot = spherepackgen.load_snapshot(root / "results/fcc_example")
        assert snapshot.n_particles == 32
        replay = spherepackgen.run_generation(root / "results/fcc_example/config_replay.yaml")
        assert np.array_equal(snapshot.positions, replay.result.particle_set.positions)

        # Check the real launcher and local server in addition to AppTest.
        with socket.socket() as bound:
            bound.bind(("127.0.0.1", 0))
            port = bound.getsockname()[1]
        process_options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with (root / "gui-server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "spherepackgen.cli", "gui",
                                        "--workdir", str(root), "--port", str(port), "--headless"],
                                       cwd=root, env=environment, stdout=log, stderr=log, **process_options)
            try:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("GUI exited before becoming healthy")
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("Installed GUI did not become healthy within 30 seconds")
                    time.sleep(.1)
            finally:
                terminate_tree(process.pid)
                process.wait(timeout=15)

        previous_directory = Path.cwd()
        previous_workspace = os.environ.get("SPHEREPACKGEN_WORKSPACE")
        try:
            os.chdir(root)
            os.environ["SPHEREPACKGEN_WORKSPACE"] = str(root)
            app_path = origin.parent / "gui/streamlit_app.py"
            app = AppTest.from_file(str(app_path), default_timeout=30).run()

            def control(kind, label):
                return next(item for item in getattr(app, kind) if item.label == label)

            assert not app.exception
            control("selectbox", "Particle-size distribution").set_value("monodisperse").run()
            for label, value in {"Mean particle diameter (um)": .2, "Lateral length Lx = Ly (um)": 1.2,
                                 "Depth Lz (um)": .6, "Target packing fraction": .15}.items():
                control("number_input", label).set_value(value)
            for label in ("Compute g2(r)", "Compute S(k)", "Generate analysis plots and 3D views"):
                control("checkbox", label).set_value(False)
            app.session_state["gui_output_path"] = str(root / "custom/gui-result")
            app.run()
            control("button", "Generate").click().run()
            deadline = time.monotonic() + 30
            while "gui_job" in app.session_state and time.monotonic() < deadline:
                time.sleep(.1)
                app.run()
            assert not app.exception
            assert "gui_job" not in app.session_state
            result = Path(app.session_state["last_generated_output_path"])
            assert result.is_relative_to(root)
            assert len(app.get("download_button")) == 4
            configuration = yaml.safe_load((result / "config_replay.yaml").read_text())
            assert configuration["output"]["overwrite"] is False
            # A new GUI session reads persistent history, including custom paths.
            restarted = AppTest.from_file(str(app_path), default_timeout=30).run()
            assert not restarted.exception
            assert any(str(result.relative_to(root)) in item.options
                       for item in restarted.selectbox)
        finally:
            os.chdir(previous_directory)
            if previous_workspace is None:
                os.environ.pop("SPHEREPACKGEN_WORKSPACE", None)
            else:
                os.environ["SPHEREPACKGEN_WORKSPACE"] = previous_workspace
        print(json.dumps({"installed_package": str(origin), "cli": "passed", "replay": "passed",
                          "gui_server": "passed", "gui_generate_download_history": "passed"}, indent=2))


if __name__ == "__main__":
    check()
