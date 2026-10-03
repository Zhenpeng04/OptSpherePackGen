"""Launch the packaged Streamlit application from any working directory."""
from __future__ import annotations

from importlib.resources import as_file, files
import os
from pathlib import Path
import subprocess
import sys


def launch_gui(workdir: str | Path, port: int = 8501, headless: bool = False) -> int:
    root = Path(workdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["SPHEREPACKGEN_WORKSPACE"] = str(root)
    resource = files("spherepackgen.gui").joinpath("streamlit_app.py")
    with as_file(resource) as app:
        command = [sys.executable, "-m", "streamlit", "run", str(app),
                   "--server.address", "127.0.0.1", "--server.port", str(port),
                   "--server.headless", str(headless).lower(),
                   "--browser.gatherUsageStats", "false"]
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        return subprocess.call(command, cwd=root, env=environment, **options)
