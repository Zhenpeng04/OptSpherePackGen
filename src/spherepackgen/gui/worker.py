"""GUI worker supervisor: generation runs in a separately bounded process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import subprocess
import signal
import traceback
import time

from spherepackgen.gui.jobs import terminate_tree


def report(directory, stage, message=None):
    progress_path = directory / "gui_progress.json"
    temporary = progress_path.with_suffix(".tmp")
    payload = json.dumps({"stage": stage, "message": message})
    temporary.write_text(payload, encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(progress_path)
            return
        except PermissionError:
            # Windows readers/antivirus can briefly deny rename while polling.
            time.sleep(.01 * (attempt + 1))
    # Progress is advisory; a transient polling lock must not fail generation.
    try:
        progress_path.write_text(payload, encoding="utf-8")
    except OSError:
        pass


def execute(config_path):
    from spherepackgen import load_config
    from spherepackgen.workflows.generation import run_generation
    try:
        run_generation(load_config(config_path), progress=lambda stage: report(config_path.parent, stage))
    except Exception as exc:
        report(config_path.parent, "failed", str(exc))
        traceback.print_exc()
        return 1
    return 0


def main():
    config_path = Path(sys.argv[1])
    if sys.argv[2] == "--execute":
        return execute(config_path)
    timeout_s = int(sys.argv[2])
    report(config_path.parent, "starting")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen([sys.executable, "-m", "spherepackgen.gui.worker", str(config_path), "--execute"], **options)
    if os.name != "nt":
        # Cancellation of the supervisor also stops the separate child group.
        def stop_child(signum, frame):
            terminate_tree(process.pid)
            process.wait(timeout=5)
            sys.exit(130)
        signal.signal(signal.SIGTERM, stop_child)
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        terminate_tree(process.pid)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        report(config_path.parent, "failed", f"Run exceeded the {timeout_s} second time limit.")
        return 124


if __name__ == "__main__":
    sys.exit(main())
