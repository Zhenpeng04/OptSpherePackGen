"""Owned subprocesses for cancellable GUI generation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4


def allocate_run_directory(requested: Path) -> Path:
    """Keep a first custom destination; never reuse an existing destination."""
    requested = requested.resolve()
    if requested.exists():
        requested = requested.with_name(f"{requested.name}_run_{uuid4().hex[:12]}")
    # Exclusive creation protects simultaneous sessions selecting the same path.
    try:
        requested.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return allocate_run_directory(requested)
    return requested


def start_job(config_path: Path, timeout_s: int):
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (config_path.parent / "gui_worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "spherepackgen.gui.worker", str(config_path), str(timeout_s)],
            cwd=config_path.parent, env=environment, stdout=log, stderr=log, **options,
        )
    return {"process": process, "directory": str(config_path.parent), "started": time.time(), "timeout_s": timeout_s}


def terminate_tree(pid: int):
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        if result.returncode:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def cancel_job(job):
    process = job["process"]
    if process.poll() is None:
        terminate_tree(process.pid)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    directory = Path(job["directory"])
    (directory / "gui_progress.json").write_text(json.dumps({"stage": "cancelled", "message": "Run cancelled."}), encoding="utf-8")


def read_progress(directory: Path):
    try:
        return json.loads((directory / "gui_progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"stage": "starting"}
