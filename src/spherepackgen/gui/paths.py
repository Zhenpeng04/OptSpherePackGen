"""User-owned GUI workspace, independent of the package installation path."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4


def workspace_root() -> Path:
    return Path(os.environ.get("SPHEREPACKGEN_WORKSPACE", Path.cwd())).expanduser().resolve()


def remembered_runs(workspace: Path) -> list[Path]:
    try:
        values = json.loads((workspace / ".spherepackgen/history.json").read_text(encoding="utf-8"))
        return [Path(value) for value in values if isinstance(value, str)] if isinstance(values, list) else []
    except (OSError, ValueError):
        return []


def remember_run(workspace: Path, directory: Path) -> None:
    path = workspace / ".spherepackgen/history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [str(directory.resolve())] + [str(p) for p in remembered_runs(workspace) if p != directory.resolve()]
    temporary = path.with_name(f"history_{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(values[:1000], indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
