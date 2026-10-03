"""Run provenance and portable input snapshots."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(config):
    """Keep caller settings intact; freeze the seed and file input before sampling."""
    effective = deepcopy(config)
    effective.runtime.__post_init__()
    requested_seed = effective.runtime.random_seed
    if requested_seed is None:
        effective.runtime.random_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    else:
        effective.runtime.random_seed = int(requested_seed)
    destination = Path(effective.output.path).expanduser().resolve()
    if effective.output.overwrite:
        destination.mkdir(parents=True, exist_ok=True)
    else:
        original = destination
        while True:
            try:
                destination.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                destination = original.with_name(f"{original.name}_run_{uuid4().hex[:12]}")
    effective.output.path = destination
    inputs = []
    if effective.size_distribution.size_file:
        source = Path(effective.size_distribution.size_file).expanduser().resolve()
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if effective.size_distribution.file_sha256 is not None and effective.size_distribution.file_sha256 != digest:
            raise ValueError("Custom size file SHA-256 checksum does not match the replay configuration")
        relative = Path("inputs") / f"particle_sizes_{digest[:12]}{source.suffix}"
        archived = destination / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(content)
        effective.size_distribution.size_file = str(archived)
        effective.size_distribution.file_sha256 = digest
        inputs.append({"role": "size_distribution", "source_name": source.name,
                       "path": relative.as_posix(), "sha256": digest})
    return effective, {"requested_random_seed": requested_seed,
                       "effective_random_seed": effective.runtime.random_seed,
                       "rng": "numpy.default_rng/PCG64; separate size and position streams initialized with the recorded seed",
                       "inputs": inputs}


def environment_provenance() -> dict:
    dependencies = {}
    for name in ("numpy", "scipy", "PyYAML", "h5py", "matplotlib", "streamlit", "pyvista"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    try:
        package_version = version("spherepackgen")
    except PackageNotFoundError:
        package_version = "1.0.0"
    commit, dirty = None, None
    root = package.parent.parent
    if (root / ".git").exists():
        try:
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                    text=True, check=True, timeout=2, **options).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                        capture_output=True, text=True, check=True, timeout=2, **options).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {"software": {"name": "spherepackgen", "version": package_version,
                         "source_sha256": digest.hexdigest(), "git_commit": commit, "git_dirty": dirty},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "machine": platform.machine(), "dependencies": dependencies}}
