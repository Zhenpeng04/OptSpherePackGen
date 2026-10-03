"""Public Python API."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from spherepackgen.config.loader import load_config
from spherepackgen.domain.result import ResultBundle
from spherepackgen.workflows.generation import run_generation as _run_generation
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.provenance import sha256_file


def run_generation(config_or_path) -> ResultBundle:
    """Run a complete generation workflow from a config object or config path."""
    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    else:
        config = config_or_path
    return _run_generation(config)


def load_snapshot(result_directory: str | Path) -> ParticleSet:
    """Load exact dimensionless geometry without rerunning an algorithm.

    The snapshot checksum is verified before loading; NPZ pickle data is disabled.
    """
    directory = Path(result_directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    snapshot = metadata["provenance"]["snapshot"]
    path = (directory / snapshot["path"]).resolve()
    if not path.is_relative_to(directory) or sha256_file(path) != snapshot["sha256"]:
        raise ValueError("Snapshot path or SHA-256 checksum is invalid")
    with np.load(path, allow_pickle=False) as values:
        return ParticleSet(values["positions"], values["radii"], values["box_lengths"],
                           values["species_id"], values["material_id"])
