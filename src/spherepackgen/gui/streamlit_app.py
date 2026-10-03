"""Streamlit prototype GUI for SpherePackGen."""

from __future__ import annotations

import os
import json
import re
import subprocess
import sys
import hashlib
import time
from copy import deepcopy
from datetime import datetime
from math import gamma as gamma_function
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Streamlit renders static images, including from worker threads.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import streamlit as st
import yaml

if __package__ in {None, ""}:
    # Allow direct execution during early prototyping.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spherepackgen import load_config, run_generation
from spherepackgen.domain.enums import lattice_basis_count, lattice_max_packing_fraction
from spherepackgen.generators.force_biased import (
    DEFAULT_OUTER_DIAMETER_RATIO,
    HIGH_DENSITY_FINAL_CLEANUP_STEPS,
    HIGH_DENSITY_INITIAL_RADIUS_FRACTION,
    HIGH_DENSITY_OUTER_DIAMETER_RATIO,
    HIGH_DENSITY_PHI_THRESHOLD,
    HIGH_DENSITY_RELAXATION_STEPS_PER_STAGE,
)
from spherepackgen.generators.radii import _truncated_normal, _validate_bounds
from spherepackgen.config.loader import config_from_mapping
from spherepackgen.workflows.resolve import resolve_parameters
from spherepackgen.gui.jobs import allocate_run_directory, start_job, cancel_job, read_progress
from spherepackgen.gui.paths import workspace_root, remembered_runs, remember_run


ROOT = workspace_root()

SIZE_DISTRIBUTION_OPTIONS = [
    ("continuous_polydisperse", "polydisperse"),
    ("monodisperse", "monodisperse"),
    ("quasi_monodisperse", "quasi_monodisperse"),
]

SPATIAL_ORDER_OPTIONS = [
    "periodic_crystal",
    "hard_core_random",
    "overlapping_random",
]

QUASI_MONODISPERSE_SPATIAL_ORDER_OPTIONS = ["hard_core_random", "overlapping_random"]
POLYDISPERSE_SPATIAL_ORDER_OPTIONS = ["hard_core_random", "overlapping_random"]

SPATIAL_ORDER_LABELS = {
    "periodic_crystal": "Periodic crystal",
    "hard_core_random": "Non-overlapping random",
    "overlapping_random": "Overlapping random",
}

DEFAULT_G2_BINS = 100
DEFAULT_SK_MAX_INDEX = 20
DEFAULT_PACKING_FRACTION_TOLERANCE = 1.0e-3
DEFAULT_OVERLAP_TOLERANCE = 1.0e-5
CRYSTAL_LATTICE_OPTIONS = ["FCC", "BCC", "SC", "HCP", "DIAMOND_CUBIC"]

CRYSTAL_LATTICE_LABELS = {
    "DIAMOND_CUBIC": "diamond cubic",
}

SIZE_PARAMETER_HELP = {
    "distribution": "Select the statistical model used to sample particle diameters before normalization to the requested mean diameter.",
    "cv": "Coefficient of variation of the particle radius: standard deviation divided by mean radius. Larger values produce broader size distributions.",
    "min_factor": "Lower radius cutoff relative to the mean radius. For example, 0.40 means the smallest radius is clipped to 40% of the mean radius.",
    "max_factor": "Upper radius cutoff relative to the mean radius. For example, 1.80 means the largest radius is clipped to 180% of the mean radius.",
    "gamma_shape": "Shape parameter of the gamma distribution. Larger values make the distribution narrower and more symmetric.",
    "weibull_shape": "Shape parameter of the Weibull/Rosin-Rammler distribution. Larger values make the distribution narrower.",
    "size_file": "Upload a CSV, TXT, or DAT file containing one particle-size column. Values can represent diameters or radii.",
    "file_values": "Specify whether the uploaded numeric values are particle diameters or particle radii.",
    "file_column": "Optional column name or zero-based column index. Leave blank to use the first numeric column.",
}


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "spherepackgen_case"


def _default_output_path(project_name: str) -> str:
    return f"results/{_slugify(project_name)}"


def _new_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _timestamped_output_path(project_name: str, timestamp: str) -> str:
    return f"{_default_output_path(project_name)}_{timestamp}"


def _result_history_options(limit: int = 30) -> list[str]:
    results_root = ROOT / "results"
    candidates = []
    directories = set(remembered_runs(ROOT))
    if results_root.exists():
        directories.update(results_root.iterdir())
    for child in directories:
        if not child.is_dir():
            continue
        has_result_artifact = any(
            (child / rel).exists()
            for rel in (
                "metadata.json",
                "validation_report.json",
                "_gui_config.yaml",
                "figures/packing_perspective.png",
            )
        )
        if has_result_artifact:
            candidates.append(child)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for path in candidates[:limit]]


def _size_distribution_label(value: str) -> str:
    labels = dict(SIZE_DISTRIBUTION_OPTIONS)
    return labels.get(value, value)

def _spatial_order_label(value: str) -> str:
    return SPATIAL_ORDER_LABELS.get(value, value)


def _crystal_lattice_label(value: str) -> str:
    return CRYSTAL_LATTICE_LABELS.get(value, value)


def _effective_algorithm_name(spatial_order: str, algorithm_name: str, packing_fraction: float) -> str:
    if algorithm_name != "default":
        return algorithm_name
    if spatial_order == "overlapping_random":
        return "marked_poisson_boolean"
    if spatial_order == "periodic_crystal":
        return "crystal"
    if spatial_order == "hard_core_random":
        if packing_fraction <= 0.18:
            return "poisson_disk"
        if packing_fraction > 0.34:
            return "force_biased"
        return "rsa"
    return algorithm_name


def _algorithm_display_name(value: str) -> str:
    labels = {
        "default": "Automatic",
        "crystal": "periodic crystal",
        "marked_poisson_boolean": "marked Poisson Boolean",
        "poisson_disk": "variable-radius Poisson-disk",
        "rsa": "RSA",
        "force_biased": "force-biased relaxation",
        "lubachevsky_stillinger": "Lubachevsky-Stillinger",
    }
    return labels.get(value, value)

def _force_biased_default_parameters(packing_fraction: float) -> dict[str, float | int]:
    high_density = float(packing_fraction) >= HIGH_DENSITY_PHI_THRESHOLD
    return {
        "stages": 24,
        "relaxation_steps_per_stage": HIGH_DENSITY_RELAXATION_STEPS_PER_STAGE if high_density else 400,
        "initial_radius_fraction": HIGH_DENSITY_INITIAL_RADIUS_FRACTION if high_density else 0.35,
        "outer_diameter_ratio": HIGH_DENSITY_OUTER_DIAMETER_RATIO if high_density else DEFAULT_OUTER_DIAMETER_RATIO,
        "contraction_rate": 1.0e-3,
        "force_scaling_factor": 0.50,
        "max_displacement_fraction": 0.35,
        "intermediate_tolerance_overlap": 1.0e-6,
        "verlet_skin_fraction": 0.50,
        "final_cleanup_steps": HIGH_DENSITY_FINAL_CLEANUP_STEPS if high_density else 400,
    }


def _uploaded_size_file_path(output_path: str, file_name: str) -> Path:
    original = Path(file_name).name
    suffix = Path(original).suffix.lower() or ".csv"
    safe_name = f"{_slugify(Path(original).stem)}{suffix}"
    return ROOT / output_path / "inputs" / safe_name


def _save_uploaded_size_file(uploaded_file, output_path: str) -> Path:
    target = _uploaded_size_file_path(output_path, uploaded_file.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _build_config(data: dict[str, Any]) -> dict[str, Any]:
    algorithm_params: dict[str, Any] = {}
    effective_algorithm_name = data.get("effective_algorithm_name") or _effective_algorithm_name(
        data["spatial_order"],
        data["algorithm_name"],
        float(data["packing_fraction"]),
    )
    if effective_algorithm_name == "crystal":
        algorithm_params["lattice_type"] = data["lattice_type"]
    if effective_algorithm_name in {"poisson_disk", "rsa", "force_biased", "lubachevsky_stillinger"}:
        algorithm_params["max_attempts_per_particle"] = data["max_attempts_per_particle"]
    if effective_algorithm_name in {"poisson_disk", "force_biased", "lubachevsky_stillinger"}:
        algorithm_params["candidates_per_particle"] = data["poisson_disk_candidates_per_particle"]
    if effective_algorithm_name == "force_biased":
        force_biased_defaults = _force_biased_default_parameters(float(data["packing_fraction"]))
        algorithm_params.update(
            {
                "initial_radius_fraction": data.get(
                    "force_biased_initial_radius_fraction",
                    force_biased_defaults["initial_radius_fraction"],
                ),
                "stages": data.get("force_biased_stages", force_biased_defaults["stages"]),
                "relaxation_steps_per_stage": data.get(
                    "force_biased_steps_per_stage",
                    force_biased_defaults["relaxation_steps_per_stage"],
                ),
                "outer_diameter_ratio": data.get(
                    "force_biased_outer_diameter_ratio",
                    force_biased_defaults["outer_diameter_ratio"],
                ),
                "contraction_rate": data.get(
                    "force_biased_contraction_rate",
                    force_biased_defaults["contraction_rate"],
                ),
                "force_scaling_factor": data.get(
                    "force_biased_force_scaling_factor",
                    force_biased_defaults["force_scaling_factor"],
                ),
                "max_displacement_fraction": data.get(
                    "force_biased_max_displacement_fraction",
                    force_biased_defaults["max_displacement_fraction"],
                ),
                "intermediate_tolerance_overlap": data.get(
                    "force_biased_intermediate_tolerance",
                    force_biased_defaults["intermediate_tolerance_overlap"],
                ),
                "verlet_skin_fraction": data.get(
                    "force_biased_verlet_skin_fraction",
                    force_biased_defaults["verlet_skin_fraction"],
                ),
                "final_cleanup_steps": data.get(
                    "force_biased_final_cleanup_steps",
                    force_biased_defaults["final_cleanup_steps"],
                ),
            }
        )
    if effective_algorithm_name == "lubachevsky_stillinger":
        algorithm_params.update(
            {
                "initial_radius_scale": data["ls_initial_radius_scale"],
                "compression_rate": data["ls_compression_rate"],
                "velocity_scale": data["ls_velocity_scale"],
                "max_events": data["ls_max_events"],
                "mass_mode": data["ls_mass_mode"],
                "event_backend": data["ls_event_backend"],
                "neighbor_skin_fraction": data["ls_neighbor_skin_fraction"],
                "neighbor_velocity_safety_factor": data["ls_neighbor_velocity_safety_factor"],
                "max_heap_factor": data["ls_max_heap_factor"],
                "final_cleanup_steps": data["ls_final_cleanup_steps"],
            }
        )

    size_distribution: dict[str, Any] = {"type": data["size_distribution"]}
    if data["size_distribution"] == "quasi_monodisperse":
        size_distribution.update(
            {
                "CV_radius": data["cv_radius"],
                "min_radius_factor": data["min_radius_factor"],
                "max_radius_factor": data["max_radius_factor"],
            }
        )
    if data["size_distribution"] == "continuous_polydisperse":
        distribution = data["continuous_distribution"]
        size_distribution.update(
            {
                "distribution": distribution,
                "min_radius_factor": data["min_radius_factor"],
                "max_radius_factor": data["max_radius_factor"],
            }
        )
        parameters: dict[str, Any] = {}
        if distribution in {"lognormal", "truncated_normal"}:
            size_distribution["CV_radius"] = data["cv_radius"]
            parameters["CV_radius"] = data["cv_radius"]
        if distribution == "gamma":
            parameters["shape"] = data["gamma_shape"]
        if distribution == "weibull":
            parameters["shape"] = data["weibull_shape"]
        if parameters:
            size_distribution["parameters"] = parameters
        if distribution == "custom_file":
            size_distribution["size_file"] = data["custom_size_file_path"]
            size_distribution["file_values"] = data["custom_file_values"]
            if data["custom_file_column"]:
                size_distribution["file_column"] = data["custom_file_column"]

    formats = ["csv", "json"]
    if data["save_hdf5"]:
        formats.append("hdf5")

    config = {
        "project": {
            "name": data["project_name"],
            "description": data["description"],
        },
        "structure": {
            "size_distribution": data["size_distribution"],
            "spatial_order": data["spatial_order"],
        },
        "physical": {
            "mean_diameter": f"{data['mean_diameter_um']} um",
            "medium_thickness": f"{data['medium_thickness_um']} um",
        },
        "domain": {"type": "periodic_cube"},
        "particles": {"num_particles": "auto" if data["auto_particles"] else data["num_particles"]},
        "size_distribution": size_distribution,
        "algorithm": {
            "name": data["algorithm_name"],
            "parameters": algorithm_params,
        },
        "analysis": {
            "compute_g2": data["compute_g2"],
            "compute_Sk": data["compute_sk"],
            "bins": data["bins"],
            "k_max_index": data["k_max_index"],
        },
        "validation": {
            "tolerance_phi": data["tolerance_phi"],
            "tolerance_overlap": data["tolerance_overlap"],
            "require_non_overlap": data["spatial_order"] != "overlapping_random",
        },
        "output": {
            "path": data["output_path"],
            "coordinate_unit": "um",
            "formats": formats,
            "make_plots": data["make_plots"],
        },
        "runtime": {
            "random_seed": data["random_seed"],
            "log_level": "info",
        },
    }
    if "lateral_length_um" in data:
        config["physical"].pop("medium_thickness")
        config["domain"] = {
            "type": "periodic_box", "length": f"{data['lateral_length_um']} um",
            "depth": f"{data['medium_thickness_um']} um",
        }
    if data["spatial_order"] == "overlapping_random":
        config["physical"]["covered_volume_fraction"] = data["covered_volume_fraction"]
    else:
        config["physical"]["packing_fraction"] = data["packing_fraction"]
    return config


def _write_gui_config(config_data: dict[str, Any], output_path: str) -> Path:
    output_dir = ROOT / output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "_gui_config.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return config_path


def _display_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} was not generated.")


def _open_folder(path: Path):
    folder = path.resolve()
    if not folder.exists():
        st.warning(f"Folder does not exist: {folder}")
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception as exc:
        st.warning(f"Could not open folder automatically: {exc}")



def _display_perspective_view(fig_dir: Path):
    perspective_path = fig_dir / "packing_perspective.png"
    legacy_path = fig_dir / "packing_diagonal.png"
    if perspective_path.exists():
        _display_image(perspective_path, "Perspective view")
    else:
        _display_image(legacy_path, "Perspective view")

def _display_result_directory(output_dir: Path, title: str):
    st.subheader(title)
    st.write("**Output directory:**", str(output_dir.resolve()))
    if st.button("Open result folder", key=f"open_{title}_{output_dir}"):
        _open_folder(output_dir)

    validation_path = output_dir / "validation_report.json"
    metadata_path = output_dir / "metadata.json"
    validation = None
    metadata = None
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if validation or metadata:
        col1, col2, col3, col4 = st.columns(4)
        if validation:
            metrics = validation.get("metrics", {})
            col1.metric("Validation", "Passed" if validation.get("passed") else "Failed")
            col3.metric("Particles", metrics.get("n_particles") or metrics.get("n_disks") or "-")
            fraction = metrics.get("packing_fraction_actual", metrics.get("actual_area_fraction"))
            col4.metric("Fraction", f"{float(fraction):.6g}" if fraction is not None else "-")
        if metadata:
            generator = metadata.get("generator", {}).get("name") or metadata.get("algorithm") or "-"
            col2.metric("Generator", generator)

    fig_dir = output_dir / "figures"
    if not fig_dir.exists() and (output_dir / "packing_2d.png").exists():
        st.image(str(output_dir / "packing_2d.png"), caption="2D packing", use_container_width=True)
        return
    _display_perspective_view(fig_dir)


def _preflight_checks(
    data: dict[str, Any],
    estimated_particle_count: int | None,
    particle_estimate_error: str | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    min_factor = data.get("min_radius_factor")
    max_factor = data.get("max_radius_factor")
    if min_factor is not None and max_factor is not None:
        if float(min_factor) <= 0.0 or float(max_factor) <= 0.0:
            errors.append("Radius factors must be positive when specified.")
        elif float(min_factor) >= float(max_factor):
            errors.append("Min radius factor must be smaller than max radius factor.")
        else:
            ratio = float(max_factor) / float(min_factor)
            if ratio > 6.0:
                warnings.append("The radius range is very broad; generation may be slow or biased toward large particles.")
    if data["size_distribution"] == "continuous_polydisperse":
        if data["continuous_distribution"] == "custom_file" and not data["custom_size_file_path"]:
            errors.append("Upload a particle-size file before running the custom_file distribution.")
        if float(data["cv_radius"]) > 1.0:
            warnings.append("Radius CV is high; verify that the sampled particle-size distribution is physically intended.")
    if data["size_distribution"] == "quasi_monodisperse" and float(data["cv_radius"]) > 0.20:
        warnings.append("Quasi-monodisperse CV is relatively large; consider continuous_polydisperse for broader distributions.")

    spatial_order = data["spatial_order"]
    if spatial_order == "periodic_crystal":
        lattice_type = data.get("lattice_type", "FCC")
        max_phi = lattice_max_packing_fraction(lattice_type)
        packing_fraction = float(data["packing_fraction"])
        if packing_fraction > max_phi + 1.0e-9:
            errors.append(
                f"Target packing fraction {packing_fraction:.4g} exceeds the theoretical maximum "
                f"for {_crystal_lattice_label(str(lattice_type))} ({max_phi:.4g})."
            )
    if spatial_order == "overlapping_random" and data.get("covered_volume_fraction") is not None:
        covered = float(data["covered_volume_fraction"])
        if covered > 0.95:
            warnings.append("Very high covered volume fraction creates a large nominal volume fraction and many overlapping spheres.")

    if spatial_order == "periodic_crystal" and "lateral_length_um" in data:
        if str(data.get("lattice_type", "FCC")).upper() == "HCP":
            errors.append("Fixed-box HCP is not supported; use a legacy cubic configuration.")
        if particle_estimate_error and "Crystal dimensions" in particle_estimate_error:
            errors.append(particle_estimate_error)
    if particle_estimate_error:
        warnings.append(f"Particle-count preview is unavailable: {particle_estimate_error}")
    particle_count = int(data["num_particles"] if not data["auto_particles"] else estimated_particle_count or 0)
    if particle_count > 100000 and not data["auto_particles"]:
        errors.append("Estimated particle count exceeds 100,000; reduce thickness, packing fraction, or particle count before running in the GUI.")
    elif particle_count > 25000:
        warnings.append("Estimated particle count is large; generation and rendering may take a long time.")

    return errors, warnings


@st.cache_data(show_spinner=False, max_entries=16)
def _resolved_preview(config_yaml: str):
    return resolve_parameters(config_from_mapping(yaml.safe_load(config_yaml), ROOT), max_particles=100000)


def _config_signature(config_data):
    snapshot = deepcopy(config_data)
    snapshot.get("output", {}).pop("path", None)
    return hashlib.sha256(yaml.safe_dump(snapshot, sort_keys=True).encode()).hexdigest()


def _cache_uploaded_file(uploaded_file):
    content = uploaded_file.getvalue()
    directory = ROOT / ".spherepackgen" / "input_cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(content).hexdigest() + Path(uploaded_file.name).suffix)
    if not path.exists():
        path.write_bytes(content)
    return path


def _apply_cell_dimensions(diameter, phi, lattice, xy, z):
    spacing = diameter * (lattice_basis_count(lattice) * np.pi / (6 * phi))**(1/3)
    st.session_state["lateral_length_input"] = float(xy * spacing)
    st.session_state["depth_input"] = float(z * spacing)


def _box_schematic(length, depth):
    # Product illustration; labels carry the actual dimensions.
    st.markdown(f'''<svg viewBox="0 0 400 180" width="100%" height="160" role="img" aria-label="Periodic box with two equal lateral lengths and independent depth">
    <g fill="#dceefa" stroke="#52758a" stroke-width="2">
    <path d="M100 55 L260 55 L305 25 L145 25 Z"/>
    <path d="M100 55 L260 55 L260 130 L100 130 Z"/>
    <path d="M260 55 L305 25 L305 100 L260 130 Z"/></g>
    <g font-family="sans-serif" font-size="14" fill="#52758a">
    <text x="130" y="155">Lx = {length:.6g} μm</text>
    <text x="40" y="25">Ly = {length:.6g} μm</text>
    <text x="310" y="92">Lz = {depth:.6g} μm</text></g></svg>''', unsafe_allow_html=True)
    st.caption("Dimensions stay fixed; all three axes are periodic. Schematic is not to scale.")


def _show_run_result(output_dir: Path, title: str):
    try:
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        manifest_path = output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    except (OSError, ValueError) as exc:
        st.warning(f"This result could not be read: {exc}")
        return
    st.subheader(title)
    validation = metadata.get("validation", {})
    generator = metadata.get("generator", {})
    parameters = metadata.get("resolved_parameters", {})
    metrics = validation.get("metrics", {})
    diagnostics = generator.get("diagnostics", {})
    if not validation.get("passed"):
        st.error("Generation finished with validation errors.")
    elif diagnostics.get("structure_target_met") is False:
        st.warning("Geometry passed. Structure target was not reached; this is an unconverged candidate.")
    elif generator.get("status") != "success":
        st.warning("Generation produced a candidate. Review the diagnostics before using it.")
    else:
        st.success("Generation complete.")
    c1, c2 = st.columns(2)
    c1.metric("Validation", "Passed" if validation.get("passed") else "Failed")
    c2.metric("Particles", parameters.get("n_particles", metrics.get("n_particles", "-")))
    st.write("**Generator:**", _algorithm_display_name(generator.get("name", "-")))
    sides = parameters.get("box_lengths_m")
    if sides:
        unit_scale = parameters.get("output_unit_m", 1e-6)
        st.metric("Box Lx × Ly × Lz", " × ".join(f"{side / unit_scale:.6g}" for side in sides) + " " + parameters.get("output_unit", "um"))
    fraction = metrics.get("packing_fraction_actual")
    label = "Nominal volume fraction eta" if metadata.get("classification", {}).get("spatial_order") == "overlapping_random" else "Actual packing fraction"
    if fraction is not None:
        st.write(f"**{label}:** {fraction:.8g} (target {metrics.get('packing_fraction_target', fraction):.8g})")
    if "expected_boolean_covered_fraction" in metrics:
        st.write(f"**Expected Boolean covered fraction:** {metrics['expected_boolean_covered_fraction']:.6g}; target {metrics['target_covered_volume_fraction']:.6g}. This is a model expectation, not a measured union volume.")
    if "structure_target_met" in diagnostics:
        st.write("**Structure target:**", "Reached" if diagnostics["structure_target_met"] else "Not reached")
    for message in validation.get("errors", []):
        st.error(message)
    for message in validation.get("warnings", []):
        st.warning(message)
    st.caption(f"Output directory: {output_dir}")
    if manifest is None:
        st.info("Legacy result: no run manifest is available. Its files may come from repeated runs in this directory.")
    files = {name.replace("\\", "/") for name in manifest["files"]} if manifest is not None else {p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file()}
    for filename, label, mime in [
        ("particles_real_units.csv", "Download particle coordinates", "text/csv"),
        ("metadata.json", "Download metadata", "application/json"),
        ("config_replay.yaml", "Download reusable configuration", "application/yaml"),
        ("run_bundle.zip", "Download complete result bundle", "application/zip"),
    ]:
        path = output_dir / filename
        if filename in files and path.exists():
            st.download_button(label, path.read_bytes(), file_name=filename, mime=mime,
                               key=f"{title}_{output_dir}_{filename}", on_click="ignore")
    if st.button("Open result folder", key=f"folder_{title}_{output_dir}"):
        _open_folder(output_dir)
    packing = next((name for name in ("figures/packing_perspective.png", "figures/packing_diagonal.png") if name in files), None)
    if packing:
        st.subheader("3D Packing View")
        if manifest:
            st.caption(f"Preview shows {manifest['rendered_particle_count']} / {manifest['particle_count']} particles. Coordinate exports include all particles.")
        _display_image(output_dir / packing, "Perspective view")
    plots = [(name, label) for name, label in [("radius_distribution.png", "Diameter distribution"), ("g2.png", "g2(r)"), ("Sk.png", "S(k)")] if f"figures/{name}" in files]
    if plots:
        with st.expander("Analysis plots", expanded=True):
            for name, label in plots:
                _display_image(output_dir / "figures" / name, label)
    with st.expander("Run parameters, validation and timing"):
        input_path = output_dir / "config_input.yaml"
        if input_path.exists():
            st.code(input_path.read_text(encoding="utf-8"), language="yaml")
        st.json(validation)
        st.json(metadata.get("stage_timings_s", {}))
        st.json(diagnostics)
        st.json(sorted(files))


@st.fragment(run_every="1s")
def _poll_gui_job():
    job = st.session_state.get("gui_job")
    if job:
        directory = Path(job["directory"])
        progress = read_progress(directory)
        if job["process"].poll() is None:
            stages = ["starting", "resolution", "generation", "validation", "analysis", "export", "complete"]
            stage = progress.get("stage", "starting")
            index = stages.index(stage) if stage in stages else 0
            st.progress(index / (len(stages) - 1), text=f"{stage.capitalize()} · elapsed {time.time() - job['started']:.0f} s")
            st.caption(f"This run: {directory}")
            if st.button("Cancel run", key="cancel_gui_run"):
                cancel_job(job)
                st.session_state["gui_last_error"] = "Run cancelled. Previous results remain available."
                st.session_state.pop("gui_job")
                st.rerun()
        else:
            if job["process"].returncode == 0 and (directory / "run_manifest.json").exists():
                st.session_state["last_generated_output_path"] = str(directory)
                st.session_state["last_generated_signature"] = job["signature"]
                st.session_state.pop("gui_last_error", None)
            else:
                st.session_state["gui_last_error"] = progress.get("message") or f"Generation stopped. Details are in {directory / 'gui_worker.log'}."
            st.session_state.pop("gui_job")
            st.rerun()


def _live_results(current_signature):
    job = st.session_state.get("gui_job")
    if job:
        _poll_gui_job()
    if st.session_state.get("gui_last_error"):
        st.error(st.session_state["gui_last_error"])
    last = st.session_state.get("last_generated_output_path")
    if last:
        if st.session_state.get("last_generated_signature") != current_signature:
            st.info("Current settings have not been generated. The result below belongs to the saved run parameters.")
        _show_run_result(Path(last), "Latest Result")
    elif not job:
        st.info("Generate a structure to view its validated result here.")


def _preview_truncated_normal(rng: np.random.Generator, mean: float, std: float, low: float, high: float, n: int) -> np.ndarray:
    return _truncated_normal(rng, mean, std, low, high, n, batch_min=128)


def _preview_clip_and_scale(radii: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if low is not None:
        radii = np.maximum(radii, low)
    if high is not None:
        radii = np.minimum(radii, high)
    if radii.size == 0 or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("Particle-size preview needs positive numeric values.")
    return radii * (0.5 / np.mean(radii))


def _parse_uploaded_size_values(uploaded_file, column: str) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Upload a particle-size file to preview custom_file.")

    text = uploaded_file.getvalue().decode("utf-8-sig")
    values: list[float] = []
    header: list[str] | None = None
    selected = column.strip()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = [part.strip() for part in line.split(",")] if "," in line else line.split()
        index = None
        if selected:
            if selected.isdigit():
                index = int(selected)
            elif header:
                normalized_header = [item.strip().lower().replace(" ", "_") for item in header]
                key = selected.lower().replace(" ", "_")
                if key in normalized_header:
                    index = normalized_header.index(key)
        candidates = []
        if index is not None and 0 <= index < len(tokens):
            candidates = [tokens[index]]
        else:
            candidates = tokens

        numeric_values: list[float] = []
        for token in candidates:
            try:
                numeric_values.append(float(token))
            except ValueError:
                pass
        if numeric_values:
            values.append(numeric_values[0])
        elif header is None:
            header = tokens

    if not values:
        raise ValueError("The uploaded file contains no numeric particle sizes.")
    values_array = np.asarray(values, dtype=float)
    if np.any(values_array <= 0):
        raise ValueError("The uploaded particle-size file must contain only positive values.")
    return values_array


def _preview_diameters_um(
    *,
    size_distribution: str,
    mean_diameter_um: float,
    random_seed: int,
    cv_radius: float,
    min_radius_factor: float | None,
    max_radius_factor: float | None,
    continuous_distribution: str,
    gamma_shape: float,
    weibull_shape: float,
    custom_size_file,
    custom_file_values: str,
    custom_file_column: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_seed) + 104729)
    n = 2000
    low = None if min_radius_factor is None else 0.5 * float(min_radius_factor)
    high = None if max_radius_factor is None else 0.5 * float(max_radius_factor)
    if size_distribution != "monodisperse":
        _validate_bounds(low, high)

    if size_distribution == "monodisperse":
        radii = np.full(n, 0.5, dtype=float)
    elif size_distribution == "quasi_monodisperse":
        radii = _preview_truncated_normal(rng, 0.5, 0.5 * float(cv_radius), low or 0.425, high or 0.575, n)
    elif continuous_distribution == "truncated_normal":
        radii = _preview_truncated_normal(rng, 0.5, 0.5 * float(cv_radius), low or 0.2, high or 0.9, n)
    elif continuous_distribution == "lognormal":
        sigma = np.sqrt(np.log(1.0 + float(cv_radius) ** 2))
        mu = np.log(0.5) - 0.5 * sigma**2
        radii = rng.lognormal(mean=mu, sigma=sigma, size=n)
    elif continuous_distribution == "uniform":
        radii = rng.uniform(low or 0.25, high or 0.75, size=n)
    elif continuous_distribution == "gamma":
        shape = float(gamma_shape)
        radii = rng.gamma(shape=shape, scale=0.5 / shape, size=n)
    elif continuous_distribution == "weibull":
        shape = float(weibull_shape)
        scale = 0.5 / gamma_function(1.0 + 1.0 / shape)
        radii = scale * rng.weibull(a=shape, size=n)
    elif continuous_distribution == "custom_file":
        values = _parse_uploaded_size_values(custom_size_file, custom_file_column)
        samples = rng.choice(values, size=n, replace=len(values) < n)
        radii = samples / 2.0 if custom_file_values == "diameter" else samples
    else:
        radii = np.full(n, 0.5, dtype=float)

    radii = _preview_clip_and_scale(radii, low, high)
    return 2.0 * radii * float(mean_diameter_um)


def _preview_histogram_settings(diameters_um: np.ndarray) -> tuple[int, np.ndarray]:
    diameters_um = np.asarray(diameters_um, dtype=float).reshape(-1)
    if diameters_um.size == 0:
        raise ValueError("Particle-size preview needs at least one sampled diameter.")
    bins = 1 if np.ptp(diameters_um) < 1.0e-12 else 32
    weights_percent = np.full(diameters_um.shape, 100.0 / diameters_um.size, dtype=float)
    return bins, weights_percent


def _display_size_preview(**kwargs):
    st.caption("Particle-size preview")
    try:
        diameters_um = _preview_diameters_um(**kwargs)
    except Exception as exc:
        st.info(str(exc))
        return

    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=140)
    bins, weights_percent = _preview_histogram_settings(diameters_um)
    ax.hist(diameters_um, bins=bins, weights=weights_percent, color="#9ecae1", edgecolor="white")
    ax.axvline(float(np.mean(diameters_um)), color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Diameter (um)")
    ax.set_ylabel("Sample fraction")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _estimate_particle_count_from_preview(
    *,
    size_distribution: str,
    spatial_order: str,
    lattice_type: str,
    mean_diameter_um: float,
    packing_fraction: float,
    medium_thickness_um: float,
    random_seed: int,
    cv_radius: float,
    min_radius_factor: float | None,
    max_radius_factor: float | None,
    continuous_distribution: str,
    gamma_shape: float,
    weibull_shape: float,
    custom_size_file,
    custom_file_values: str,
    custom_file_column: str,
    lateral_length_um: float | None = None,
) -> tuple[int | None, float | None, str | None]:
    try:
        diameters_um = _preview_diameters_um(
            size_distribution=size_distribution,
            mean_diameter_um=mean_diameter_um,
            random_seed=random_seed,
            cv_radius=cv_radius,
            min_radius_factor=min_radius_factor,
            max_radius_factor=max_radius_factor,
            continuous_distribution=continuous_distribution,
            gamma_shape=gamma_shape,
            weibull_shape=weibull_shape,
            custom_size_file=custom_size_file,
            custom_file_values=custom_file_values,
            custom_file_column=custom_file_column,
        )
    except Exception as exc:
        return None, None, str(exc)

    dimensionless_radii = diameters_um / (2.0 * float(mean_diameter_um))
    mean_particle_volume = float(np.mean(4.0 * np.pi * dimensionless_radii**3 / 3.0))
    target_box_length = float(medium_thickness_um) / float(mean_diameter_um)
    target_lateral_length = target_box_length if lateral_length_um is None else float(lateral_length_um) / float(mean_diameter_um)
    estimate = int(round(float(packing_fraction) * target_lateral_length**2 * target_box_length / mean_particle_volume))
    estimate = max(1, estimate)
    if spatial_order == "periodic_crystal":
        basis_count = lattice_basis_count(lattice_type)
        if lateral_length_um is None:
            unit_cells = max(1, int(round((estimate / basis_count) ** (1.0 / 3.0))))
            estimate = basis_count * unit_cells**3
        else:
            spacing = (basis_count * np.pi / (6 * float(packing_fraction)))**(1/3)
            cells_xy = max(1, round(target_lateral_length / spacing))
            cells_z = max(1, round(target_box_length / spacing))
            estimate = basis_count * cells_xy**2 * cells_z
            if not np.allclose([target_lateral_length, target_box_length], np.array([cells_xy, cells_z]) * spacing, rtol=1e-5):
                return estimate, mean_particle_volume, (
                    "Crystal dimensions must fit complete unit cells at the target density. "
                    f"Nearby compatible length/depth: {cells_xy * spacing * mean_diameter_um:.8g} / "
                    f"{cells_z * spacing * mean_diameter_um:.8g} um."
                )
    return estimate, mean_particle_volume, None


def _box_length_from_particle_count_um(
    *,
    n_particles: int,
    mean_particle_volume: float,
    packing_fraction: float,
    mean_diameter_um: float,
) -> float:
    box_length_dimensionless = (int(n_particles) * float(mean_particle_volume) / float(packing_fraction)) ** (1.0 / 3.0)
    return box_length_dimensionless * float(mean_diameter_um)


def main():
    st.set_page_config(page_title="SpherePackGen", layout="wide")
    st.title("SpherePackGen")
    st.caption("Configure a periodic structure, check its parameters, and review saved results.")

    with st.sidebar:
        st.header("Project")
        project_name = st.text_input("Project name", value="gui_demo_random_polydisperse")
        description = st.text_area("Description", value="add your description of this project")

    tab_basic, tab_advanced, tab_config, tab_results, tab_history = st.tabs(["Run", "Advanced", "Config Preview", "Results", "History"])

    with tab_basic:
        st.header("Structure")
        size_distribution = st.selectbox(
            "Particle-size distribution",
            [value for value, _ in SIZE_DISTRIBUTION_OPTIONS],
            index=0,
            format_func=_size_distribution_label,
        )
        if size_distribution == "continuous_polydisperse":
            spatial_order_options = POLYDISPERSE_SPATIAL_ORDER_OPTIONS
        elif size_distribution == "quasi_monodisperse":
            spatial_order_options = QUASI_MONODISPERSE_SPATIAL_ORDER_OPTIONS
        else:
            spatial_order_options = SPATIAL_ORDER_OPTIONS
        spatial_order = st.selectbox(
            "Spatial order / structure type",
            spatial_order_options,
            index=spatial_order_options.index("hard_core_random"),
            format_func=_spatial_order_label,
        )
        if spatial_order == "overlapping_random":
            algorithm_options = ["default", "marked_poisson_boolean"]
        elif spatial_order == "hard_core_random":
            algorithm_options = [
                "default",
                "poisson_disk",
                "rsa",
                "force_biased",
                "lubachevsky_stillinger",
            ]
        elif spatial_order == "periodic_crystal":
            algorithm_options = ["default", "crystal"]
        else:
            algorithm_options = ["default"]
        algorithm_name = st.selectbox(
            "Algorithm",
            algorithm_options,
            index=0,
            format_func=_algorithm_display_name,
            help=(
                "non_overlapping_random maps to the internal hard_core_random route. "
                "With Algorithm=default, SpherePackGen chooses the concrete generator from the target packing fraction."
            ),
        )

        st.header("Macro Physical Parameters")
        left, right = st.columns(2)
        with left:
            mean_diameter_um = st.number_input("Mean particle diameter (um)", min_value=0.001, value=0.30, step=0.05, format="%.6g")
            lateral_length_um = st.number_input("Lateral length Lx = Ly (um)", min_value=0.01, value=2.0, step=0.1, format="%.8g", key="lateral_length_input")
        with right:
            if spatial_order == "overlapping_random":
                covered_volume_fraction = st.number_input("Target covered volume fraction phi", min_value=0.001, max_value=0.999, value=0.70, step=0.01)
                packing_fraction = float(-np.log1p(-covered_volume_fraction))
                st.caption(f"Nominal fraction eta = {packing_fraction:.6g}")
            else:
                covered_volume_fraction = None
                packing_fraction = st.number_input("Target packing fraction", min_value=0.001, max_value=0.74, value=0.50, step=0.01, format="%.8g")
            medium_thickness_um = st.number_input("Depth Lz (um)", min_value=0.01, value=2.0, step=0.1, format="%.8g", key="depth_input")
        _box_schematic(lateral_length_um, medium_thickness_um)

    with st.sidebar:
        history_options = _result_history_options()
        selected_history = st.selectbox(
            "Load historical result",
            [""] + history_options,
            index=0,
            format_func=lambda value: "Select a previous result" if not value else value,
        )

    with tab_advanced:
        with st.expander("Output options"):
            st.header("Output")
            timestamp_output = st.checkbox(
                "Add timestamp to output folder",
                value=True,
                help="Creates a fresh result directory for each run to avoid overwriting older results.",
            )
            if "gui_output_timestamp" not in st.session_state:
                st.session_state["gui_output_timestamp"] = _new_run_timestamp()
            if st.button("New run folder"):
                st.session_state["gui_output_timestamp"] = _new_run_timestamp()
                st.session_state["gui_output_path"] = _timestamped_output_path(project_name, st.session_state["gui_output_timestamp"])

            suggested_output_path = (
                _timestamped_output_path(project_name, st.session_state["gui_output_timestamp"])
                if timestamp_output
                else _default_output_path(project_name)
            )
            previous_suggestion = st.session_state.get("gui_output_suggestion")
            if "gui_output_path" not in st.session_state or st.session_state["gui_output_path"] == previous_suggestion:
                st.session_state["gui_output_path"] = suggested_output_path
            st.session_state["gui_output_suggestion"] = suggested_output_path
            output_path = st.text_input("Output path", key="gui_output_path")
            st.caption(f"This run will be saved to: {(ROOT / output_path).resolve()}")

            make_plots = st.checkbox(
                "Generate analysis plots and 3D views",
                value=True,
                key="make_plots_default_true_v1",
            )
            save_hdf5 = st.checkbox("Save HDF5", value=False, key="save_hdf5_default_false_v3")


    with tab_advanced:
        with st.expander("Size distribution", expanded=True):
            st.subheader("Size Distribution")
            cv_radius = 0.03
            min_radius_factor = 0.90
            max_radius_factor = 1.10
            continuous_distribution = "lognormal"
            gamma_shape = 4.0
            weibull_shape = 2.5
            custom_size_file = None
            custom_size_file_path = ""
            custom_file_values = "diameter"
            custom_file_column = ""
            preview_seed = int(st.session_state.get("random_seed", 12345))

            if size_distribution == "quasi_monodisperse":
                cv_radius = st.number_input("Radius CV", min_value=0.0, max_value=1.0, value=0.03, step=0.01, help=SIZE_PARAMETER_HELP["cv"])
                min_radius_factor = st.number_input(
                    "Min radius factor",
                    min_value=0.01,
                    max_value=5.0,
                    value=0.90,
                    step=0.05,
                    help=SIZE_PARAMETER_HELP["min_factor"],
                )
                max_radius_factor = st.number_input(
                    "Max radius factor",
                    min_value=0.01,
                    max_value=5.0,
                    value=1.10,
                    step=0.05,
                    help=SIZE_PARAMETER_HELP["max_factor"],
                )
            elif size_distribution == "continuous_polydisperse":
                continuous_distribution = st.selectbox(
                    "Continuous distribution",
                    ["lognormal", "truncated_normal", "uniform", "gamma", "weibull", "custom_file"],
                    index=0,
                    help=SIZE_PARAMETER_HELP["distribution"],
                )
                if continuous_distribution in {"lognormal", "truncated_normal"}:
                    cv_radius = st.number_input("Radius CV", min_value=0.001, max_value=2.0, value=0.15, step=0.01, help=SIZE_PARAMETER_HELP["cv"])
                    min_radius_factor = st.number_input(
                        "Min radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=0.40,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["min_factor"],
                    )
                    max_radius_factor = st.number_input(
                        "Max radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=1.80,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["max_factor"],
                    )
                elif continuous_distribution == "uniform":
                    min_radius_factor = st.number_input(
                        "Uniform min radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=0.60,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["min_factor"],
                    )
                    max_radius_factor = st.number_input(
                        "Uniform max radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=1.60,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["max_factor"],
                    )
                elif continuous_distribution == "gamma":
                    gamma_shape = st.number_input("Gamma shape", min_value=0.05, max_value=200.0, value=4.0, step=0.1, help=SIZE_PARAMETER_HELP["gamma_shape"])
                    min_radius_factor = st.number_input(
                        "Min radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=0.20,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["min_factor"],
                    )
                    max_radius_factor = st.number_input(
                        "Max radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=2.50,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["max_factor"],
                    )
                elif continuous_distribution == "weibull":
                    weibull_shape = st.number_input("Weibull shape", min_value=0.05, max_value=200.0, value=2.5, step=0.1, help=SIZE_PARAMETER_HELP["weibull_shape"])
                    min_radius_factor = st.number_input(
                        "Min radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=0.20,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["min_factor"],
                    )
                    max_radius_factor = st.number_input(
                        "Max radius factor",
                        min_value=0.01,
                        max_value=10.0,
                        value=2.50,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["max_factor"],
                    )
                elif continuous_distribution == "custom_file":
                    custom_size_file = st.file_uploader("Particle size file", type=["csv", "txt", "dat"], help=SIZE_PARAMETER_HELP["size_file"])
                    custom_file_values = st.selectbox("File values represent", ["diameter", "radius"], index=0, help=SIZE_PARAMETER_HELP["file_values"])
                    custom_file_column = st.text_input("Column name or index", value="", help=SIZE_PARAMETER_HELP["file_column"])
                    min_radius_factor = st.number_input(
                        "Optional min radius factor",
                        min_value=0.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["min_factor"],
                    )
                    max_radius_factor = st.number_input(
                        "Optional max radius factor",
                        min_value=0.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.05,
                        help=SIZE_PARAMETER_HELP["max_factor"],
                    )
                    min_radius_factor = None if min_radius_factor == 0 else min_radius_factor
                    max_radius_factor = None if max_radius_factor == 0 else max_radius_factor
                    if custom_size_file is not None:
                        custom_size_file_path = str(_cache_uploaded_file(custom_size_file))

            # Preview is rendered from the shared parameter resolver below.
            preview_container = st.container()
        with st.container():
            st.subheader("Algorithm Parameters")
            effective_algorithm_name = _effective_algorithm_name(spatial_order, algorithm_name, packing_fraction)
            if algorithm_name == "default":
                st.caption(f"Default currently resolves to: {_algorithm_display_name(effective_algorithm_name)}")
            else:
                st.caption(f"Showing parameters for: {_algorithm_display_name(effective_algorithm_name)}")

            lattice_type = "FCC"
            max_attempts_per_particle = 10000
            poisson_disk_candidates_per_particle = 30
            force_biased_defaults = _force_biased_default_parameters(packing_fraction)
            force_biased_stages = int(force_biased_defaults["stages"])
            force_biased_steps_per_stage = int(force_biased_defaults["relaxation_steps_per_stage"])
            force_biased_initial_radius_fraction = float(force_biased_defaults["initial_radius_fraction"])
            force_biased_outer_diameter_ratio = float(force_biased_defaults["outer_diameter_ratio"])
            force_biased_contraction_rate = float(force_biased_defaults["contraction_rate"])
            force_biased_force_scaling_factor = float(force_biased_defaults["force_scaling_factor"])
            force_biased_max_displacement_fraction = float(force_biased_defaults["max_displacement_fraction"])
            force_biased_intermediate_tolerance = float(force_biased_defaults["intermediate_tolerance_overlap"])
            force_biased_verlet_skin_fraction = float(force_biased_defaults["verlet_skin_fraction"])
            force_biased_final_cleanup_steps = int(force_biased_defaults["final_cleanup_steps"])
            ls_initial_radius_scale = 0.70
            ls_compression_rate = 1.0e-3
            ls_velocity_scale = 1.0
            ls_max_events = 25000
            ls_mass_mode = "equal"
            ls_event_backend = "verlet_heap"
            ls_neighbor_skin_fraction = 1.0
            ls_neighbor_velocity_safety_factor = 2.0
            ls_max_heap_factor = 8.0
            ls_final_cleanup_steps = 200

            if effective_algorithm_name == "crystal":
                lattice_type = st.selectbox(
                    "Crystal lattice",
                    [value for value in CRYSTAL_LATTICE_OPTIONS if value != "HCP"],
                    index=0,
                    format_func=_crystal_lattice_label,
                    help=(
                        "Select the periodic unit-cell basis. Theoretical maximum packing fractions: "
                        "FCC/HCP 0.7405, BCC 0.6802, SC 0.5236, diamond cubic 0.3401."
                    ),
                )
                spacing = mean_diameter_um * (lattice_basis_count(lattice_type) * np.pi / (6 * packing_fraction))**(1/3)
                cells_xy = st.number_input("Crystal cells nx = ny", min_value=1, value=max(1, round(lateral_length_um / spacing)), step=1)
                cells_z = st.number_input("Crystal cells nz", min_value=1, value=max(1, round(medium_thickness_um / spacing)), step=1)
                st.caption(f"Compatible L / D: {cells_xy * spacing:.8g} / {cells_z * spacing:.8g} μm")
                st.button("Apply compatible crystal dimensions", on_click=_apply_cell_dimensions,
                          args=(mean_diameter_um, packing_fraction, lattice_type, cells_xy, cells_z))
            elif effective_algorithm_name == "poisson_disk":
                max_attempts_per_particle = st.number_input(
                    "Hard-core max attempts per particle",
                    min_value=100,
                    value=10000,
                    step=100,
                    help="Maximum random trial budget for placing each non-overlapping particle before reporting failure.",
                )
                poisson_disk_candidates_per_particle = st.number_input(
                    "Poisson-disk candidates per particle",
                    min_value=1,
                    value=30,
                    step=1,
                    help="Number of candidate locations tested per accepted particle; larger values improve spatial spread but cost more time.",
                )
            elif effective_algorithm_name == "rsa":
                max_attempts_per_particle = st.number_input(
                    "RSA max attempts per particle",
                    min_value=100,
                    value=10000,
                    step=100,
                    help="Maximum random insertion attempts allowed for each hard sphere before RSA reports failure.",
                )
            elif effective_algorithm_name == "force_biased":
                force_biased_initial_radius_fraction = st.number_input(
                    "Initial radius fraction",
                    min_value=0.01,
                    max_value=1.0,
                    value=force_biased_initial_radius_fraction,
                    step=0.05,
                    help="Initial radius as a fraction of the final resolved radius; larger values start closer to the target but are harder to initialize.",
                )
                force_biased_stages = st.number_input(
                    "Radius-growth stages",
                    min_value=1,
                    value=force_biased_stages,
                    step=1,
                    help="Number of radius-growth / relaxation stages from the initial smaller spheres to the target radii.",
                )
                force_biased_steps_per_stage = st.number_input(
                    "Relaxation steps per stage",
                    min_value=1,
                    value=force_biased_steps_per_stage,
                    step=10,
                    help="Relaxation iterations performed at each stage before increasing the particle radii.",
                )
                force_biased_contraction_rate = st.number_input(
                    "Contraction rate",
                    min_value=1.0e-8,
                    value=force_biased_contraction_rate,
                    format="%.1e",
                    help="Rate for shrinking the outer repulsion shell toward the true hard-core diameter.",
                )
                with st.expander("Advanced force-biased controls"):
                    max_attempts_per_particle = st.number_input(
                        "Initializer max attempts per particle",
                        min_value=100,
                        value=10000,
                        step=100,
                        help="Trial budget used by the low-density initializer before force-biased relaxation starts.",
                    )
                    poisson_disk_candidates_per_particle = st.number_input(
                        "Initializer candidates per particle",
                        min_value=1,
                        value=30,
                        step=1,
                        help="Candidate count used by the Poisson-disk style initializer; larger values usually give better starting spread.",
                    )
                    force_biased_outer_diameter_ratio = st.number_input(
                        "Outer diameter ratio",
                        min_value=1.0,
                        max_value=3.0,
                        value=force_biased_outer_diameter_ratio,
                        step=0.01,
                        help="Outer repulsion shell size relative to true contact distance; values above 1 push near neighbors apart before contact.",
                    )
                    force_biased_force_scaling_factor = st.number_input(
                        "Force scale",
                        min_value=0.001,
                        value=force_biased_force_scaling_factor,
                        step=0.05,
                        help="Multiplier converting shell overlap forces into position updates.",
                    )
                    force_biased_max_displacement_fraction = st.number_input(
                        "Max displacement fraction",
                        min_value=0.001,
                        max_value=2.0,
                        value=force_biased_max_displacement_fraction,
                        step=0.05,
                        help="Caps each update step as a fraction of the mean particle radius to avoid unstable jumps.",
                    )
                    force_biased_intermediate_tolerance = st.number_input(
                        "Intermediate overlap tolerance",
                        min_value=1e-12,
                        value=force_biased_intermediate_tolerance,
                        format="%.1e",
                        help="Temporary overlap tolerance allowed during staged growth before final cleanup.",
                    )
                    force_biased_verlet_skin_fraction = st.number_input(
                        "Verlet skin fraction",
                        min_value=0.01,
                        max_value=5.0,
                        value=force_biased_verlet_skin_fraction,
                        step=0.05,
                        help="Extra neighbor-list margin as a fraction of mean radius; larger values rebuild less often but track more pairs.",
                    )
                    force_biased_final_cleanup_steps = st.number_input(
                        "Final cleanup steps",
                        min_value=0,
                        value=force_biased_final_cleanup_steps,
                        step=50,
                        help="Final true-contact relaxation budget after all radius-growth stages are complete.",
                    )
            elif effective_algorithm_name == "lubachevsky_stillinger":
                max_attempts_per_particle = st.number_input(
                    "LS initializer max attempts per particle",
                    min_value=100,
                    value=30000,
                    step=100,
                    help="Random trial budget used to create the reduced-radius non-overlapping starting configuration.",
                )
                poisson_disk_candidates_per_particle = st.number_input(
                    "LS initializer candidates per particle",
                    min_value=1,
                    value=30,
                    step=1,
                    help="Candidate count used by the reduced-radius Poisson-disk initializer before event-driven growth starts.",
                )
                ls_initial_radius_scale = st.number_input(
                    "LS initial radius scale",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.70,
                    step=0.05,
                    help="Starting radius divided by final radius; lower values initialize more easily but require more compression events.",
                )
                ls_compression_rate = st.number_input(
                    "LS compression rate",
                    min_value=0.0,
                    value=1.0e-3,
                    format="%.1e",
                    help="Linear rate at which all radii grow during event-driven hard-sphere dynamics.",
                )
                ls_velocity_scale = st.number_input(
                    "LS velocity scale",
                    min_value=1.0e-6,
                    value=1.0,
                    format="%.3g",
                    help="Root-mean-square particle speed used for the initial random velocities.",
                )
                ls_max_events = st.number_input(
                    "LS max collision events",
                    min_value=0,
                    value=25000,
                    step=1000,
                    help="Maximum elastic collision events before the algorithm stops and reports that the target radii were not reached.",
                )
                ls_mass_mode = st.selectbox(
                    "LS collision mass mode",
                    ["equal", "volume"],
                    index=0,
                    help="Equal treats all particles as the same mass; volume uses mass proportional to radius cubed for polydisperse spheres.",
                )
                ls_event_backend = st.selectbox(
                    "LS event backend",
                    ["verlet_heap", "exhaustive"],
                    index=0,
                    help="verlet_heap uses a Verlet-neighbor event queue for larger systems; exhaustive scans every pair and is mainly a conservative small-system fallback.",
                )
                ls_neighbor_skin_fraction = st.number_input(
                    "LS neighbor skin fraction",
                    min_value=0.05,
                    max_value=10.0,
                    value=1.0,
                    step=0.05,
                    help="Extra Verlet-neighbor margin measured in mean diameters; larger values reduce missed far collisions but keep more candidate pairs.",
                )
                ls_neighbor_velocity_safety_factor = st.number_input(
                    "LS neighbor velocity safety factor",
                    min_value=0.1,
                    max_value=20.0,
                    value=2.0,
                    step=0.1,
                    help="Safety multiplier used to decide when the Verlet neighbor list must be rebuilt after fast collision-driven motion.",
                )
                ls_max_heap_factor = st.number_input(
                    "LS heap rebuild factor",
                    min_value=1.0,
                    max_value=100.0,
                    value=8.0,
                    step=1.0,
                    help="Rebuilds the event queue when invalidated queued events grow too large relative to the active neighbor-pair count.",
                )
                ls_final_cleanup_steps = st.number_input(
                    "LS final cleanup steps",
                    min_value=0,
                    value=200,
                    step=50,
                    help="Small deterministic post-pass budget used only to remove numerical residual overlaps after event-driven growth.",
                )
            elif effective_algorithm_name == "marked_poisson_boolean":
                st.info("This algorithm has no additional controls: centers are independent random points and radii come from the selected size distribution.")
            else:
                st.info("No algorithm-specific parameters are needed for the current selection.")
        with st.expander("Analysis and validation"):
            st.subheader("Analysis and Validation")
            compute_g2 = st.checkbox("Compute g2(r)", value=True)
            compute_sk = st.checkbox("Compute S(k)", value=True)
            profile = st.selectbox("Analysis detail", ["Quick", "Standard", "Detailed"], index=1)
            bins = DEFAULT_G2_BINS
            k_max_index = st.number_input("S(k) maximum index", min_value=1, max_value=30, value={"Quick": 5, "Standard": 10, "Detailed": DEFAULT_SK_MAX_INDEX}[profile], key=f"sk_index_{profile}")
            st.caption(f"S(k) samples {(2 * k_max_index + 1)**3 - 1:,} wavevectors when enabled; plotting is controlled separately.")
            run_timeout_s = st.number_input("Run time limit (s)", min_value=10, max_value=86400, value=600, step=60)
            tolerance_phi = st.number_input(
                "Packing fraction tolerance",
                min_value=1e-12,
                value=DEFAULT_PACKING_FRACTION_TOLERANCE,
                format="%.3g",
                help=(
                    "Allowed absolute difference between target and actual packing fraction during validation. "
                    "The default 0.001 is suitable for GUI exploratory runs."
                ),
            )
            tolerance_overlap = st.number_input(
                "Overlap tolerance",
                min_value=1e-14,
                value=DEFAULT_OVERLAP_TOLERANCE,
                format="%.1e",
                key="overlap_tolerance_default_1e_minus_5_v2",
            )

    estimated_particle_count, estimated_mean_volume, particle_estimate_error = _estimate_particle_count_from_preview(
        size_distribution=size_distribution,
        spatial_order=spatial_order,
        lattice_type=lattice_type,
        mean_diameter_um=mean_diameter_um,
        packing_fraction=packing_fraction,
        medium_thickness_um=medium_thickness_um,
        lateral_length_um=lateral_length_um,
        random_seed=int(st.session_state.get("random_seed", 12345)),
        cv_radius=cv_radius,
        min_radius_factor=min_radius_factor,
        max_radius_factor=max_radius_factor,
        continuous_distribution=continuous_distribution,
        gamma_shape=gamma_shape,
        weibull_shape=weibull_shape,
        custom_size_file=custom_size_file,
        custom_file_values=custom_file_values,
        custom_file_column=custom_file_column,
    )

    with tab_basic:
        st.subheader("Particles")
        auto_particles = st.checkbox("Automatically compute particle count", value=True,
                                     help="Resolve the count from actual sampled radii, fixed dimensions and target density.")
        num_particles = int(estimated_particle_count or 1)
        if not auto_particles:
            num_particles = st.number_input("Particle count", min_value=1, value=max(1, num_particles),
                                            step=1, key="manual_particle_count")
        random_seed = st.number_input("Random seed", min_value=0, value=12345, step=1, key="random_seed")

    form_data = {
        "project_name": project_name,
        "description": description,
        "output_path": output_path,
        "size_distribution": size_distribution,
        "spatial_order": spatial_order,
        "algorithm_name": algorithm_name,
        "effective_algorithm_name": effective_algorithm_name,
        "mean_diameter_um": mean_diameter_um,
        "packing_fraction": packing_fraction,
        "covered_volume_fraction": covered_volume_fraction,
        "medium_thickness_um": medium_thickness_um,
        "lateral_length_um": lateral_length_um,
        "auto_particles": auto_particles,
        "num_particles": int(num_particles),
        "make_plots": make_plots,
        "save_hdf5": save_hdf5,
        "cv_radius": cv_radius,
        "min_radius_factor": min_radius_factor,
        "max_radius_factor": max_radius_factor,
        "continuous_distribution": continuous_distribution,
        "gamma_shape": gamma_shape,
        "weibull_shape": weibull_shape,
        "custom_size_file_path": custom_size_file_path,
        "custom_file_values": custom_file_values,
        "custom_file_column": custom_file_column.strip(),
        "lattice_type": lattice_type,
        "max_attempts_per_particle": int(max_attempts_per_particle),
        "poisson_disk_candidates_per_particle": int(poisson_disk_candidates_per_particle),
        "force_biased_stages": int(force_biased_stages),
        "force_biased_steps_per_stage": int(force_biased_steps_per_stage),
        "force_biased_initial_radius_fraction": float(force_biased_initial_radius_fraction),
        "force_biased_outer_diameter_ratio": float(force_biased_outer_diameter_ratio),
        "force_biased_contraction_rate": float(force_biased_contraction_rate),
        "force_biased_force_scaling_factor": float(force_biased_force_scaling_factor),
        "force_biased_max_displacement_fraction": float(force_biased_max_displacement_fraction),
        "force_biased_intermediate_tolerance": float(force_biased_intermediate_tolerance),
        "force_biased_verlet_skin_fraction": float(force_biased_verlet_skin_fraction),
        "force_biased_final_cleanup_steps": int(force_biased_final_cleanup_steps),
        "ls_initial_radius_scale": float(ls_initial_radius_scale),
        "ls_compression_rate": float(ls_compression_rate),
        "ls_velocity_scale": float(ls_velocity_scale),
        "ls_max_events": int(ls_max_events),
        "ls_mass_mode": ls_mass_mode,
        "ls_event_backend": ls_event_backend,
        "ls_neighbor_skin_fraction": float(ls_neighbor_skin_fraction),
        "ls_neighbor_velocity_safety_factor": float(ls_neighbor_velocity_safety_factor),
        "ls_max_heap_factor": float(ls_max_heap_factor),
        "ls_final_cleanup_steps": int(ls_final_cleanup_steps),
        "compute_g2": compute_g2,
        "compute_sk": compute_sk,
        "bins": int(bins),
        "k_max_index": int(k_max_index),
        "tolerance_phi": float(tolerance_phi),
        "tolerance_overlap": float(tolerance_overlap),
        "random_seed": int(random_seed),
    }
    preflight_errors, preflight_warnings = _preflight_checks(form_data, estimated_particle_count, particle_estimate_error)
    config_data = _build_config(form_data)
    resolved = None
    if not preflight_errors:
        try:
            resolved = _resolved_preview(yaml.safe_dump(config_data, sort_keys=True))
            if resolved.n_particles > 100000:
                preflight_errors.append("Resolved particle count exceeds 100,000. Reduce the box dimensions or density.")
        except Exception as exc:
            preflight_errors.append(str(exc))
    if effective_algorithm_name == "rsa" and packing_fraction > .34:
        preflight_warnings.append("RSA is intended for lower density. Force-biased is recommended for this target density.")
    if effective_algorithm_name == "poisson_disk" and packing_fraction > .18:
        preflight_warnings.append("Poisson-disk may become slow at this density. Consider RSA or force-biased.")
    if spatial_order != "overlapping_random" and packing_fraction >= .64:
        preflight_warnings.append("Near-jamming density may require more time or a different algorithm; convergence is not guaranteed.")
    if max(lateral_length_um, medium_thickness_um) / min(lateral_length_um, medium_thickness_um) > 10:
        preflight_warnings.append("This box has an extreme aspect ratio. Inspect structural statistics and convergence carefully.")

    with tab_config:
        st.subheader("Configuration for current settings")
        st.code(yaml.safe_dump(config_data, sort_keys=False), language="yaml")
    with tab_basic:
        st.subheader("Review and generate")
        st.write("**Selected algorithm:**", _algorithm_display_name(effective_algorithm_name))
        st.write(f"**Fixed periodic box:** {lateral_length_um:.8g} × {lateral_length_um:.8g} × {medium_thickness_um:.8g} μm")
        if resolved:
            if auto_particles:
                st.number_input("Particle count", min_value=0, value=resolved.n_particles, disabled=True)
            st.write(f"**Resolved fraction:** {resolved.resolved_phi:.8g}; target {packing_fraction:.8g}; difference {resolved.resolved_phi - packing_fraction:+.3g}")
            if compute_sk:
                work = resolved.n_particles * ((2 * k_max_index + 1)**3 - 1)
                st.caption(f"S(k) analysis: {work:,} particle–wavevector evaluations.")
                if work > 100_000_000:
                    preflight_warnings.append("S(k) analysis is expensive for this configuration. Use Quick analysis or disable Compute S(k).")
            with preview_container:
                diameters = 2 * resolved.radii * mean_diameter_um
                fig, ax = plt.subplots(figsize=(6, 2.5))
                bins_preview, weights = _preview_histogram_settings(diameters)
                ax.hist(diameters, bins=bins_preview, weights=weights, color="#9ecae1")
                ax.set_xlabel("Diameter (μm)")
                ax.set_ylabel("Sample fraction (%)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Preview uses the resolved particles for the current seed and settings.")
        else:
            with preview_container:
                st.info("Fix the input errors to display the resolved size distribution.")
        for message in dict.fromkeys(preflight_errors):
            st.error(message)
        for message in dict.fromkeys(preflight_warnings):
            st.warning(message)
        st.caption(f"Output destination: {(ROOT / output_path).resolve()}. Each run receives a separate directory; existing results are preserved.")
        active_job = st.session_state.get("gui_job")
        if st.button("Generate", type="primary", disabled=bool(preflight_errors) or bool(active_job)):
            try:
                requested = ROOT / output_path
                if timestamp_output and output_path == suggested_output_path:
                    requested = ROOT / _timestamped_output_path(project_name, _new_run_timestamp())
                destination = allocate_run_directory(requested)
                remember_run(ROOT, destination)
                run_config = deepcopy(config_data)
                run_config["output"]["path"] = str(destination)
                if custom_size_file is not None:
                    saved_size_file = _save_uploaded_size_file(custom_size_file, str(destination))
                    run_config["size_distribution"]["size_file"] = str(saved_size_file)
                config_path = _write_gui_config(run_config, str(destination))
                job = start_job(config_path, int(run_timeout_s))
                job["signature"] = _config_signature(config_data)
                st.session_state["gui_job"] = job
                st.session_state.pop("gui_last_error", None)
                st.info("Generation started. Progress and results are shown in the Results tab.")
            except Exception as exc:
                st.error(f"Could not start generation: {exc}")
    with tab_results:
        _live_results(_config_signature(config_data))
    with tab_history:
        if selected_history:
            _show_run_result(ROOT / selected_history, "Historical Result")
        else:
            st.info("Select a saved result from the sidebar to view its parameters, validation, plots and downloads.")


if __name__ == "__main__":
    main()
