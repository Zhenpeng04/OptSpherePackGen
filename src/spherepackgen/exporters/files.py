"""Result exporters."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import yaml

from spherepackgen.config.schema import PackingConfig
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.result import ResultBundle, StructuralDescriptors
from spherepackgen.visualization.packing_views import render_packing_views, DEFAULT_MAX_RENDER_PARTICLES
from spherepackgen.provenance import sha256_file

PLOT_G2_SMOOTH_SIGMA = 2.0
PLOT_SK_SMOOTH_SIGMA = 3.0


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_particles_csv(path: Path, particles: ParticleSet):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_id", "x", "y", "z", "radius", "diameter", "species_id", "material_id"])
        for i, (pos, radius) in enumerate(zip(particles.positions, particles.radii)):
            writer.writerow([
                i,
                f"{pos[0]:.16g}",
                f"{pos[1]:.16g}",
                f"{pos[2]:.16g}",
                f"{radius:.16g}",
                f"{2.0 * radius:.16g}",
                particles.species_id[i],
                particles.material_id[i],
            ])


def _write_series_csv(path: Path, headers: list[str], columns: list[np.ndarray]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in zip(*columns):
            writer.writerow([f"{float(v):.16g}" if isinstance(v, (float, np.floating)) else v for v in row])


def _save_hdf5(path: Path, particles: ParticleSet, descriptors: StructuralDescriptors):
    try:
        import h5py
    except Exception:
        return False
    with h5py.File(path, "w") as h5:
        h5.create_dataset("positions", data=particles.positions)
        h5.create_dataset("radii", data=particles.radii)
        h5.attrs["box_lengths"] = particles.box_lengths
        h5.attrs["box_volume"] = particles.domain.volume
        if np.ndim(particles.box_length) == 0:
            h5.attrs["box_length"] = particles.box_length
        if descriptors.g2:
            grp = h5.create_group("g2")
            grp.create_dataset("r", data=descriptors.g2["r"])
            grp.create_dataset("g2", data=descriptors.g2["g2"])
        if descriptors.Sk:
            grp = h5.create_group("Sk")
            grp.create_dataset("k", data=descriptors.Sk["k"])
            grp.create_dataset("Sk", data=descriptors.Sk["Sk"])
            grp.create_dataset("shell_count", data=descriptors.Sk["shell_count"])
    return True


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if sigma <= 0.0 or arr.size < 3:
        return arr.copy()

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    if not np.all(finite):
        idx = np.arange(arr.size)
        arr = arr.copy()
        arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])

    radius = max(1, int(np.ceil(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(arr, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def _make_plots(out_dir: Path, particles: ParticleSet, descriptors: StructuralDescriptors, timings=None):
    start = perf_counter()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    files = []
    plt.figure()
    diameter_scale = 1.0 if timings is None else timings.get("diameter_scale", 1.0)
    diameter_unit = "dimensionless" if timings is None else timings.get("diameter_unit", "dimensionless")
    plt.hist(2 * particles.radii * diameter_scale, bins=30,
             weights=np.full(particles.n_particles, 100.0 / particles.n_particles))
    plt.xlabel(f"Diameter ({diameter_unit})")
    plt.ylabel("Sample fraction (%)")
    path = fig_dir / "radius_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    files.append(str(path))
    if descriptors.g2:
        r_values = np.asarray(descriptors.g2["r"], dtype=float)
        g2_values = _gaussian_smooth(descriptors.g2["g2"], PLOT_G2_SMOOTH_SIGMA)
        plt.figure()
        plt.plot(r_values, g2_values, linewidth=1.8)
        plt.xlabel("r")
        plt.ylabel("g2(r)")
        path = fig_dir / "g2.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        files.append(str(path))
    if descriptors.Sk:
        k_values = np.asarray(descriptors.Sk["k"], dtype=float)
        sk_values = _gaussian_smooth(descriptors.Sk["Sk"], PLOT_SK_SMOOTH_SIGMA)
        plt.figure()
        plt.plot(k_values, sk_values, linewidth=1.8)
        plt.xlabel("k")
        plt.ylabel("S(k)")
        path = fig_dir / "Sk.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        files.append(str(path))
    if timings is not None:
        timings["analysis_plots"] = perf_counter() - start
    start = perf_counter()
    files.extend(render_packing_views(particles, fig_dir))
    if timings is not None:
        timings["rendering"] = perf_counter() - start
    return files


def export_bundle(bundle: ResultBundle) -> list[str]:
    start = perf_counter()
    config: PackingConfig = bundle.config
    out_dir = Path(config.output.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    exported: list[str] = []

    particle_set = bundle.result.particle_set
    particle_set.validate_geometry()
    real_scale = bundle.resolved.mean_diameter_m / bundle.resolved.output_unit_m
    real_particles = particle_set.scaled(real_scale)

    # Preserve exact floating-point geometry even when coordinate CSVs are disabled.
    snapshot = out_dir / "particles_snapshot.npz"
    np.savez(snapshot, positions=particle_set.positions, radii=particle_set.radii,
             box_lengths=particle_set.box_lengths, species_id=particle_set.species_id,
             material_id=particle_set.material_id)
    exported.append(str(snapshot))
    bundle.provenance["snapshot"] = {"path": snapshot.name, "sha256": sha256_file(snapshot),
                                     "coordinate_unit": "dimensionless", "mean_diameter_m": bundle.resolved.mean_diameter_m}
    for item in bundle.provenance.get("inputs", []):
        exported.append(str(out_dir / item["path"]))
    environment_path = out_dir / "environment.json"
    environment_path.write_text(json.dumps(_json_ready({key: bundle.provenance.get(key) for key in ("software", "environment")}), indent=2), encoding="utf-8")
    exported.append(str(environment_path))

    if config.output.save_dimensionless_coordinates:
        path = out_dir / "particles_dimensionless.csv"
        _write_particles_csv(path, particle_set)
        exported.append(str(path))
    if config.output.save_real_coordinates:
        path = out_dir / "particles_real_units.csv"
        _write_particles_csv(path, real_particles)
        exported.append(str(path))

    metadata = {
        "readiness": bundle.readiness,
        "provenance": bundle.provenance,
        "project": asdict(config.project),
        "classification": {
            "size_distribution": config.structure.size_distribution.value,
            "spatial_order": config.structure.spatial_order.value,
            "density_class": config.structure.density_class.value,
        },
        "resolved_parameters": bundle.resolved.to_json_dict(),
        "generator": {
            "name": bundle.result.generator_name,
            "status": bundle.result.status,
            "elapsed_time_s": bundle.result.elapsed_time_s,
            "diagnostics": bundle.result.diagnostics,
        },
        "validation": asdict(bundle.validation),
    }
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    exported.append(str(path))

    path = out_dir / "validation_report.json"
    path.write_text(json.dumps(_json_ready(asdict(bundle.validation)), indent=2), encoding="utf-8")
    exported.append(str(path))

    path = out_dir / "config_input.yaml"
    path.write_text(yaml.safe_dump(_json_ready(config.raw), sort_keys=False), encoding="utf-8")
    exported.append(str(path))

    path = out_dir / "config_resolved.yaml"
    path.write_text(yaml.safe_dump(_json_ready(bundle.resolved.to_json_dict()), sort_keys=False), encoding="utf-8")
    exported.append(str(path))

    # A loadable configuration, unlike the diagnostic resolved-parameters YAML.
    replay = _replay_config(config, out_dir)
    path = out_dir / "config_replay.yaml"
    path.write_text(yaml.safe_dump(_json_ready(replay), sort_keys=False), encoding="utf-8")
    exported.append(str(path))

    if bundle.descriptors.nearest_neighbor:
        nn = bundle.descriptors.nearest_neighbor["distances"]
        _write_series_csv(out_dir / "analysis" / "nearest_neighbor.csv", ["distance"], [nn])
        exported.append(str(out_dir / "analysis" / "nearest_neighbor.csv"))
    if bundle.descriptors.g2:
        _write_series_csv(out_dir / "analysis" / "g2.csv", ["r", "g2"], [bundle.descriptors.g2["r"], bundle.descriptors.g2["g2"]])
        exported.append(str(out_dir / "analysis" / "g2.csv"))
    if bundle.descriptors.Sk:
        _write_series_csv(
            out_dir / "analysis" / "Sk.csv",
            ["k", "Sk", "shell_count"],
            [bundle.descriptors.Sk["k"], bundle.descriptors.Sk["Sk"], bundle.descriptors.Sk["shell_count"]],
        )
        exported.append(str(out_dir / "analysis" / "Sk.csv"))

    if "hdf5" in {fmt.lower() for fmt in config.output.formats}:
        h5_path = out_dir / "particles.h5"
        if _save_hdf5(h5_path, particle_set, bundle.descriptors):
            exported.append(str(h5_path))
        else:
            raise RuntimeError("HDF5 was requested but h5py is unavailable. Install h5py or disable Save HDF5.")

    bundle.stage_timings_s["analysis_plots"] = 0.0
    bundle.stage_timings_s["rendering"] = 0.0
    if config.output.make_plots:
        plot_timings = {"diameter_scale": real_scale, "diameter_unit": bundle.resolved.output_unit}
        exported.extend(_make_plots(out_dir, particle_set, bundle.descriptors, plot_timings))
        for key in ("analysis_plots", "rendering"):
            bundle.stage_timings_s[key] = plot_timings[key]

    log_path = out_dir / "logs" / "run.log"
    log_path.write_text(
        "\n".join(
            [
                f"case={config.project.name}",
                f"generator={bundle.result.generator_name}",
                f"status={bundle.result.status}",
                f"validation_passed={bundle.validation.passed}",
                f"errors={bundle.validation.errors}",
                f"warnings={bundle.validation.warnings}",
            ]
        ),
        encoding="utf-8",
    )
    exported.append(str(log_path))
    bundle.stage_timings_s["export"] = max(0.0, perf_counter() - start - bundle.stage_timings_s["analysis_plots"] - bundle.stage_timings_s["rendering"])
    metadata["stage_timings_s"] = bundle.stage_timings_s
    (out_dir / "metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    manifest_path = out_dir / "run_manifest.json"
    manifest = {
        "readiness": bundle.readiness,
        "files": [Path(item).resolve().relative_to(out_dir.resolve()).as_posix() for item in exported]
                 + ["run_manifest.json", "run_bundle.zip"],
        "sha256": {Path(item).resolve().relative_to(out_dir.resolve()).as_posix(): sha256_file(Path(item)) for item in exported},
        "integrity_note": "Checksums cover payload files; the manifest and archive are excluded to avoid recursive hashing.",
        "rendered_particle_count": min(particle_set.n_particles, DEFAULT_MAX_RENDER_PARTICLES)
            if config.output.make_plots else 0,
        "particle_count": particle_set.n_particles,
        "status": "failed" if not bundle.validation.passed else bundle.result.status,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    exported.append(str(manifest_path))
    archive = out_dir / "run_bundle.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        for item in exported:
            path = Path(item)
            zipped.write(path, path.resolve().relative_to(out_dir.resolve()).as_posix())
    exported.append(str(archive))
    return exported


def _replay_config(config: PackingConfig, out_dir: Path) -> dict:
    size_distribution = asdict(config.size_distribution)
    if size_distribution["size_file"]:
        size_distribution["size_file"] = Path(size_distribution["size_file"]).resolve().relative_to(out_dir.resolve()).as_posix()
    physical = {"mean_diameter": f"{config.physical.mean_diameter_m:.17g} m",
                "packing_fraction": config.physical.packing_fraction}
    if config.physical.covered_volume_fraction is not None:
        physical["covered_volume_fraction"] = config.physical.covered_volume_fraction
    domain = {"type": "periodic_box" if config.domain.fixed_dimensions else "periodic_cube"}
    if config.domain.fixed_dimensions:
        domain.update(length=f"{config.domain.length_m:.17g} m", depth=f"{config.domain.depth_m:.17g} m")
    else:
        if config.physical.medium_thickness_m is not None:
            physical["medium_thickness"] = f"{config.physical.medium_thickness_m:.17g} m"
        if config.domain.box_size_m is not None:
            domain["box_size"] = f"{config.domain.box_size_m:.17g} m"
    return {
        "project": asdict(config.project), "structure": _json_ready(asdict(config.structure)),
        "physical": physical, "domain": domain, "particles": asdict(config.particles),
        "size_distribution": size_distribution, "algorithm": asdict(config.algorithm),
        "analysis": asdict(config.analysis), "validation": asdict(config.validation),
        "output": {**asdict(config.output), "path": "replay", "relative_to_config": True, "overwrite": False},
        "runtime": asdict(config.runtime),
    }
