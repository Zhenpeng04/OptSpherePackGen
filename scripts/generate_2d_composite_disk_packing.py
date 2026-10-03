"""Generate a 2D composite disk packing from three monodisperse disk sizes.

Default case:
    200 nm disks: 17189
    400 nm disks: 4297
    600 nm disks: 1910
    box = 30 x 90 um
    total target area fraction ~= 0.6

The implementation reuses the 2D force-biased periodic packing functions from
generate_2d_disk_packing.py, but builds a mixed radius/species array.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_2d_disk_packing import (  # noqa: E402
    DEFAULT_OUTER_DIAMETER_RATIO,
    ValidationReport,
    area_fraction,
    find_overlap_pairs,
    json_ready,
    poisson_disk_initialize,
    relax_stage,
)


DEFAULT_OUTPUT = Path("results/disk2d_composite_200_400_600nm_phi060_30x90um")


def build_radii_and_species(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    diameters = np.asarray(args.diameters_um, dtype=float)
    counts = np.asarray(args.counts, dtype=int)
    if diameters.shape != counts.shape:
        raise ValueError("--diameters-um and --counts must have the same length.")
    if np.any(diameters <= 0.0):
        raise ValueError("All diameters must be positive.")
    if np.any(counts <= 0):
        raise ValueError("All counts must be positive.")

    radii_parts = []
    species_parts = []
    for species, (diameter, count) in enumerate(zip(diameters, counts)):
        radii_parts.append(np.full(int(count), 0.5 * float(diameter), dtype=float))
        species_parts.append(np.full(int(count), int(species), dtype=np.int32))
    return np.concatenate(radii_parts), np.concatenate(species_parts)


def generate_composite(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    start = time.perf_counter()
    box_lengths = np.asarray(args.box, dtype=float)
    radii, species_id = build_radii_and_species(args)
    rng = np.random.default_rng(args.seed)

    # Randomize species order before the size-sorted initialization. The placement
    # routine still places larger disks first for robustness, then restores this
    # randomized order in the returned coordinates.
    permutation = rng.permutation(len(radii))
    radii = radii[permutation]
    species_id = species_id[permutation]

    initial_fraction = float(args.initial_radius_fraction)
    if not 0.0 < initial_fraction <= 1.0:
        raise ValueError("--initial-radius-fraction must be in (0, 1].")
    start_radii = radii * initial_fraction
    positions, init_diag = poisson_disk_initialize(
        start_radii,
        box_lengths,
        rng,
        max_attempts_per_disk=int(args.max_attempts_per_disk),
        candidates_per_disk=int(args.candidates_per_disk),
        progress_every=int(args.progress_every),
    )

    mean_diameter = 2.0 * float(np.mean(radii))
    verlet_skin = float(args.verlet_skin)
    if verlet_skin <= 0.0:
        verlet_skin = float(args.verlet_skin_fraction) * mean_diameter
    if verlet_skin <= 0.0:
        raise ValueError("Verlet skin must be positive.")

    stage_diagnostics = []
    for stage in range(1, int(args.stages) + 1):
        frac = stage / float(args.stages)
        stage_radii = start_radii + frac * (radii - start_radii)
        positions, diag = relax_stage(
            positions,
            stage_radii,
            box_lengths,
            rng,
            max_iterations=int(args.max_iterations_per_stage),
            initial_outer_ratio=float(args.outer_diameter_ratio),
            contraction_rate=float(args.contraction_rate),
            force_scaling_factor=float(args.force_scaling_factor),
            max_displacement_fraction=float(args.max_displacement_fraction),
            tolerance_overlap=float(args.intermediate_tolerance_overlap),
            verlet_skin=verlet_skin,
        )
        stage_diagnostics.append(
            {
                "stage": int(stage),
                "radius_fraction": float(initial_fraction + frac * (1.0 - initial_fraction)),
                **diag,
            }
        )
        print(
            f"stage {stage}/{args.stages}: "
            f"max_overlap={diag['max_overlap_um']:.3e} um, "
            f"overlap_count={diag['overlap_count']}, stop={diag['stop_reason']}"
        )

    positions, final_diag = relax_stage(
        positions,
        radii,
        box_lengths,
        rng,
        max_iterations=int(args.final_cleanup_steps),
        initial_outer_ratio=1.0,
        contraction_rate=float(args.contraction_rate),
        force_scaling_factor=float(args.force_scaling_factor),
        max_displacement_fraction=float(args.max_displacement_fraction),
        tolerance_overlap=float(args.tolerance_overlap),
        verlet_skin=verlet_skin,
    )
    diagnostics = {
        "name": "force_biased_2d_periodic_composite",
        "status": "success" if final_diag["max_overlap_um"] <= float(args.tolerance_overlap) else "warning",
        "elapsed_time_s": float(time.perf_counter() - start),
        "note": (
            "2D composite force-biased disk packing: mixed radii, reduced-radius "
            "Poisson-disk initialization, staged radius growth, and periodic "
            "outer-shell repulsive relaxation."
        ),
        "initialization": init_diag,
        "stages": int(args.stages),
        "initial_radius_fraction": float(initial_fraction),
        "max_iterations_per_stage": int(args.max_iterations_per_stage),
        "final_cleanup_steps": int(args.final_cleanup_steps),
        "outer_diameter_ratio_initial": float(args.outer_diameter_ratio),
        "contraction_rate": float(args.contraction_rate),
        "force_scaling_factor": float(args.force_scaling_factor),
        "max_displacement_fraction": float(args.max_displacement_fraction),
        "verlet_skin_um": float(verlet_skin),
        "final_cleanup": final_diag,
        "stage_diagnostics": stage_diagnostics,
    }
    return np.mod(positions, box_lengths), radii, species_id, diagnostics


def validate_composite(
    positions: np.ndarray,
    radii: np.ndarray,
    species_id: np.ndarray,
    box_lengths: np.ndarray,
    target_area_fraction: float,
    tolerance_overlap: float,
    tolerance_area_fraction: float,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    out_of_bounds = int(np.sum((positions < -1.0e-12) | (positions >= box_lengths + 1.0e-12)))
    if out_of_bounds:
        errors.append(f"{out_of_bounds} coordinate values are outside the 2D box.")

    overlap_pairs, min_gap = find_overlap_pairs(positions, radii, box_lengths, tolerance_overlap)
    if overlap_pairs:
        errors.append(f"{len(overlap_pairs)} overlapping disk pairs exceed tolerance.")

    actual_area_fraction = area_fraction(radii, box_lengths)
    area_fraction_error = abs(actual_area_fraction - float(target_area_fraction))
    if area_fraction_error > tolerance_area_fraction:
        warnings.append(
            f"Area fraction differs from target by {area_fraction_error:.3e}; "
            "integer disk counts from source packings make exact equality impossible."
        )
    max_overlap = max(0.0, -float(min_gap)) if np.isfinite(min_gap) else 0.0
    metrics: dict[str, float | int | list[float]] = {
        "n_disks": int(len(radii)),
        "box_lengths_um": [float(v) for v in box_lengths],
        "box_area_um2": float(np.prod(box_lengths)),
        "target_area_fraction": float(target_area_fraction),
        "actual_area_fraction": float(actual_area_fraction),
        "area_fraction_error": float(area_fraction_error),
        "radius_min_um": float(np.min(radii)),
        "radius_max_um": float(np.max(radii)),
        "min_gap_um": float(min_gap),
        "max_overlap_um": float(max_overlap),
        "overlap_pair_count": int(len(overlap_pairs)),
        "out_of_bounds_value_count": int(out_of_bounds),
    }
    for species in sorted(Counter(species_id).keys()):
        mask = species_id == species
        species_area = float(np.sum(np.pi * radii[mask] ** 2))
        diameter_um = 2.0 * float(np.mean(radii[mask]))
        metrics[f"species_{int(species)}_count"] = int(np.count_nonzero(mask))
        metrics[f"species_{int(species)}_diameter_um"] = float(diameter_um)
        metrics[f"species_{int(species)}_area_fraction"] = species_area / float(np.prod(box_lengths))
    return ValidationReport(passed=not errors, errors=errors, warnings=warnings, metrics=metrics)


def write_particles_csv(
    path: Path,
    positions: np.ndarray,
    radii: np.ndarray,
    species_id: np.ndarray,
):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["disk_id", "x", "y", "radius", "diameter", "species_id", "material_id"])
        for i, (position, radius) in enumerate(zip(positions, radii)):
            writer.writerow(
                [
                    i,
                    f"{position[0]:.16g}",
                    f"{position[1]:.16g}",
                    f"{radius:.16g}",
                    f"{2.0 * radius:.16g}",
                    int(species_id[i]),
                    int(species_id[i]),
                ]
            )


def write_hdf5(
    path: Path,
    positions: np.ndarray,
    radii: np.ndarray,
    species_id: np.ndarray,
    box_lengths: np.ndarray,
) -> bool:
    try:
        import h5py
    except Exception:
        return False
    with h5py.File(path, "w") as h5:
        h5.create_dataset("positions", data=positions)
        h5.create_dataset("radii", data=radii)
        h5.create_dataset("species_id", data=species_id)
        h5.attrs["coordinate_unit"] = "um"
        h5.attrs["box_lengths_um"] = np.asarray(box_lengths, dtype=float)
        h5.attrs["boundary"] = "periodic_2d"
    return True


def write_plot(
    path: Path,
    positions: np.ndarray,
    radii: np.ndarray,
    species_id: np.ndarray,
    box_lengths: np.ndarray,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle, Rectangle
    except Exception:
        return False

    palette = {
        0: "#6baed6",
        1: "#74c476",
        2: "#fd8d3c",
    }
    fig, ax = plt.subplots(figsize=(5.5, 16), dpi=180)
    for species in sorted(Counter(species_id).keys()):
        mask = species_id == species
        patches = [
            Circle((float(x), float(y)), float(r))
            for (x, y), r in zip(positions[mask], radii[mask])
        ]
        collection = PatchCollection(
            patches,
            facecolor=palette.get(int(species), "#9e9ac8"),
            edgecolor="white",
            linewidth=0.05,
            alpha=0.80,
            label=f"species {int(species)}",
        )
        ax.add_collection(collection)
    ax.add_patch(
        Rectangle((0.0, 0.0), float(box_lengths[0]), float(box_lengths[1]), fill=False, edgecolor="black", linewidth=1.0)
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, float(box_lengths[0]))
    ax.set_ylim(0.0, float(box_lengths[1]))
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("2D composite disk packing")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_outputs(
    output_dir: Path,
    positions: np.ndarray,
    radii: np.ndarray,
    species_id: np.ndarray,
    diagnostics: dict,
    validation: ValidationReport,
    args: argparse.Namespace,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    box_lengths = np.asarray(args.box, dtype=float)
    write_particles_csv(output_dir / "particles_2d_real_units.csv", positions, radii, species_id)
    hdf5_written = write_hdf5(output_dir / "particles_2d.h5", positions, radii, species_id, box_lengths)
    plot_written = write_plot(output_dir / "packing_2d.png", positions, radii, species_id, box_lengths) if args.make_plot else False
    species_info = []
    diameters = np.asarray(args.diameters_um, dtype=float)
    counts = np.asarray(args.counts, dtype=int)
    for species, (diameter, count) in enumerate(zip(diameters, counts)):
        species_info.append(
            {
                "species_id": int(species),
                "diameter_um": float(diameter),
                "diameter_nm": float(diameter * 1000.0),
                "count": int(count),
            }
        )
    metadata = {
        "project": {
            "name": output_dir.name,
            "description": (
                "2D composite disk packing with 200 nm, 400 nm, and 600 nm disks "
                "mixed in a 30 x 90 um periodic rectangle at total area fraction 0.6."
            ),
        },
        "physical": {
            "box_lengths_um": [float(v) for v in box_lengths],
            "target_area_fraction": float(args.area_fraction),
            "area_fraction_interpretation": "2D analogue of volume fraction",
            "species": species_info,
        },
        "generator": diagnostics,
        "validation": asdict(validation),
        "outputs": {
            "particles_csv": "particles_2d_real_units.csv",
            "particles_h5": "particles_2d.h5" if hdf5_written else None,
            "plot": "packing_2d.png" if plot_written else None,
            "metadata": "metadata.json",
            "validation_report": "validation_report.json",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
    (output_dir / "validation_report.json").write_text(
        json.dumps(json_ready(asdict(validation)), indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 2D composite disk packing.")
    parser.add_argument("--diameters-um", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--counts", type=int, nargs="+", default=[17189, 4297, 1910])
    parser.add_argument("--box", type=float, nargs=2, default=[30.0, 90.0], metavar=("LX", "LY"))
    parser.add_argument("--area-fraction", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stages", type=int, default=28)
    parser.add_argument("--initial-radius-fraction", type=float, default=0.35)
    parser.add_argument("--max-iterations-per-stage", type=int, default=600)
    parser.add_argument("--final-cleanup-steps", type=int, default=1000)
    parser.add_argument("--force-scaling-factor", type=float, default=0.5)
    parser.add_argument("--outer-diameter-ratio", type=float, default=DEFAULT_OUTER_DIAMETER_RATIO)
    parser.add_argument("--contraction-rate", type=float, default=1.0e-3)
    parser.add_argument("--max-displacement-fraction", type=float, default=0.30)
    parser.add_argument("--verlet-skin", type=float, default=0.0, help="Absolute skin in um; 0 uses --verlet-skin-fraction.")
    parser.add_argument("--verlet-skin-fraction", type=float, default=0.5)
    parser.add_argument("--max-attempts-per-disk", type=int, default=30000)
    parser.add_argument("--candidates-per-disk", type=int, default=24)
    parser.add_argument("--tolerance-overlap", type=float, default=1.0e-9)
    parser.add_argument("--intermediate-tolerance-overlap", type=float, default=1.0e-7)
    parser.add_argument("--tolerance-area-fraction", type=float, default=5.0e-4)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--make-plot", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    box_lengths = np.asarray(args.box, dtype=float)
    if box_lengths.shape != (2,) or np.any(box_lengths <= 0.0):
        raise ValueError("--box must contain two positive lengths in um.")
    radii, species_id = build_radii_and_species(args)
    phi = area_fraction(radii, box_lengths)
    print(f"resolved disks: {len(radii)}")
    for species in sorted(Counter(species_id).keys()):
        mask = species_id == species
        print(
            f"species {int(species)}: count={int(np.count_nonzero(mask))}, "
            f"diameter={2.0 * float(np.mean(radii[mask])):g} um"
        )
    print(f"box: {box_lengths.tolist()} um")
    print(f"target area fraction: {args.area_fraction:g}")
    print(f"actual area fraction from selected counts: {phi:.12g}")
    if args.dry_run:
        return 0

    positions, radii, species_id, diagnostics = generate_composite(args)
    validation = validate_composite(
        positions,
        radii,
        species_id,
        box_lengths,
        float(args.area_fraction),
        float(args.tolerance_overlap),
        float(args.tolerance_area_fraction),
    )
    write_outputs(args.output, positions, radii, species_id, diagnostics, validation, args)
    print(f"output: {args.output}")
    print(f"validation_passed: {validation.passed}")
    print(f"actual_area_fraction: {validation.metrics['actual_area_fraction']:.12g}")
    print(f"max_overlap_um: {validation.metrics['max_overlap_um']:.3e}")
    print(f"min_gap_um: {validation.metrics['min_gap_um']:.3e}")
    return 0 if validation.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
