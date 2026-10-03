"""Generate a 2D disk packing with truncated-normal disk diameters.

Default case:
    diameter range = 0.2 to 0.6 um
    mean diameter = 0.4 um
    sigma = (0.6 - 0.2) / 6 um, so the range is approximately +/- 3 sigma
    box = 30 x 90 um
    total target area fraction = 0.6

The script reuses the same 2D force-biased periodic packing functions used by
the monodisperse and composite 2D generators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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


DEFAULT_OUTPUT = Path("results/disk2d_normal_200_600nm_mean400nm_phi060_30x90um")


def truncated_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    sigma: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    values = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        batch = rng.normal(mean, sigma, size=max(size - filled, 1024))
        batch = batch[(batch >= low) & (batch <= high)]
        n = min(len(batch), size - filled)
        if n:
            values[filled : filled + n] = batch[:n]
            filled += n
    return values


def recenter_with_bounds(values: np.ndarray, *, target_mean: float, low: float, high: float, max_iter: int = 50) -> np.ndarray:
    """Shift samples so their mean matches the target while preserving bounds."""
    out = np.asarray(values, dtype=float).copy()
    for _ in range(max_iter):
        delta = target_mean - float(np.mean(out))
        if abs(delta) < 1.0e-12:
            break
        out = np.clip(out + delta, low, high)
    return out


def sample_diameters(args: argparse.Namespace, n_disks: int, rng: np.random.Generator) -> np.ndarray:
    diameters = truncated_normal(
        rng,
        mean=float(args.mean_diameter_um),
        sigma=float(args.sigma_diameter_um),
        low=float(args.min_diameter_um),
        high=float(args.max_diameter_um),
        size=int(n_disks),
    )
    if args.force_sample_mean:
        diameters = recenter_with_bounds(
            diameters,
            target_mean=float(args.mean_diameter_um),
            low=float(args.min_diameter_um),
            high=float(args.max_diameter_um),
        )
    return diameters


def estimate_disk_count(args: argparse.Namespace, box_lengths: np.ndarray, rng: np.random.Generator) -> int:
    probe = sample_diameters(args, int(args.count_probe_samples), rng)
    mean_disk_area = float(np.mean(np.pi * (0.5 * probe) ** 2))
    target_area = float(args.area_fraction) * float(np.prod(box_lengths))
    return max(1, int(round(target_area / mean_disk_area)))


def build_radii(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    box_lengths = np.asarray(args.box, dtype=float)
    rng = np.random.default_rng(args.seed)
    n_disks = int(args.num_disks) if args.num_disks else estimate_disk_count(args, box_lengths, rng)

    best_diameters = None
    best_error = float("inf")
    best_phi = float("nan")
    for _ in range(max(1, int(args.count_refinement_attempts))):
        diameters = sample_diameters(args, n_disks, rng)
        radii = 0.5 * diameters
        phi = area_fraction(radii, box_lengths)
        error = abs(phi - float(args.area_fraction))
        if error < best_error:
            best_diameters = diameters
            best_error = error
            best_phi = phi
        if error <= float(args.tolerance_area_fraction):
            break

    if best_diameters is None:
        raise RuntimeError("Could not sample disk diameters.")
    radii = 0.5 * best_diameters
    stats = {
        "n_disks": int(n_disks),
        "diameter_min_um": float(np.min(best_diameters)),
        "diameter_max_um": float(np.max(best_diameters)),
        "diameter_mean_um": float(np.mean(best_diameters)),
        "diameter_std_um": float(np.std(best_diameters)),
        "diameter_cv": float(np.std(best_diameters) / np.mean(best_diameters)),
        "area_fraction_from_sampled_sizes": float(best_phi),
        "area_fraction_error_from_sampled_sizes": float(best_error),
    }
    return radii, stats


def generate_normal_packing(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    start = time.perf_counter()
    box_lengths = np.asarray(args.box, dtype=float)
    radii, size_stats = build_radii(args)
    rng = np.random.default_rng(int(args.seed) + 1)

    # Randomize the sampled size order before larger-first initialization.
    permutation = rng.permutation(len(radii))
    radii = radii[permutation]

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
        "name": "force_biased_2d_periodic_truncated_normal",
        "status": "success" if final_diag["max_overlap_um"] <= float(args.tolerance_overlap) else "warning",
        "elapsed_time_s": float(time.perf_counter() - start),
        "note": (
            "2D polydisperse force-biased disk packing: truncated-normal diameters, "
            "reduced-radius Poisson-disk initialization, staged radius growth, and "
            "periodic outer-shell repulsive relaxation."
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
    return np.mod(positions, box_lengths), radii, size_stats, diagnostics


def validate_normal_packing(
    positions: np.ndarray,
    radii: np.ndarray,
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
            "integer disk count and sampled diameters make exact equality impossible."
        )
    diameters = 2.0 * radii
    max_overlap = max(0.0, -float(min_gap)) if np.isfinite(min_gap) else 0.0
    return ValidationReport(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        metrics={
            "n_disks": int(len(radii)),
            "box_lengths_um": [float(v) for v in box_lengths],
            "box_area_um2": float(np.prod(box_lengths)),
            "target_area_fraction": float(target_area_fraction),
            "actual_area_fraction": float(actual_area_fraction),
            "area_fraction_error": float(area_fraction_error),
            "diameter_min_um": float(np.min(diameters)),
            "diameter_max_um": float(np.max(diameters)),
            "diameter_mean_um": float(np.mean(diameters)),
            "diameter_std_um": float(np.std(diameters)),
            "diameter_cv": float(np.std(diameters) / np.mean(diameters)),
            "radius_min_um": float(np.min(radii)),
            "radius_max_um": float(np.max(radii)),
            "min_gap_um": float(min_gap),
            "max_overlap_um": float(max_overlap),
            "overlap_pair_count": int(len(overlap_pairs)),
            "out_of_bounds_value_count": int(out_of_bounds),
        },
    )


def write_particles_csv(path: Path, positions: np.ndarray, radii: np.ndarray):
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
                    0,
                    0,
                ]
            )


def write_hdf5(path: Path, positions: np.ndarray, radii: np.ndarray, box_lengths: np.ndarray) -> bool:
    try:
        import h5py
    except Exception:
        return False
    with h5py.File(path, "w") as h5:
        h5.create_dataset("positions", data=positions)
        h5.create_dataset("radii", data=radii)
        h5.create_dataset("diameters", data=2.0 * radii)
        h5.attrs["coordinate_unit"] = "um"
        h5.attrs["box_lengths_um"] = np.asarray(box_lengths, dtype=float)
        h5.attrs["boundary"] = "periodic_2d"
        h5.attrs["diameter_distribution"] = "truncated_normal"
    return True


def write_plot(path: Path, positions: np.ndarray, radii: np.ndarray, box_lengths: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle, Rectangle
    except Exception:
        return False

    diameters = 2.0 * radii
    fig, ax = plt.subplots(figsize=(5.5, 16), dpi=180)
    patches = [Circle((float(x), float(y)), float(r)) for (x, y), r in zip(positions, radii)]
    collection = PatchCollection(
        patches,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.04,
        alpha=0.82,
    )
    collection.set_array(diameters)
    collection.set_clim(float(np.min(diameters)), float(np.max(diameters)))
    ax.add_collection(collection)
    ax.add_patch(
        Rectangle((0.0, 0.0), float(box_lengths[0]), float(box_lengths[1]), fill=False, edgecolor="black", linewidth=1.0)
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, float(box_lengths[0]))
    ax.set_ylim(0.0, float(box_lengths[1]))
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("2D normal-distributed disk packing")
    cbar = fig.colorbar(collection, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("diameter (um)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_diameter_histogram(path: Path, radii: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    diameters_nm = 2000.0 * radii
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    ax.hist(diameters_nm, bins=50, color="#74a9cf", edgecolor="white")
    ax.set_xlabel("diameter (nm)")
    ax.set_ylabel("count")
    ax.set_title("Diameter distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_outputs(
    output_dir: Path,
    positions: np.ndarray,
    radii: np.ndarray,
    size_stats: dict,
    diagnostics: dict,
    validation: ValidationReport,
    args: argparse.Namespace,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    box_lengths = np.asarray(args.box, dtype=float)
    write_particles_csv(output_dir / "particles_2d_real_units.csv", positions, radii)
    hdf5_written = write_hdf5(output_dir / "particles_2d.h5", positions, radii, box_lengths)
    plot_written = write_plot(output_dir / "packing_2d.png", positions, radii, box_lengths) if args.make_plot else False
    hist_written = write_diameter_histogram(output_dir / "diameter_distribution.png", radii) if args.make_plot else False
    metadata = {
        "project": {
            "name": output_dir.name,
            "description": (
                "2D disk packing with truncated-normal diameters from 200 to 600 nm, "
                "mean 400 nm, in a 30 x 90 um periodic rectangle at area fraction 0.6."
            ),
        },
        "physical": {
            "box_lengths_um": [float(v) for v in box_lengths],
            "target_area_fraction": float(args.area_fraction),
            "area_fraction_interpretation": "2D analogue of volume fraction",
            "diameter_distribution": {
                "type": "truncated_normal",
                "min_diameter_um": float(args.min_diameter_um),
                "max_diameter_um": float(args.max_diameter_um),
                "mean_diameter_um": float(args.mean_diameter_um),
                "sigma_diameter_um": float(args.sigma_diameter_um),
                "force_sample_mean": bool(args.force_sample_mean),
            },
            "sampled_size_statistics": size_stats,
        },
        "generator": diagnostics,
        "validation": asdict(validation),
        "outputs": {
            "particles_csv": "particles_2d_real_units.csv",
            "particles_h5": "particles_2d.h5" if hdf5_written else None,
            "plot": "packing_2d.png" if plot_written else None,
            "diameter_distribution_plot": "diameter_distribution.png" if hist_written else None,
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
    parser = argparse.ArgumentParser(description="Generate a 2D truncated-normal disk packing.")
    parser.add_argument("--min-diameter-um", type=float, default=0.2)
    parser.add_argument("--max-diameter-um", type=float, default=0.6)
    parser.add_argument("--mean-diameter-um", type=float, default=0.4)
    parser.add_argument("--sigma-diameter-um", type=float, default=(0.6 - 0.2) / 6.0)
    parser.add_argument("--force-sample-mean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-disks", type=int, default=0, help="0 means estimate from target area fraction.")
    parser.add_argument("--count-probe-samples", type=int, default=200000)
    parser.add_argument("--count-refinement-attempts", type=int, default=12)
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
    if not (0.0 < args.min_diameter_um < args.mean_diameter_um < args.max_diameter_um):
        raise ValueError("Require min_diameter < mean_diameter < max_diameter.")
    if args.area_fraction <= 0.0:
        raise ValueError("--area-fraction must be positive.")

    radii, size_stats = build_radii(args)
    print(f"resolved disks: {len(radii)}")
    print(f"box: {box_lengths.tolist()} um")
    print(
        "diameter distribution: "
        f"min={size_stats['diameter_min_um'] * 1000.0:.3f} nm, "
        f"max={size_stats['diameter_max_um'] * 1000.0:.3f} nm, "
        f"mean={size_stats['diameter_mean_um'] * 1000.0:.3f} nm, "
        f"std={size_stats['diameter_std_um'] * 1000.0:.3f} nm"
    )
    print(f"target area fraction: {args.area_fraction:g}")
    print(f"sampled-size area fraction: {size_stats['area_fraction_from_sampled_sizes']:.12g}")
    if args.dry_run:
        return 0

    positions, radii, size_stats, diagnostics = generate_normal_packing(args)
    validation = validate_normal_packing(
        positions,
        radii,
        box_lengths,
        float(args.area_fraction),
        float(args.tolerance_overlap),
        float(args.tolerance_area_fraction),
    )
    write_outputs(args.output, positions, radii, size_stats, diagnostics, validation, args)
    print(f"output: {args.output}")
    print(f"validation_passed: {validation.passed}")
    print(f"actual_area_fraction: {validation.metrics['actual_area_fraction']:.12g}")
    print(f"max_overlap_um: {validation.metrics['max_overlap_um']:.3e}")
    print(f"min_gap_um: {validation.metrics['min_gap_um']:.3e}")
    return 0 if validation.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
