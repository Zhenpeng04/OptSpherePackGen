"""Generate a 2D monodisperse disk packing.

Default case:
    diameter = 0.4 um
    box = 30 x 30 um
    target area fraction = 0.6

The script is independent of the SpherePackGen package. It uses a 2D periodic
force-biased relaxation workflow: reduced-radius non-overlapping initialization,
staged radius growth, and short-range repulsive relaxation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path("results/disk2d_monodisperse_400nm_phi060_30x30um")
NOMINAL_DENSITY_RATIO = 1.2
DEFAULT_OUTER_DIAMETER_RATIO = NOMINAL_DENSITY_RATIO ** 0.5


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, float | int | list[float]]


class PeriodicCellList2D:
    """Simple periodic cell list for 2D neighbor searches."""

    def __init__(self, box_lengths: np.ndarray, cutoff: float):
        self.box_lengths = np.asarray(box_lengths, dtype=float)
        self.cutoff = max(float(cutoff), 1.0e-12)
        self.n_cells = np.maximum(1, np.floor(self.box_lengths / self.cutoff).astype(int))
        self.cell_size = self.box_lengths / self.n_cells
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _cell_index(self, position: np.ndarray) -> tuple[int, int]:
        wrapped = np.mod(position, self.box_lengths)
        idx = np.floor(wrapped / self.cell_size).astype(int)
        idx = np.mod(idx, self.n_cells)
        return int(idx[0]), int(idx[1])

    def build(self, positions: np.ndarray):
        self.cells.clear()
        for i, position in enumerate(positions):
            self.cells[self._cell_index(position)].append(int(i))

    def add(self, index: int, position: np.ndarray):
        self.cells[self._cell_index(position)].append(int(index))

    def nearby_indices(self, position: np.ndarray, search_radius: float):
        span = np.maximum(1, np.ceil(float(search_radius) / self.cell_size).astype(int))
        base = self._cell_index(position)
        seen: set[int] = set()
        for dx in range(-int(span[0]), int(span[0]) + 1):
            for dy in range(-int(span[1]), int(span[1]) + 1):
                cell = (
                    (base[0] + dx) % int(self.n_cells[0]),
                    (base[1] + dy) % int(self.n_cells[1]),
                )
                for idx in self.cells.get(cell, []):
                    if idx not in seen:
                        seen.add(idx)
                        yield idx


class VerletPairCache2D:
    def __init__(self, radii: np.ndarray, box_lengths: np.ndarray, outer_ratio_cutoff: float, skin: float):
        self.radii = np.asarray(radii, dtype=float)
        self.box_lengths = np.asarray(box_lengths, dtype=float)
        self.outer_ratio_cutoff = float(outer_ratio_cutoff)
        self.skin = max(float(skin), 1.0e-12)
        self.reference_positions: np.ndarray | None = None
        self.pair_i = np.empty(0, dtype=np.int64)
        self.pair_j = np.empty(0, dtype=np.int64)
        self.rebuild_count = 0
        self.max_displacement_since_rebuild = 0.0

    def ensure_current(self, positions: np.ndarray):
        if self.reference_positions is None:
            self._rebuild(positions)
            return
        deltas = positions - self.reference_positions
        deltas -= self.box_lengths * np.rint(deltas / self.box_lengths)
        max_displacement = float(np.max(np.linalg.norm(deltas, axis=1))) if len(deltas) else 0.0
        self.max_displacement_since_rebuild = max_displacement
        if max_displacement > 0.5 * self.skin:
            self._rebuild(positions)

    def _rebuild(self, positions: np.ndarray):
        pair_i: list[int] = []
        pair_j: list[int] = []
        if len(self.radii) >= 2:
            max_radius = float(np.max(self.radii))
            cutoff = max(2.0 * self.outer_ratio_cutoff * max_radius + self.skin, 1.0e-12)
            cells = PeriodicCellList2D(self.box_lengths, cutoff)
            cells.build(positions)
            for i, position in enumerate(positions):
                search_radius = self.outer_ratio_cutoff * (self.radii[i] + max_radius) + self.skin
                for j in cells.nearby_indices(position, search_radius):
                    if j <= i:
                        continue
                    delta = minimum_image_delta(position, positions[j], self.box_lengths)
                    distance = float(np.linalg.norm(delta))
                    pair_cutoff = self.outer_ratio_cutoff * (self.radii[i] + self.radii[j]) + self.skin
                    if distance <= pair_cutoff:
                        pair_i.append(i)
                        pair_j.append(j)
        self.pair_i = np.asarray(pair_i, dtype=np.int64)
        self.pair_j = np.asarray(pair_j, dtype=np.int64)
        self.reference_positions = positions.copy()
        self.rebuild_count += 1
        self.max_displacement_since_rebuild = 0.0


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def minimum_image_delta(a: np.ndarray, b: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return delta - box_lengths * np.rint(delta / box_lengths)


def pair_gap(position_i, radius_i, position_j, radius_j, box_lengths: np.ndarray) -> float:
    delta = minimum_image_delta(position_i, position_j, box_lengths)
    return float(np.linalg.norm(delta) - radius_i - radius_j)


def area_fraction(radii: np.ndarray, box_lengths: np.ndarray) -> float:
    return float(np.sum(np.pi * radii**2) / float(np.prod(box_lengths)))


def resolve_particle_count(diameter_um: float, box_lengths: np.ndarray, target_area_fraction: float) -> int:
    radius = 0.5 * float(diameter_um)
    disk_area = math.pi * radius**2
    return max(1, int(round(float(target_area_fraction) * float(np.prod(box_lengths)) / disk_area)))


def candidate_min_gap(
    candidate: np.ndarray,
    radius: float,
    positions: np.ndarray,
    radii: np.ndarray,
    accepted_cells: PeriodicCellList2D,
    box_lengths: np.ndarray,
    max_radius: float,
) -> float:
    min_gap = float("inf")
    for j in accepted_cells.nearby_indices(candidate, radius + max_radius):
        gap = pair_gap(candidate, radius, positions[j], radii[j], box_lengths)
        if gap < min_gap:
            min_gap = gap
            if min_gap < 0.0:
                break
    return min_gap


def poisson_disk_initialize(
    radii: np.ndarray,
    box_lengths: np.ndarray,
    rng: np.random.Generator,
    *,
    max_attempts_per_disk: int,
    candidates_per_disk: int,
    progress_every: int,
) -> tuple[np.ndarray, dict]:
    order = np.argsort(-radii)
    sorted_radii = radii[order]
    positions_sorted = np.empty((len(sorted_radii), 2), dtype=float)
    max_radius = float(np.max(sorted_radii))
    cells = PeriodicCellList2D(box_lengths, cutoff=max(2.0 * max_radius, 1.0e-12))
    total_attempts = 0
    valid_candidates = 0
    accepted = 0

    for local_idx, radius in enumerate(sorted_radii):
        if accepted == 0:
            positions_sorted[local_idx] = rng.random(2) * box_lengths
            cells.add(local_idx, positions_sorted[local_idx])
            accepted += 1
            continue

        best_candidate = None
        best_gap = -np.inf
        attempts = 0
        batch_size = max(1, int(candidates_per_disk))
        while attempts < max_attempts_per_disk:
            current_batch = min(batch_size, max_attempts_per_disk - attempts)
            candidates = rng.random((current_batch, 2)) * box_lengths
            attempts += current_batch
            total_attempts += current_batch
            for candidate in candidates:
                gap = candidate_min_gap(candidate, radius, positions_sorted, sorted_radii, cells, box_lengths, max_radius)
                if gap >= 0.0:
                    valid_candidates += 1
                    if gap > best_gap:
                        best_gap = gap
                        best_candidate = candidate
            if best_candidate is not None:
                break

        if best_candidate is None:
            raise RuntimeError(
                f"Reduced-radius initialization failed after {max_attempts_per_disk} attempts "
                f"for disk {local_idx}."
            )
        positions_sorted[local_idx] = best_candidate
        cells.add(local_idx, best_candidate)
        accepted += 1
        if progress_every and accepted % progress_every == 0:
            print(f"initial placement: accepted {accepted}/{len(radii)}")

    positions = np.empty_like(positions_sorted)
    positions[order] = positions_sorted
    return positions, {
        "accepted": int(accepted),
        "total_attempts": int(total_attempts),
        "valid_candidate_count": int(valid_candidates),
        "candidates_per_disk": int(candidates_per_disk),
        "strategy": "2d_periodic_best_candidate_poisson_disk",
    }


def outer_shell_forces(
    positions: np.ndarray,
    radii: np.ndarray,
    box_lengths: np.ndarray,
    outer_ratio: float,
    rng: np.random.Generator,
    pair_cache: VerletPairCache2D,
) -> tuple[np.ndarray, dict]:
    forces = np.zeros_like(positions)
    pair_cache.ensure_current(positions)
    pair_i = pair_cache.pair_i
    pair_j = pair_cache.pair_j
    candidate_pairs = int(len(pair_i))
    if candidate_pairs == 0:
        return forces, {
            "max_shell_overlap_um": 0.0,
            "shell_overlap_count": 0,
            "shell_overlap_sum_um": 0.0,
            "max_overlap_um": 0.0,
            "overlap_count": 0,
            "overlap_sum_um": 0.0,
            "inner_diameter_ratio": float("inf"),
            "candidate_pair_count": 0,
        }

    deltas = positions[pair_i] - positions[pair_j]
    deltas -= box_lengths * np.rint(deltas / box_lengths)
    distances = np.linalg.norm(deltas, axis=1)
    contacts = radii[pair_i] + radii[pair_j]
    valid = contacts > 0.0
    inner_ratio = float(np.min(distances[valid] / contacts[valid])) if np.any(valid) else float("inf")

    shell_overlaps = outer_ratio * contacts - distances
    shell_active = shell_overlaps > 0.0
    shell_count = int(np.count_nonzero(shell_active))
    shell_sum = float(np.sum(shell_overlaps[shell_active])) if shell_count else 0.0
    max_shell = float(np.max(shell_overlaps[shell_active])) if shell_count else 0.0

    true_overlaps = contacts - distances
    overlap_active = true_overlaps > 0.0
    overlap_count = int(np.count_nonzero(overlap_active))
    overlap_sum = float(np.sum(true_overlaps[overlap_active])) if overlap_count else 0.0
    max_overlap = float(np.max(true_overlaps[overlap_active])) if overlap_count else 0.0

    if shell_count:
        active_i = pair_i[shell_active]
        active_j = pair_j[shell_active]
        active_deltas = deltas[shell_active]
        active_distances = distances[shell_active]
        directions = np.zeros_like(active_deltas)
        nonzero = active_distances > 1.0e-12
        directions[nonzero] = active_deltas[nonzero] / active_distances[nonzero, None]
        if np.any(~nonzero):
            random_dirs = rng.normal(size=(int(np.count_nonzero(~nonzero)), 2))
            norms = np.linalg.norm(random_dirs, axis=1)
            bad = norms <= 1.0e-12
            if np.any(bad):
                random_dirs[bad] = np.array([1.0, 0.0])
                norms[bad] = 1.0
            directions[~nonzero] = random_dirs / norms[:, None]
        pair_forces = shell_overlaps[shell_active, None] * directions
        np.add.at(forces, active_i, pair_forces)
        np.add.at(forces, active_j, -pair_forces)

    return forces, {
        "max_shell_overlap_um": float(max_shell),
        "shell_overlap_count": int(shell_count),
        "shell_overlap_sum_um": float(shell_sum),
        "max_overlap_um": float(max_overlap),
        "overlap_count": int(overlap_count),
        "overlap_sum_um": float(overlap_sum),
        "inner_diameter_ratio": float(inner_ratio),
        "candidate_pair_count": candidate_pairs,
    }


def updated_outer_ratio(
    *,
    outer_ratio: float,
    initial_outer_ratio: float,
    inner_ratio: float,
    active_area_fraction: float,
    contraction_rate: float,
) -> float:
    if not np.isfinite(inner_ratio):
        return 1.0
    inner_void = 1.0 - active_area_fraction * inner_ratio**2
    nominal_void = 1.0 - active_area_fraction * NOMINAL_DENSITY_RATIO
    difference = inner_void - nominal_void
    if difference <= 0.0:
        return float(outer_ratio)
    j = float(np.ceil(-np.log10(max(difference, 1.0e-300))))
    decrement = (0.5**j) * float(initial_outer_ratio) * float(contraction_rate)
    return max(1.0, float(outer_ratio) - decrement)


def relax_stage(
    positions: np.ndarray,
    radii: np.ndarray,
    box_lengths: np.ndarray,
    rng: np.random.Generator,
    *,
    max_iterations: int,
    initial_outer_ratio: float,
    contraction_rate: float,
    force_scaling_factor: float,
    max_displacement_fraction: float,
    tolerance_overlap: float,
    verlet_skin: float,
) -> tuple[np.ndarray, dict]:
    outer_ratio = float(initial_outer_ratio)
    active_area_fraction = area_fraction(radii, box_lengths)
    diameters = np.maximum(2.0 * radii, 1.0e-12)
    max_displacement = np.maximum(float(max_displacement_fraction) * radii, 1.0e-12)
    pair_cache = VerletPairCache2D(radii, box_lengths, outer_ratio, verlet_skin)
    forces, last_diag = outer_shell_forces(positions, radii, box_lengths, outer_ratio, rng, pair_cache)
    stop_reason = "max_iterations"
    iterations = 0

    for _ in range(max_iterations):
        inner_ratio = float(last_diag["inner_diameter_ratio"])
        if last_diag["shell_overlap_count"] == 0:
            stop_reason = "no_outer_shell_contacts"
            break
        if outer_ratio <= inner_ratio and last_diag["max_overlap_um"] <= tolerance_overlap:
            stop_reason = "outer_ratio_reached_inner_ratio"
            break
        if outer_ratio <= 1.0 + 1.0e-12 and last_diag["max_overlap_um"] <= tolerance_overlap:
            stop_reason = "true_non_overlap_reached"
            break

        displacement = (
            float(force_scaling_factor)
            * outer_ratio**2
            * forces
            / (2.0 * diameters[:, None])
        )
        norms = np.linalg.norm(displacement, axis=1)
        too_large = norms > max_displacement
        if np.any(too_large):
            displacement[too_large] *= (max_displacement[too_large] / norms[too_large])[:, None]

        positions = np.mod(positions + displacement, box_lengths)
        iterations += 1
        forces, last_diag = outer_shell_forces(positions, radii, box_lengths, outer_ratio, rng, pair_cache)
        outer_ratio = updated_outer_ratio(
            outer_ratio=outer_ratio,
            initial_outer_ratio=initial_outer_ratio,
            inner_ratio=float(last_diag["inner_diameter_ratio"]),
            active_area_fraction=active_area_fraction,
            contraction_rate=contraction_rate,
        )

    return positions, {
        "iterations_used": int(iterations),
        "max_iterations": int(max_iterations),
        "outer_diameter_ratio_final": float(outer_ratio),
        "area_fraction_active": float(active_area_fraction),
        "stop_reason": stop_reason,
        "verlet_skin_um": float(verlet_skin),
        "pair_cache_rebuilds": int(pair_cache.rebuild_count),
        **last_diag,
    }


def generate_force_biased_2d(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    start = time.perf_counter()
    box_lengths = np.asarray(args.box, dtype=float)
    radius = 0.5 * float(args.diameter_um)
    n_disks = int(args.num_disks) if args.num_disks else resolve_particle_count(args.diameter_um, box_lengths, args.area_fraction)
    radii = np.full(n_disks, radius, dtype=float)
    rng = np.random.default_rng(args.seed)

    initial_fraction = float(args.initial_radius_fraction)
    start_radii = radii * initial_fraction
    positions, init_diag = poisson_disk_initialize(
        start_radii,
        box_lengths,
        rng,
        max_attempts_per_disk=int(args.max_attempts_per_disk),
        candidates_per_disk=int(args.candidates_per_disk),
        progress_every=int(args.progress_every),
    )

    verlet_skin = float(args.verlet_skin)
    if verlet_skin <= 0.0:
        verlet_skin = float(args.verlet_skin_fraction) * float(args.diameter_um)
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
        "name": "force_biased_2d_periodic",
        "status": "success" if final_diag["max_overlap_um"] <= float(args.tolerance_overlap) else "warning",
        "elapsed_time_s": float(time.perf_counter() - start),
        "note": (
            "2D force-biased disk packing: reduced-radius Poisson-disk initialization, "
            "staged radius growth, and periodic outer-shell repulsive relaxation."
        ),
        "initialization": init_diag,
        "stages": int(args.stages),
        "initial_radius_fraction": float(initial_fraction),
        "max_iterations_per_stage": int(args.max_iterations_per_stage),
        "final_cleanup_steps": int(args.final_cleanup_steps),
        "outer_diameter_ratio_initial": float(args.outer_diameter_ratio),
        "nominal_density_ratio": float(NOMINAL_DENSITY_RATIO),
        "contraction_rate": float(args.contraction_rate),
        "force_scaling_factor": float(args.force_scaling_factor),
        "max_displacement_fraction": float(args.max_displacement_fraction),
        "verlet_skin_um": float(verlet_skin),
        "final_cleanup": final_diag,
        "stage_diagnostics": stage_diagnostics,
    }
    return np.mod(positions, box_lengths), radii, diagnostics


def find_overlap_pairs(
    positions: np.ndarray,
    radii: np.ndarray,
    box_lengths: np.ndarray,
    tolerance: float,
) -> tuple[list[tuple[int, int, float]], float]:
    if len(radii) < 2:
        return [], float("inf")
    max_radius = float(np.max(radii))
    cells = PeriodicCellList2D(box_lengths, cutoff=max(2.0 * max_radius, 1.0e-12))
    cells.build(positions)
    pairs = []
    min_gap = float("inf")
    for i, position in enumerate(positions):
        for j in cells.nearby_indices(position, radii[i] + max_radius):
            if j <= i:
                continue
            gap = pair_gap(position, radii[i], positions[j], radii[j], box_lengths)
            min_gap = min(min_gap, gap)
            if gap < -tolerance:
                pairs.append((int(i), int(j), float(gap)))
    return pairs, min_gap


def validate_packing(
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
            "integer disk count makes exact equality impossible."
        )
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
            "radius_um": float(radii[0]),
            "diameter_um": float(2.0 * radii[0]),
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
        h5.attrs["coordinate_unit"] = "um"
        h5.attrs["box_lengths_um"] = np.asarray(box_lengths, dtype=float)
        h5.attrs["boundary"] = "periodic_2d"
    return True


def write_plot(path: Path, positions: np.ndarray, radii: np.ndarray, box_lengths: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle, Rectangle
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    patches = [Circle((float(x), float(y)), float(r)) for (x, y), r in zip(positions, radii)]
    collection = PatchCollection(patches, facecolor="#6baed6", edgecolor="#1f4e79", linewidth=0.12, alpha=0.75)
    ax.add_collection(collection)
    ax.add_patch(Rectangle((0.0, 0.0), float(box_lengths[0]), float(box_lengths[1]), fill=False, edgecolor="black", linewidth=1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, float(box_lengths[0]))
    ax.set_ylim(0.0, float(box_lengths[1]))
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("2D monodisperse disk packing")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_outputs(
    output_dir: Path,
    positions: np.ndarray,
    radii: np.ndarray,
    diagnostics: dict,
    validation: ValidationReport,
    args: argparse.Namespace,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    box_lengths = np.asarray(args.box, dtype=float)
    diameter_nm = float(args.diameter_um) * 1000.0
    diameter_label = f"{diameter_nm:g} nm"
    write_particles_csv(output_dir / "particles_2d_real_units.csv", positions, radii)
    hdf5_written = write_hdf5(output_dir / "particles_2d.h5", positions, radii, box_lengths)
    plot_written = write_plot(output_dir / "packing_2d.png", positions, radii, box_lengths) if args.make_plot else False
    metadata = {
        "project": {
            "name": output_dir.name,
            "description": (
                f"2D monodisperse disk packing, {diameter_label} disk diameter, "
                f"{box_lengths[0]:g} x {box_lengths[1]:g} um box, "
                f"area fraction {float(args.area_fraction):g}."
            ),
        },
        "physical": {
            "diameter_um": float(args.diameter_um),
            "box_lengths_um": [float(v) for v in box_lengths],
            "target_area_fraction": float(args.area_fraction),
            "area_fraction_interpretation": "2D analogue of volume fraction",
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
    parser = argparse.ArgumentParser(description="Generate a 2D monodisperse disk packing.")
    parser.add_argument("--diameter-um", type=float, default=0.4)
    parser.add_argument("--box", type=float, nargs=2, default=[30.0, 30.0], metavar=("LX", "LY"))
    parser.add_argument("--area-fraction", type=float, default=0.6)
    parser.add_argument("--num-disks", type=int, default=0, help="0 means compute from target area fraction.")
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
    if args.diameter_um <= 0.0:
        raise ValueError("--diameter-um must be positive.")
    if args.area_fraction <= 0.0:
        raise ValueError("--area-fraction must be positive.")

    n_disks = int(args.num_disks) if args.num_disks else resolve_particle_count(args.diameter_um, box_lengths, args.area_fraction)
    actual_phi = n_disks * math.pi * (0.5 * args.diameter_um) ** 2 / float(np.prod(box_lengths))
    print(f"resolved disks: {n_disks}")
    print(f"box: {box_lengths.tolist()} um")
    print(f"diameter: {args.diameter_um:g} um")
    print(f"target area fraction: {args.area_fraction:g}")
    print(f"integer-count area fraction: {actual_phi:.12g}")
    if args.dry_run:
        return 0

    positions, radii, diagnostics = generate_force_biased_2d(args)
    validation = validate_packing(
        positions,
        radii,
        box_lengths,
        float(args.area_fraction),
        float(args.tolerance_overlap),
        float(args.tolerance_area_fraction),
    )
    write_outputs(args.output, positions, radii, diagnostics, validation, args)
    print(f"output: {args.output}")
    print(f"validation_passed: {validation.passed}")
    print(f"actual_area_fraction: {validation.metrics['actual_area_fraction']:.12g}")
    print(f"max_overlap_um: {validation.metrics['max_overlap_um']:.3e}")
    print(f"min_gap_um: {validation.metrics['min_gap_um']:.3e}")
    return 0 if validation.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
