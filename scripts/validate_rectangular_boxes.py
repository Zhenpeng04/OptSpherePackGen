"""Bounded, repeatable stability and scale checks for periodic rectangular boxes.

Each case runs in a separate process with a timeout. Reports distinguish
generation failure, quality-gate failure and timeout. This is a local benchmark,
not a guarantee of arbitrary near-jamming convergence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def run_case(case, output_root, source_root=None):
    sys.path.insert(0, str(source_root or ROOT / "src"))
    import numpy as np
    from spherepackgen.api import load_config, run_generation
    from spherepackgen.domain.enums import SizeDistributionClass, SpatialOrderClass

    config = load_config(ROOT / "configs/examples/hard_core_random.yaml")
    config.structure.size_distribution = SizeDistributionClass.MONODISPERSE
    config.size_distribution.type = "monodisperse"
    config.physical.mean_diameter_m = 1e-6
    config.physical.packing_fraction = case["phi"]
    config.physical.medium_thickness_m = None
    config.particles.num_particles = case["n"]
    lateral = (case["n"] * np.pi / (6 * case["phi"] * case["aspect"]))**(1/3)
    if source_root is None:
        config.domain.type = "periodic_box"
        config.domain.length_m = lateral * 1e-6
        config.domain.depth_m = lateral * case["aspect"] * 1e-6
    config.algorithm.name = case["algorithm"]
    config.algorithm.parameters = {}
    if case["algorithm"] == "marked_poisson_boolean":
        config.structure.spatial_order = SpatialOrderClass.OVERLAPPING_RANDOM
    config.runtime.random_seed = case["seed"]
    config.output.path = output_root / case["name"]
    config.output.make_plots = case.get("plots", False)
    config.output.formats = ["json"]
    config.analysis.compute_g2 = case.get("analysis", False)
    config.analysis.compute_Sk = case.get("analysis", False)
    start = perf_counter()
    try:
        bundle = run_generation(config)
        return {**case, "passed": bundle.validation.passed, "elapsed_s": perf_counter() - start,
                "generation_s": bundle.result.elapsed_time_s,
                "timings_s": getattr(bundle, "stage_timings_s", {}),
                "max_overlap": bundle.validation.metrics["max_overlap"],
                "errors": bundle.validation.errors, "generator_status": bundle.result.status}
    except Exception as exc:
        return {**case, "passed": False, "elapsed_s": perf_counter() - start,
                "error": str(exc), "error_type": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["stability", "scale", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/rectangular_validation")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--scale-aspect", type=float, default=0.25)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.source_root is not None and not (args.source_root / "spherepackgen/api.py").is_file():
        parser.error("--source-root must contain the spherepackgen package")
    if args.case:
        print(json.dumps(run_case(json.loads(args.case), args.output_root, args.source_root)))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = []
    if args.suite in {"stability", "all"}:
        for phi in [0.50, 0.60, 0.62]:
            for aspect in [0.25, 1.0, 4.0]:
                for seed in [123, 456, 789]:
                    cases.append(dict(name=f"dense_phi{phi}_aspect{aspect}_seed{seed}",
                        algorithm="force_biased", n=500, phi=phi, aspect=aspect, seed=seed))
    if args.suite in {"scale", "all"}:
        for algorithm, phi in [("marked_poisson_boolean", 1.2), ("poisson_disk", 0.12),
                               ("rsa", 0.25), ("force_biased", 0.5)]:
            for n in [1000, 10000]:
                cases.append(dict(name=f"scale_{algorithm}_n{n}", algorithm=algorithm,
                                  n=n, phi=phi, aspect=args.scale_aspect, seed=123))
    records = []
    for case in cases:
        command = [sys.executable, str(Path(__file__).resolve()), "--case", json.dumps(case),
                   "--output-root", str(args.output_root)]
        if args.source_root:
            command.extend(["--source-root", str(args.source_root)])
        start = perf_counter()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            if result.returncode:
                record = {**case, "passed": False, "error": result.stderr[-2000:]}
            else:
                record = json.loads(result.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            record = {**case, "passed": False, "error_type": "Timeout", "elapsed_s": perf_counter() - start}
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.output_root / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Passed {sum(r['passed'] for r in records)}/{len(records)} cases", flush=True)
    if any(not r["passed"] for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
