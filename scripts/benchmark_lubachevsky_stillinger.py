"""Benchmark Lubachevsky-Stillinger event backends and tuning parameters.

This script is intentionally lightweight: it disables analysis and plotting,
uses the normal project workflow, and reports enough diagnostics to compare the
exhaustive and Verlet-heap event schedulers under repeatable settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spherepackgen.api import load_config, run_generation  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LS event backends.")
    parser.add_argument("--config", default="configs/examples/lubachevsky_stillinger.yaml")
    parser.add_argument("--output-root", default="results/ls_backend_benchmark")
    parser.add_argument("--n", nargs="+", type=int, default=[12, 16, 24, 32, 64])
    parser.add_argument("--phi", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--compression-rate", type=float, default=2.0e-2)
    parser.add_argument("--max-events", type=int, default=12000)
    parser.add_argument("--final-cleanup-steps", type=int, default=25)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["auto", "exhaustive", "verlet_heap"],
        choices=["auto", "exhaustive", "verlet_heap"],
    )
    parser.add_argument("--neighbor-rebuild-fraction", type=float, default=None)
    parser.add_argument("--write-json", default=None)
    return parser


def _run_case(args: argparse.Namespace, n_particles: int, backend: str) -> dict:
    config = load_config(args.config)
    config.output.path = Path(args.output_root) / f"n{n_particles}_{backend}"
    config.output.make_plots = False
    config.output.formats = ["json"]
    config.analysis.compute_g2 = False
    config.analysis.compute_Sk = False
    config.particles.num_particles = int(n_particles)
    config.physical.packing_fraction = float(args.phi)
    config.runtime.random_seed = int(args.seed)
    config.algorithm.parameters["event_backend"] = backend
    config.algorithm.parameters["compression_rate"] = float(args.compression_rate)
    config.algorithm.parameters["max_events"] = int(args.max_events)
    config.algorithm.parameters["final_cleanup_steps"] = int(args.final_cleanup_steps)
    if args.neighbor_rebuild_fraction is not None:
        config.algorithm.parameters["neighbor_rebuild_fraction"] = float(args.neighbor_rebuild_fraction)

    start = time.perf_counter()
    row = {
        "n_particles": int(n_particles),
        "backend_requested": backend,
        "phi": float(args.phi),
        "seed": int(args.seed),
        "ok": False,
    }
    try:
        bundle = run_generation(config)
    except Exception as exc:  # pragma: no cover - used for benchmark reporting.
        row.update(
            {
                "elapsed_s": round(time.perf_counter() - start, 6),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return row

    diag = bundle.result.diagnostics
    row.update(
        {
            "ok": True,
            "validation_passed": bool(bundle.validation.passed),
            "status": bundle.result.status,
            "elapsed_s": round(time.perf_counter() - start, 6),
            "generator_elapsed_s": round(float(bundle.result.elapsed_time_s), 6),
            "event_backend_selected": diag.get("event_backend_selected"),
            "event_search_backend": diag.get("event_search_backend"),
            "events": diag.get("events"),
            "neighbor_rebuilds": diag.get("neighbor_rebuilds"),
            "neighbor_rebuild_fraction": diag.get("neighbor_rebuild_fraction"),
            "scheduled_event_count": diag.get("scheduled_event_count"),
            "event_invalidations": diag.get("event_invalidations"),
            "max_candidate_pair_count": diag.get("max_candidate_pair_count"),
            "max_heap_size": diag.get("max_heap_size"),
            "pre_cleanup_overlap_pair_count": diag.get("pre_cleanup_overlap_pair_count"),
            "final_overlap_pair_count": diag.get("final_overlap_pair_count"),
            "max_overlap": bundle.validation.metrics.get("max_overlap"),
        }
    )
    return row


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = [_run_case(args, n_particles, backend) for n_particles in args.n for backend in args.backends]
    text = json.dumps(rows, indent=2)
    print(text)
    if args.write_json:
        Path(args.write_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_json).write_text(text + "\n", encoding="utf-8")
    return 0 if all(row.get("ok") and row.get("validation_passed", False) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
