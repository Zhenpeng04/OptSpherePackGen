"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from importlib.resources import files

from spherepackgen.api import load_config, run_generation
from spherepackgen.utils.exceptions import GenerationError


def _json_ready(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def cmd_run(args) -> int:
    config = load_config(args.config)
    bundle = run_generation(config)
    config = bundle.config
    summary = {
        "case": config.project.name,
        "generator": bundle.result.generator_name,
        "status": bundle.result.status,
        "validation_passed": bundle.validation.passed,
        **bundle.readiness,
        "success_policy": "allow_candidate" if args.allow_candidate else "strict",
        "random_seed": config.runtime.random_seed,
        "inferred_density_class": config.structure.density_class.value,
        "packing_fraction_actual": bundle.validation.metrics["packing_fraction_actual"],
        "target_covered_volume_fraction": bundle.validation.metrics.get("target_covered_volume_fraction"),
        "nominal_volume_fraction": bundle.validation.metrics.get("nominal_volume_fraction"),
        "expected_boolean_covered_fraction": bundle.validation.metrics.get("expected_boolean_covered_fraction"),
        "n_particles": bundle.resolved.n_particles,
        "box_length_dimensionless": bundle.resolved.box_length,
        "box_length_real": bundle.resolved.box_length_m / bundle.resolved.output_unit_m,
        "box_lengths_dimensionless": bundle.resolved.box_lengths.tolist(),
        "box_lengths_real": (bundle.resolved.box_lengths_m / bundle.resolved.output_unit_m).tolist(),
        "real_length_unit": bundle.resolved.output_unit,
        "output_path": str(config.output.path),
        "exported_files": bundle.exported_files,
        "errors": bundle.validation.errors,
        "warnings": bundle.validation.warnings,
    }
    print(json.dumps(_json_ready(summary), indent=2))
    if not bundle.validation.passed:
        return 2
    return 0 if bundle.readiness["export_ready"] or args.allow_candidate else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spherepackgen", description="Microsphere geometry for optical simulation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a generation config")
    run.add_argument("config", help="YAML configuration file")
    policy = run.add_mutually_exclusive_group()
    policy.add_argument("--strict", dest="allow_candidate", action="store_false",
                        help="Require valid geometry and a successful structure target (default)")
    policy.add_argument("--allow-candidate", action="store_true",
                        help="Return zero for geometrically valid candidates, including unconverged structures")
    run.set_defaults(allow_candidate=False)
    run.set_defaults(func=cmd_run)
    gui = sub.add_parser("gui", help="Start the local graphical interface")
    gui.add_argument("--workdir", default=".", help="Directory for outputs, upload cache and history (default: current directory)")
    gui.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=8501)
    gui.add_argument("--headless", action="store_true", help="Start without opening a browser")
    gui.set_defaults(func=cmd_gui)
    examples = sub.add_parser("examples", help="Copy packaged YAML examples to a user directory")
    examples.add_argument("--output", default="examples", help="Destination directory (default: examples)")
    examples.set_defaults(func=cmd_examples)
    return parser


def cmd_gui(args) -> int:
    from spherepackgen.gui.launcher import launch_gui
    return launch_gui(args.workdir, args.port, args.headless)


def cmd_examples(args) -> int:
    destination = Path(args.output).expanduser().resolve()
    templates = sorted((p for p in files("spherepackgen.examples").iterdir() if p.name.endswith(".yaml")), key=lambda p: p.name)
    # Preflight all paths so an existing file is never silently replaced.
    for template in templates:
        if (destination / template.name).exists():
            raise FileExistsError(f"Example already exists: {destination / template.name}. Choose a new output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    for template in templates:
        with (destination / template.name).open("xb") as stream:
            stream.write(template.read_bytes())
    print(json.dumps({"output": str(destination), "files": [p.name for p in templates]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, GenerationError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
