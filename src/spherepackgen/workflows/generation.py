"""End-to-end generation workflow."""

from __future__ import annotations

import numpy as np
from time import perf_counter

from spherepackgen.analysis.descriptors import analyze_particle_set
from spherepackgen.config.schema import PackingConfig
from spherepackgen.domain.domain import PeriodicBox
from spherepackgen.domain.enums import SpatialOrderClass
from spherepackgen.domain.result import ResultBundle
from spherepackgen.exporters.files import export_bundle
from spherepackgen.generators.base import GeneratorContext
from spherepackgen.generators.registry import GENERATORS, default_generator_name, get_generator
from spherepackgen.utils.exceptions import GenerationError
from spherepackgen.validation.quality_gate import validate_particle_set
from spherepackgen.workflows.resolve import resolve_parameters
from spherepackgen.provenance import prepare_run, environment_provenance


def run_generation(config: PackingConfig, progress=None) -> ResultBundle:
    def report(stage):
        if progress is not None:
            progress(stage)

    report("resolution")
    start = perf_counter()
    config, provenance = prepare_run(config)
    resolved = resolve_parameters(config)
    timings = {"resolution": perf_counter() - start}
    rng = np.random.default_rng(config.runtime.random_seed)
    domain = PeriodicBox(resolved.box_lengths)
    generator = get_generator(config)
    context = GeneratorContext(config=config, resolved=resolved, domain=domain, rng=rng)
    report("generation")
    start = perf_counter()
    try:
        result = generator.generate(context)
    except GenerationError as exc:
        requested_name = (config.algorithm.name or "default").strip().lower()
        selected_name = default_generator_name(config) if requested_name == "default" else requested_name
        can_fallback = (
            requested_name == "default"
            and config.structure.spatial_order == SpatialOrderClass.HARD_CORE_RANDOM
            and selected_name in {"poisson_disk", "rsa"}
        )
        if not can_fallback:
            raise
        result = GENERATORS["force_biased"].generate(context)
        result.diagnostics["fallback_from"] = selected_name
        result.diagnostics["fallback_reason"] = str(exc)
    timings["generation"] = perf_counter() - start
    report("validation")
    start = perf_counter()
    validation = validate_particle_set(result.particle_set, config, resolved)
    if not validation.metrics.get("input_valid", True):
        raise GenerationError("Invalid generated geometry: " + "; ".join(validation.errors))
    timings["validation"] = perf_counter() - start
    report("analysis")
    start = perf_counter()
    descriptors = analyze_particle_set(result.particle_set, config.analysis)
    timings["analysis"] = perf_counter() - start
    bundle = ResultBundle(
        config=config,
        resolved=resolved,
        result=result,
        validation=validation,
        descriptors=descriptors,
        exported_files=[],
        stage_timings_s=timings,
        provenance=provenance,
    )
    bundle.provenance.update(environment_provenance())
    report("export")
    bundle.exported_files = export_bundle(bundle)
    report("complete")
    return bundle
