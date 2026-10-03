"""Generator strategy base classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from spherepackgen.config.schema import PackingConfig
from spherepackgen.domain.domain import PeriodicBox
from spherepackgen.domain.result import PackingResult, ResolvedParameters


@dataclass
class GeneratorContext:
    config: PackingConfig
    resolved: ResolvedParameters
    domain: PeriodicBox
    rng: np.random.Generator


class GeneratorStrategy(Protocol):
    name: str

    def generate(self, context: GeneratorContext) -> PackingResult:
        ...
