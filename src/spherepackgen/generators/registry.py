"""Generator registry and default selection."""

from __future__ import annotations

from spherepackgen.config.schema import PackingConfig
from spherepackgen.domain.enums import SpatialOrderClass
from spherepackgen.generators.crystal import CrystalGenerator
from spherepackgen.generators.force_biased import ForceBiasedGenerator
from spherepackgen.generators.lubachevsky_stillinger import LubachevskyStillingerGenerator
from spherepackgen.generators.metropolis import MetropolisGenerator
from spherepackgen.generators.poisson import MarkedPoissonBooleanGenerator
from spherepackgen.generators.rsa import RSAGenerator, VariableRadiusPoissonDiskGenerator


GENERATORS = {
    "crystal": CrystalGenerator(),
    "periodic_crystal": CrystalGenerator(),
    "marked_poisson_boolean": MarkedPoissonBooleanGenerator(),
    "boolean_overlapping_spheres": MarkedPoissonBooleanGenerator(),
    "overlapping_random": MarkedPoissonBooleanGenerator(),
    "poisson": MarkedPoissonBooleanGenerator(),
    "poisson_random": MarkedPoissonBooleanGenerator(),
    "uncorrelated_random": MarkedPoissonBooleanGenerator(),
    "poisson_disk": VariableRadiusPoissonDiskGenerator(),
    "variable_radius_poisson_disk": VariableRadiusPoissonDiskGenerator(),
    "rsa": RSAGenerator(),
    "hard_core_random": RSAGenerator(),
    "metropolis": MetropolisGenerator(),
    "force_biased": ForceBiasedGenerator(),
    "force_biased_algorithm": ForceBiasedGenerator(),
    "fba": ForceBiasedGenerator(),
    "jodrey_tory": ForceBiasedGenerator(),
    "bezrukov_jodrey_tory": ForceBiasedGenerator(),
    "lubachevsky_stillinger": LubachevskyStillingerGenerator(),
    "lubachevsky-stillinger": LubachevskyStillingerGenerator(),
    "lubachevsky_stillinger_algorithm": LubachevskyStillingerGenerator(),
    "ls": LubachevskyStillingerGenerator(),
}


def default_generator_name(config: PackingConfig) -> str:
    spatial = config.structure.spatial_order
    phi = config.physical.packing_fraction
    if spatial == SpatialOrderClass.PERIODIC_CRYSTAL:
        return "crystal"
    if spatial == SpatialOrderClass.OVERLAPPING_RANDOM:
        return "marked_poisson_boolean"
    if spatial == SpatialOrderClass.HARD_CORE_RANDOM:
        if phi <= 0.18:
            return "poisson_disk"
        if phi > 0.34:
            return "force_biased"
        return "rsa"
    raise NotImplementedError(f"No default generator for spatial_order={spatial.value}")


def get_generator(config: PackingConfig):
    name = (config.algorithm.name or "default").strip().lower()
    if name == "default":
        name = default_generator_name(config)
    if name not in GENERATORS:
        choices = "crystal, marked_poisson_boolean, poisson_disk, rsa, force_biased, lubachevsky_stillinger, metropolis"
        raise NotImplementedError(f"Unknown generator {name!r}. Choose one of: {choices}.")
    return GENERATORS[name]
