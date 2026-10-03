"""Project enums and normalization helpers."""

from __future__ import annotations

from enum import Enum
from math import pi, sqrt


class NormalizedEnum(str, Enum):
    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = getattr(cls, "_ALIASES", {})
        key = aliases.get(key, key)
        for member in cls:
            if member.value == key or member.name.lower() == key:
                return member
        allowed = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown {cls.__name__}: {value!r}. Allowed values: {allowed}")


class SizeDistributionClass(NormalizedEnum):
    MONODISPERSE = "monodisperse"
    QUASI_MONODISPERSE = "quasi_monodisperse"
    CONTINUOUS_POLYDISPERSE = "continuous_polydisperse"
    DISCRETE_MIXTURE = "discrete_mixture"
    IMPORTED_DISTRIBUTION = "imported_distribution"


SizeDistributionClass._ALIASES = {
    "mono": "monodisperse",
    "single": "monodisperse",
    "quasi": "quasi_monodisperse",
    "polydisperse": "continuous_polydisperse",
    "continuous": "continuous_polydisperse",
    "lognormal": "continuous_polydisperse",
    "normal": "continuous_polydisperse",
    "mixture": "discrete_mixture",
    "discrete": "discrete_mixture",
    "imported": "imported_distribution",
}


class SpatialOrderClass(NormalizedEnum):
    OVERLAPPING_RANDOM = "overlapping_random"
    HARD_CORE_RANDOM = "hard_core_random"
    HYPERUNIFORM = "hyperuniform"
    QUASICRYSTAL = "quasicrystal"
    PERIODIC_CRYSTAL = "periodic_crystal"
    CUSTOM_TARGET_CORRELATION = "custom_target_correlation"


SpatialOrderClass._ALIASES = {
    "poisson": "overlapping_random",
    "poisson_random": "overlapping_random",
    "uncorrelated_random": "overlapping_random",
    "uncorrelated": "overlapping_random",
    "overlap": "overlapping_random",
    "overlapping": "overlapping_random",
    "boolean": "overlapping_random",
    "boolean_model": "overlapping_random",
    "random": "overlapping_random",
    "hardcore": "hard_core_random",
    "hard_core": "hard_core_random",
    "hard_sphere": "hard_core_random",
    "hard_sphere_random": "hard_core_random",
    "crystal": "periodic_crystal",
    "fcc": "periodic_crystal",
    "bcc": "periodic_crystal",
    "sc": "periodic_crystal",
    "diamond": "periodic_crystal",
    "diamond_cubic": "periodic_crystal",
    "custom": "custom_target_correlation",
}


class DensityClass(NormalizedEnum):
    DILUTE = "dilute"
    SPARSE = "sparse"
    MODERATE = "moderate"
    DENSE = "dense"
    VERY_DENSE = "very_dense"
    CRYSTAL_DENSE = "crystal_dense"
    OVERLAPPING_NOMINAL = "overlapping_nominal"


DensityClass._ALIASES = {
    "close_packed": "crystal_dense",
    "crystal": "crystal_dense",
    "overlapping": "overlapping_nominal",
    "nominal": "overlapping_nominal",
}


def infer_density_class(packing_fraction: float) -> DensityClass:
    """Infer the engineering density class from target packing fraction."""
    phi = float(packing_fraction)
    if phi <= 0:
        raise ValueError("packing_fraction must be positive")
    if phi < 0.05:
        return DensityClass.DILUTE
    if phi < 0.20:
        return DensityClass.SPARSE
    if phi < 0.45:
        return DensityClass.MODERATE
    if phi < 0.58:
        return DensityClass.DENSE
    if phi <= 0.66:
        return DensityClass.VERY_DENSE
    return DensityClass.CRYSTAL_DENSE


class LatticeType(NormalizedEnum):
    SC = "SC"
    BCC = "BCC"
    FCC = "FCC"
    HCP = "HCP"
    DIAMOND_CUBIC = "DIAMOND_CUBIC"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "DIAMOND": "DIAMOND_CUBIC",
            "DC": "DIAMOND_CUBIC",
        }
        key = aliases.get(key, key)
        for member in cls:
            if member.value == key or member.name == key:
                return member
        allowed = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown lattice type {value!r}. Allowed values: {allowed}")


LATTICE_BASIS_COUNTS = {
    LatticeType.SC: 1,
    LatticeType.BCC: 2,
    LatticeType.FCC: 4,
    LatticeType.HCP: 4,
    LatticeType.DIAMOND_CUBIC: 8,
}

LATTICE_MAX_PACKING_FRACTIONS = {
    LatticeType.SC: pi / 6.0,
    LatticeType.BCC: sqrt(3.0) * pi / 8.0,
    LatticeType.FCC: pi / (3.0 * sqrt(2.0)),
    LatticeType.HCP: pi / (3.0 * sqrt(2.0)),
    LatticeType.DIAMOND_CUBIC: sqrt(3.0) * pi / 16.0,
}


def lattice_basis_count(value) -> int:
    return LATTICE_BASIS_COUNTS[LatticeType.from_value(value)]


def lattice_max_packing_fraction(value) -> float:
    return LATTICE_MAX_PACKING_FRACTIONS[LatticeType.from_value(value)]


