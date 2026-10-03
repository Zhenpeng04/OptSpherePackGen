"""Radius sampling for P0."""

from __future__ import annotations

from math import gamma
from pathlib import Path
from typing import Any

import numpy as np

from spherepackgen.config.schema import PackingConfig
from spherepackgen.domain.enums import SizeDistributionClass


def _truncated_normal(rng, mean, std, low, high, size, batch_min=64):
    if not np.all(np.isfinite([mean, std, low, high])) or std < 0 or low >= high:
        raise ValueError("Truncated normal requires finite bounds with min < max and a nonnegative CV.")
    if std == 0:
        if not low <= mean <= high:
            raise ValueError("For CV=0, the radius range must contain the mean radius (factor 1).")
        return np.full(size, mean, dtype=float)
    out = np.empty(size, dtype=float)
    filled = 0
    for _ in range(100):
        if filled == size:
            return out
        batch = rng.normal(mean, std, size=max(size - filled, batch_min))
        batch = batch[(batch >= low) & (batch <= high)]
        n = min(len(batch), size - filled)
        if n:
            out[filled : filled + n] = batch[:n]
            filled += n
    # Bounded fallback also handles valid but very unlikely intervals.
    from scipy.stats import truncnorm
    out[filled:] = truncnorm.rvs((low - mean) / std, (high - mean) / std,
                                loc=mean, scale=std, size=size - filled, random_state=rng)
    return out


def _normalized_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalized_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {_normalized_key(key): value for key, value in parameters.items()}


def _parameter(parameters: dict[str, Any], names: tuple[str, ...], default=None):
    for name in names:
        key = _normalized_key(name)
        if key in parameters and parameters[key] is not None:
            return parameters[key]
    return default


def _radius_bounds(size_cfg, parameters: dict[str, Any], default_low=None, default_high=None) -> tuple[float | None, float | None]:
    min_factor = _parameter(parameters, ("min_radius_factor", "min_factor", "radius_min_factor"), size_cfg.min_radius_factor)
    max_factor = _parameter(parameters, ("max_radius_factor", "max_factor", "radius_max_factor"), size_cfg.max_radius_factor)
    low = default_low if min_factor is None else 0.5 * float(min_factor)
    high = default_high if max_factor is None else 0.5 * float(max_factor)
    _validate_bounds(low, high)
    return low, high


def _validate_bounds(low, high):
    if any(x is not None and (not np.isfinite(x) or x <= 0) for x in (low, high)):
        raise ValueError("Radius bounds must be positive finite values.")
    if low is not None and high is not None and low >= high:
        raise ValueError("Min radius factor must be smaller than max radius factor.")


def _clip_and_rescale(radii: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if radii.size == 0:
        raise ValueError("size distribution produced no radius samples")
    if low is not None:
        radii = np.maximum(radii, low)
    if high is not None:
        radii = np.minimum(radii, high)
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("all sampled particle sizes must be positive finite values")
    radii *= 0.5 / np.mean(radii)
    return radii


def _split_size_file_line(line: str) -> list[str]:
    line = line.strip()
    if "," in line:
        return [part.strip() for part in line.split(",")]
    return line.split()


def _parse_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def _column_index(tokens: list[str], column, header_tokens: list[str] | None) -> int | None:
    if column is None:
        return None
    if isinstance(column, int):
        return column
    text = str(column).strip()
    if text.isdigit():
        return int(text)
    if header_tokens:
        normalized = [_normalized_key(token) for token in header_tokens]
        key = _normalized_key(text)
        if key in normalized:
            return normalized.index(key)
    return None


def _load_size_file_values(file_path: str | Path, column=None) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"size_distribution.size_file does not exist: {path}")

    values: list[float] = []
    header_tokens: list[str] | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = _split_size_file_line(line)
        index = _column_index(tokens, column, header_tokens)
        if index is not None and 0 <= index < len(tokens):
            number = _parse_float(tokens[index])
            if number is not None:
                values.append(number)
            elif header_tokens is None:
                header_tokens = tokens
            continue

        numeric_tokens = [number for number in (_parse_float(token) for token in tokens) if number is not None]
        if numeric_tokens:
            values.append(numeric_tokens[0])
        elif header_tokens is None:
            header_tokens = tokens

    if not values:
        raise ValueError(f"size_distribution.size_file contains no numeric particle sizes: {path}")

    values_array = np.asarray(values, dtype=float)
    if np.any(values_array <= 0):
        raise ValueError("custom particle-size file must contain only positive values")
    return values_array


def _sample_from_size_file(size_cfg, n_particles: int, rng: np.random.Generator) -> np.ndarray:
    if not size_cfg.size_file:
        raise ValueError("custom_file distribution requires size_distribution.size_file")
    values = _load_size_file_values(size_cfg.size_file, size_cfg.file_column)
    choices = rng.choice(values, size=n_particles, replace=len(values) < n_particles)
    file_values = _normalized_key(size_cfg.file_values or "diameter")
    if file_values in {"diameter", "diameters", "d", "size", "particle_diameter"}:
        return choices / 2.0
    if file_values in {"radius", "radii", "r", "particle_radius"}:
        return choices
    raise ValueError("size_distribution.file_values must be either 'diameter' or 'radius'")


def sample_radii(config: PackingConfig, n_particles: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample dimensionless radii with mean diameter approximately equal to 1."""
    cls = config.structure.size_distribution
    size_cfg = config.size_distribution
    species_id = np.zeros(n_particles, dtype=int)

    if cls == SizeDistributionClass.MONODISPERSE:
        return np.full(n_particles, 0.5, dtype=float), species_id

    if cls == SizeDistributionClass.QUASI_MONODISPERSE:
        cv = float(0.03 if size_cfg.cv_radius is None else size_cfg.cv_radius)
        low = 0.5 * float(0.85 if size_cfg.min_radius_factor is None else size_cfg.min_radius_factor)
        high = 0.5 * float(1.15 if size_cfg.max_radius_factor is None else size_cfg.max_radius_factor)
        _validate_bounds(low, high)
        radii = _truncated_normal(rng, 0.5, 0.5 * cv, low, high, n_particles)
        radii *= 0.5 / np.mean(radii)
        return radii, species_id

    if cls == SizeDistributionClass.CONTINUOUS_POLYDISPERSE:
        distribution = _normalized_key(size_cfg.distribution or size_cfg.type or "lognormal")
        parameters = _normalized_parameters(size_cfg.parameters)
        cv = float(_parameter(parameters, ("CV_radius", "cv_radius", "cv", "coefficient_of_variation"), size_cfg.cv_radius or 0.15))
        if distribution in {"normal", "truncated_normal", "gaussian"}:
            low, high = _radius_bounds(size_cfg, parameters, default_low=0.5 * 0.4, default_high=0.5 * 1.8)
            radii = _truncated_normal(rng, 0.5, 0.5 * cv, low, high, n_particles)
            return _clip_and_rescale(radii, low, high), species_id
        if distribution in {"lognormal", "log_normal"}:
            sigma = float(_parameter(parameters, ("sigma", "log_sigma"), np.sqrt(np.log(1.0 + cv**2))))
            mu = np.log(0.5) - 0.5 * sigma**2
            radii = rng.lognormal(mean=mu, sigma=sigma, size=n_particles)
            low, high = _radius_bounds(size_cfg, parameters)
            return _clip_and_rescale(radii, low, high), species_id
        if distribution in {"uniform", "linear_uniform"}:
            low, high = _radius_bounds(size_cfg, parameters, default_low=0.5 * 0.5, default_high=0.5 * 1.5)
            radii = rng.uniform(low, high, size=n_particles)
            return _clip_and_rescale(radii, low, high), species_id
        if distribution in {"gamma"}:
            shape = _parameter(parameters, ("shape", "gamma_shape", "k"), None)
            shape = float(shape) if shape is not None else 1.0 / max(cv, 1.0e-12) ** 2
            if shape <= 0:
                raise ValueError("gamma distribution shape must be positive")
            radii = rng.gamma(shape=shape, scale=0.5 / shape, size=n_particles)
            low, high = _radius_bounds(size_cfg, parameters)
            return _clip_and_rescale(radii, low, high), species_id
        if distribution in {"weibull", "rosin_rammler", "rosin_rammler_weibull"}:
            shape = float(_parameter(parameters, ("shape", "weibull_shape", "rosin_rammler_shape"), 2.5))
            if shape <= 0:
                raise ValueError("Weibull/Rosin-Rammler shape must be positive")
            scale = 0.5 / gamma(1.0 + 1.0 / shape)
            radii = scale * rng.weibull(a=shape, size=n_particles)
            low, high = _radius_bounds(size_cfg, parameters)
            return _clip_and_rescale(radii, low, high), species_id
        if distribution in {"custom", "custom_file", "file", "uploaded_file", "imported"}:
            radii = _sample_from_size_file(size_cfg, n_particles, rng)
            low, high = _radius_bounds(size_cfg, parameters)
            return _clip_and_rescale(radii, low, high), species_id
        raise ValueError(f"Unsupported continuous distribution {distribution!r}")

    if cls == SizeDistributionClass.DISCRETE_MIXTURE:
        if not size_cfg.species:
            raise ValueError("discrete_mixture requires size_distribution.species")
        fractions = np.array([float(s.get("fraction", s.get("number_fraction", 0.0))) for s in size_cfg.species])
        if np.sum(fractions) <= 0:
            raise ValueError("mixture species fractions must sum to a positive value")
        fractions = fractions / np.sum(fractions)
        choices = rng.choice(len(size_cfg.species), size=n_particles, p=fractions)
        species_id = choices.astype(int)
        radii = np.array([float(size_cfg.species[i].get("radius", size_cfg.species[i].get("radius_factor", 1.0))) for i in choices])
        if np.mean(radii) > 1.5:
            radii = radii / 2.0
        radii *= 0.5 / np.mean(radii)
        return radii, species_id

    if cls == SizeDistributionClass.IMPORTED_DISTRIBUTION:
        if not size_cfg.size_file:
            raise ValueError("imported_distribution requires size_distribution.size_file")
        radii = _sample_from_size_file(size_cfg, n_particles, rng)
        return _clip_and_rescale(radii, None, None), species_id

    raise ValueError(f"Unsupported size distribution class {cls}")
