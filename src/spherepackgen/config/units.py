"""Unit parsing helpers."""

from __future__ import annotations

import re

UNIT_TO_METER = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "um": 1.0e-6,
    "micron": 1.0e-6,
    "microns": 1.0e-6,
    "µm": 1.0e-6,
    "μm": 1.0e-6,
    "nm": 1.0e-9,
}


def parse_length_to_meters(value, default_unit: str | None = None) -> float:
    """Parse a numeric length or a string such as ``"300 nm"`` into meters."""
    if isinstance(value, (int, float)):
        if default_unit is None:
            return float(value)
        return float(value) * UNIT_TO_METER[default_unit]
    text = str(value).strip()
    if text.lower() == "auto":
        raise ValueError("Cannot parse auto as a length")
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]+)?", text)
    if not match:
        raise ValueError(f"Invalid length value: {value!r}")
    number = float(match.group(1))
    unit = match.group(2) or default_unit
    if unit is None:
        return number
    unit = unit.strip()
    if unit not in UNIT_TO_METER:
        raise ValueError(f"Unsupported length unit {unit!r}")
    return number * UNIT_TO_METER[unit]


def unit_scale_to_meters(unit: str) -> float:
    key = unit.strip()
    if key not in UNIT_TO_METER:
        raise ValueError(f"Unsupported output coordinate unit {unit!r}")
    return UNIT_TO_METER[key]

