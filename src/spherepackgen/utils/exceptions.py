"""Custom exceptions."""


class SpherePackGenError(Exception):
    """Base exception for SpherePackGen."""


class GenerationError(SpherePackGenError):
    """Raised when a generator cannot produce a requested structure."""


class ConfigurationError(SpherePackGenError):
    """Raised for invalid configuration."""

