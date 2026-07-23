"""FutsalMOT-RL exceptions hierarchy."""

from __future__ import annotations


class FutsalMOTRLError(Exception):
    """Base exception for all FutsalMOT-RL errors."""


class ConfigurationError(FutsalMOTRLError):
    """Project root or path resolution failure."""


class EpisodeValidationError(FutsalMOTRLError):
    """Episode data violates domain constraints."""


class TrainingError(FutsalMOTRLError):
    """Training pipeline failure (non-recoverable)."""


class UEIntegrationError(FutsalMOTRLError):
    """Unreal Engine integration failure."""
