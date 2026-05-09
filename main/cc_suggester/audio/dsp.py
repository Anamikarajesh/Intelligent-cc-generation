"""Placeholder DSP feature definitions for the first scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DspFeatureSummary:
    """Small explainability summary for future DSP extraction."""

    rms_energy: float
    spectral_flux: float
    onset_strength: float


def describe_planned_features() -> list[str]:
    """Return the DSP features planned for the first real audio backend."""

    return [
        "RMS energy",
        "short-time Fourier transform",
        "log-mel spectrogram",
        "spectral flux",
        "onset strength",
        "zero-crossing rate",
    ]
