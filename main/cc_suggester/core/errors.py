"""Friendly error types surfaced by CLI and UI clients."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CCSuggesterError(Exception):
    """Base exception with user-facing suggestions."""

    message: str
    code: str = "ccs_error"
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class InputNotFoundError(CCSuggesterError):
    """Raised when the requested input video does not exist."""


class InvalidMediaError(CCSuggesterError):
    """Raised when a file cannot be processed as required media."""


class AudioExtractionError(CCSuggesterError):
    """Raised when ffmpeg audio extraction fails."""


class DeviceUnavailableError(CCSuggesterError):
    """Raised when a required device, such as CUDA, is unavailable."""


class BackendUnavailableError(CCSuggesterError):
    """Raised when a requested model backend is not installed or registered."""
