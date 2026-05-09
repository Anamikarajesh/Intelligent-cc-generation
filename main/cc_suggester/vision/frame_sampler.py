"""Frame sampling policy for event-aligned visual analysis."""

from __future__ import annotations


def sample_offsets(before: float = 1.0, after: float = 1.0) -> list[float]:
    """Return relative frame offsets around an event start/end window."""

    return [-before, -0.5, 0.0, 0.5, after]
