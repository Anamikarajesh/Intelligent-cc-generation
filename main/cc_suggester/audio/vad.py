"""Voice activity masking placeholder."""

from __future__ import annotations


def is_speech_masking_available() -> bool:
    """Return whether a real VAD backend has been configured."""

    return False
