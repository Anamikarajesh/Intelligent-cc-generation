"""Normalize model-specific sound labels into project event IDs."""

from __future__ import annotations


LABEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("horn_honk", ("vehicle horn", "car horn", "honking", "horn")),
    ("glass_break", ("glass", "shatter", "breaking")),
    ("crowd_cheer", ("cheering", "cheer", "crowd cheering")),
    ("applause", ("applause", "clapping")),
    ("laughter", ("laughter", "laughing", "giggle")),
    ("music", ("music", "song", "singing", "musical")),
    ("alarm", ("alarm", "beep", "buzzer")),
    ("siren", ("siren", "police car", "ambulance")),
    ("explosion", ("explosion", "blast", "boom")),
    ("gunshot", ("gunshot", "gunfire", "shooting")),
    ("door_slam", ("door", "slam", "knock")),
    ("phone_ring", ("telephone", "ringtone", "ringing", "phone")),
    ("dog_bark", ("bark", "dog")),
)


def normalize_sound_label(label: str) -> str | None:
    """Map an AudioSet/YAMNet label to an internal event ID."""

    normalized = label.lower().replace("_", " ")
    for event_id, needles in LABEL_RULES:
        required = _required_tokens(event_id)
        if required and all(_matches(normalized, token) for token in required):
            return event_id
        if any(needle in normalized for needle in needles):
            return event_id
    return None


def _matches(label: str, token: str) -> bool:
    return token in label


def _required_tokens(event_id: str) -> tuple[str, ...]:
    if event_id == "glass_break":
        return ("glass",)
    if event_id == "door_slam":
        return ("door",)
    return ()
