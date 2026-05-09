"""Glossary helpers for non-speech caption labels."""

from __future__ import annotations

from cc_suggester.decision.labels import LABELS, caption_for


def supported_event_ids() -> list[str]:
    """Return event IDs available in the curated glossary."""

    return sorted(LABELS)


def get_caption(event_id: str, language: str) -> str:
    """Return a caption from the curated glossary."""

    return caption_for(event_id, language)
