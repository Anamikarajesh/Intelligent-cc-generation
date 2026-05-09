"""Post-processing helpers for audio events."""

from __future__ import annotations

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.types import AudioEventCandidate


def smooth_events(
    events: list[AudioEventCandidate],
    config: PipelineConfig,
) -> list[AudioEventCandidate]:
    """Merge adjacent same-label events and remove very short events."""

    filtered = [
        event
        for event in sorted(events, key=lambda item: (item.start_time, item.end_time))
        if event.end_time - event.start_time >= config.min_event_duration
    ]
    if not filtered:
        return []

    merged: list[AudioEventCandidate] = [filtered[0]]
    for event in filtered[1:]:
        previous = merged[-1]
        gap = event.start_time - previous.end_time
        if event.event_id == previous.event_id and gap <= config.merge_gap:
            previous.end_time = max(previous.end_time, event.end_time)
            previous.audio_confidence = max(previous.audio_confidence, event.audio_confidence)
            previous.debug_info["merged"] = True
        else:
            merged.append(event)
    return merged
