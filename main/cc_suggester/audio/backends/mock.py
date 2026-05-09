"""Deterministic demo audio backend.

This backend keeps the first scaffold runnable without large model downloads. It
will be replaced by YAMNet/PANNs/AST/BEATs implementations through the same
interface.
"""

from __future__ import annotations

from pathlib import Path

from cc_suggester.audio.backends.base import AudioBackend
from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.types import AudioEventCandidate, VideoMetadata


class MockAudioBackend(AudioBackend):
    """Return classroom-style non-speech events for pipeline testing."""

    name = "mock"

    def detect(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        config: PipelineConfig,
    ) -> list[AudioEventCandidate]:
        duration = metadata.duration or 402.0
        anchors = [
            (0.34, "children_cheer", "Children cheering", 0.91),
            (0.43, "school_bell", "School bell", 0.86),
            (0.54, "applause", "Applause", 0.74),
            (0.71, "chair_scrape", "Chair scrape", 0.58),
            (0.81, "background_chatter", "Background chatter", 0.52),
        ]
        events: list[AudioEventCandidate] = []
        for ratio, event_id, label, confidence in anchors:
            start = max(0.0, min(duration - 1.0, duration * ratio))
            end = min(duration, start + _duration_for(event_id))
            if confidence < config.audio_threshold:
                continue
            events.append(
                AudioEventCandidate(
                    event_id=event_id,
                    label=label,
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    audio_confidence=confidence,
                    audio_backend=self.name,
                    raw_class_name=label,
                    debug_info={
                        "source": "deterministic mock backend",
                        "input_name": video_path.name,
                    },
                )
            )
        return events


def _duration_for(event_id: str) -> float:
    durations = {
        "children_cheer": 2.1,
        "school_bell": 1.6,
        "applause": 3.3,
        "chair_scrape": 1.1,
        "background_chatter": 7.5,
    }
    return durations.get(event_id, 1.5)
