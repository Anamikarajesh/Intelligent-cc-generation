"""Deterministic demo visual reaction backend."""

from __future__ import annotations

from pathlib import Path

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.types import AudioEventCandidate, ReactionResult, VideoMetadata
from cc_suggester.vision.backends.base import VisionBackend


class MockVisionBackend(VisionBackend):
    """Return plausible reaction scores for classroom-style events."""

    name = "mock"

    def analyze(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        audio_events: list[AudioEventCandidate],
        config: PipelineConfig,
    ) -> list[ReactionResult]:
        return [_reaction_for(event) for event in audio_events]


def _reaction_for(event: AudioEventCandidate) -> ReactionResult:
    reaction_map = {
        "children_cheer": (0.82, {"raised_hands": 0.89, "face_change": 0.72, "motion_spike": 0.78}),
        "school_bell": (0.61, {"head_turn": 0.67, "posture_shift": 0.52, "motion_spike": 0.64}),
        "applause": (0.54, {"hand_motion": 0.79, "face_change": 0.38, "motion_spike": 0.68}),
        "chair_scrape": (0.39, {"posture_shift": 0.35, "motion_spike": 0.42}),
        "background_chatter": (0.16, {"ambient_scene": 0.73, "head_turn": 0.09}),
    }
    confidence, signals = reaction_map.get(event.event_id, (0.25, {}))
    return ReactionResult(
        event_id=event.event_id,
        start_time=event.start_time,
        end_time=event.end_time,
        reaction_confidence=confidence,
        reaction_signals=signals,
        frames_sampled=7,
        vision_backend="mock",
        debug_info={"source": "deterministic mock backend"},
    )
