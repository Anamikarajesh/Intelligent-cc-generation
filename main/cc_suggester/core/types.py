"""Shared data models passed between pipeline modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(slots=True)
class VideoMetadata:
    """Basic metadata discovered from an input video."""

    path: Path
    exists: bool
    size_bytes: int | None = None
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    has_video: bool | None = None
    audio_codec: str | None = None
    video_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    container: str | None = None
    probe_error: str | None = None

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class DiagnosticsReport:
    """Environment and device diagnostics for a run."""

    python_version: str
    ffmpeg_path: str | None
    ffprobe_path: str | None
    selected_device: str
    actual_device: str
    cuda_available: bool
    gpu_name: str | None = None
    torch_available: bool = False
    fallback_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class AudioEventCandidate:
    """Detected sound event before visual reaction analysis."""

    event_id: str
    label: str
    start_time: float
    end_time: float
    audio_confidence: float
    audio_backend: str
    raw_class_name: str | None = None
    debug_info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class ReactionResult:
    """Visual reaction evidence around an audio event."""

    event_id: str
    start_time: float
    end_time: float
    reaction_confidence: float
    reaction_signals: JsonDict = field(default_factory=dict)
    frames_sampled: int = 0
    vision_backend: str = "mock"
    debug_info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class CaptionSuggestion:
    """Final caption decision for a candidate event."""

    event_id: str
    start_time: float
    end_time: float
    audio_confidence: float
    reaction_confidence: float
    decision_score: float
    accepted: bool
    reason: str
    caption_text: str
    language: str
    requires_review: bool = False
    debug_info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    """Complete result returned by the pipeline."""

    input_path: Path
    output_dir: Path
    metadata: VideoMetadata
    diagnostics: DiagnosticsReport
    audio_events: list[AudioEventCandidate]
    reactions: list[ReactionResult]
    suggestions: list[CaptionSuggestion]
    files: dict[str, Path] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "metadata": self.metadata.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "audio_events": [event.to_dict() for event in self.audio_events],
            "reactions": [reaction.to_dict() for reaction in self.reactions],
            "suggestions": [suggestion.to_dict() for suggestion in self.suggestions],
            "files": {name: str(path) for name, path in self.files.items()},
            "artifacts": {name: str(path) for name, path in self.artifacts.items()},
        }
