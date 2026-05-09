"""Vision backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.types import AudioEventCandidate, ReactionResult, VideoMetadata


class VisionBackend(ABC):
    """Interface implemented by visual reaction analysis backends."""

    name: str
    requires_valid_media: bool = False

    @abstractmethod
    def analyze(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        audio_events: list[AudioEventCandidate],
        config: PipelineConfig,
    ) -> list[ReactionResult]:
        """Analyze visible reactions around audio event timestamps."""
