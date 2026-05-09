"""Registered placeholders for planned advanced audio backends."""

from __future__ import annotations

from pathlib import Path

from cc_suggester.audio.backends.base import AudioBackend
from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.errors import BackendUnavailableError
from cc_suggester.core.types import AudioEventCandidate, VideoMetadata


class UnavailableAudioBackend(AudioBackend):
    """Backend placeholder that explains how to proceed."""

    requires_audio_file = True
    requires_valid_media = True

    def __init__(self, name: str, install_hint: str) -> None:
        self.name = name
        self.install_hint = install_hint

    def detect(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        config: PipelineConfig,
    ) -> list[AudioEventCandidate]:
        raise BackendUnavailableError(
            message=f"The {self.name} backend is registered but not implemented in this environment yet.",
            code=f"{self.name}_not_installed",
            suggestions=[
                "Use --audio-backend dsp for a real offline CPU baseline.",
                "Use --audio-backend mock for deterministic demos/tests.",
                self.install_hint,
            ],
        )
