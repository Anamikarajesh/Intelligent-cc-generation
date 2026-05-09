"""Audio extraction helpers for real model backends."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cc_suggester.core.errors import AudioExtractionError, BackendUnavailableError


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """Extract mono WAV audio using ffmpeg."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise BackendUnavailableError(
            message="ffmpeg was not found, so audio extraction cannot run.",
            code="ffmpeg_missing",
            suggestions=[
                "Install ffmpeg and ensure it is on PATH.",
                "Run ccs doctor to verify the environment.",
            ],
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    try:
        subprocess.run(command, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AudioExtractionError(
            message="ffmpeg failed while extracting audio.",
            code="audio_extraction_failed",
            suggestions=[
                "Run ccs inspect on the input file.",
                "Try a different video container or re-encode the video.",
                "Run ccs doctor to verify ffmpeg availability.",
            ],
            details={
                "stderr": exc.stderr[-1000:] if exc.stderr else "",
                "returncode": exc.returncode,
            },
        ) from exc
    return output_path
