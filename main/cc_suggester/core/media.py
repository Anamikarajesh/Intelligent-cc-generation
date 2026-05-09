"""Video metadata inspection utilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from cc_suggester.core.errors import InvalidMediaError
from cc_suggester.core.types import VideoMetadata


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}


def inspect_video(path: Path) -> VideoMetadata:
    """Inspect a video using ffprobe when available, with a safe fallback."""

    path = Path(path)
    exists = path.exists()
    metadata = VideoMetadata(
        path=path,
        exists=exists,
        size_bytes=path.stat().st_size if exists else None,
        container=path.suffix.lstrip(".").lower() or None,
    )
    if not exists:
        return metadata

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        metadata.probe_error = "ffprobe not found"
        _inspect_with_opencv(metadata)
        return metadata

    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=True, text=True)
        payload = json.loads(completed.stdout or "{}")
    except Exception as exc:
        metadata.probe_error = str(exc)
        return metadata

    fmt = payload.get("format", {})
    try:
        metadata.duration = float(fmt["duration"]) if "duration" in fmt else None
    except (TypeError, ValueError):
        metadata.duration = None

    has_video = False
    has_audio = False
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "audio":
            has_audio = True
            metadata.audio_codec = stream.get("codec_name")
            sample_rate = stream.get("sample_rate")
            try:
                metadata.audio_sample_rate = int(sample_rate) if sample_rate else None
            except (TypeError, ValueError):
                metadata.audio_sample_rate = None
            metadata.audio_channels = stream.get("channels")
        if stream.get("codec_type") == "video":
            has_video = True
            metadata.video_codec = stream.get("codec_name")
            metadata.width = stream.get("width")
            metadata.height = stream.get("height")
            rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            metadata.fps = _parse_fraction(rate)
    metadata.has_audio = has_audio
    metadata.has_video = has_video
    return metadata


def _inspect_with_opencv(metadata: VideoMetadata) -> None:
    if metadata.path.suffix.lower() not in VIDEO_EXTENSIONS:
        return
    try:
        import cv2  # type: ignore
    except Exception:
        return

    capture = cv2.VideoCapture(str(metadata.path))
    if not capture.isOpened():
        return
    try:
        metadata.has_video = True
        metadata.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        metadata.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        metadata.fps = fps or None
        metadata.duration = frame_count / fps if fps > 0 and frame_count > 0 else None
        metadata.video_codec = "opencv-readable"
    finally:
        capture.release()


def validate_media(
    metadata: VideoMetadata,
    *,
    require_video: bool = True,
    require_audio: bool = True,
    allow_probe_failure: bool = False,
) -> None:
    """Validate input media for real processing backends."""

    if not metadata.exists:
        raise InvalidMediaError(
            message=f"Input file was not found: {metadata.path}",
            code="input_not_found",
            suggestions=["Check the path and run ccs inspect /path/to/video.mp4."],
        )

    suffix = metadata.path.suffix.lower()
    if require_video and suffix not in VIDEO_EXTENSIONS:
        raise InvalidMediaError(
            message=f"Input does not look like a supported video file: {metadata.path}",
            code="unsupported_video_type",
            suggestions=[
                "Use MP4, MKV, MOV, AVI, WEBM, or M4V video input.",
                "For demo-only testing, run with --allow-demo-input and mock backends.",
            ],
            details={"suffix": suffix or "none"},
        )

    if metadata.probe_error and not allow_probe_failure and metadata.has_video is not True:
        raise InvalidMediaError(
            message="Video metadata could not be probed.",
            code="probe_failed",
            suggestions=[
                "Install ffprobe/ffmpeg and ensure they are on PATH.",
                "Run ccs doctor to inspect the environment.",
            ],
            details={"probe_error": metadata.probe_error},
        )

    if require_video and metadata.has_video is False:
        raise InvalidMediaError(
            message="No video stream was found in the input file.",
            code="missing_video_stream",
            suggestions=["Use a video file that contains a valid video stream."],
        )

    if require_audio and metadata.has_audio is False:
        raise InvalidMediaError(
            message="No audio stream was found in the input file.",
            code="missing_audio_stream",
            suggestions=[
                "Use a video file that contains audio.",
                "Run ccs inspect /path/to/video.mp4 to confirm stream details.",
            ],
        )


def _parse_fraction(value: str | None) -> float | None:
    if not value:
        return None
    try:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    except Exception:
        try:
            return float(value)
        except Exception:
            return None
