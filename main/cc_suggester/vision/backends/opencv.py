"""OpenCV visual reaction backend."""

from __future__ import annotations

from pathlib import Path

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.errors import BackendUnavailableError
from cc_suggester.core.types import AudioEventCandidate, ReactionResult, VideoMetadata
from cc_suggester.vision.backends.base import VisionBackend


class OpenCvVisionBackend(VisionBackend):
    """Estimate scene reaction using frame differences around each event."""

    name = "opencv"
    requires_valid_media = True

    def analyze(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        audio_events: list[AudioEventCandidate],
        config: PipelineConfig,
    ) -> list[ReactionResult]:
        cv2 = _import_cv2()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise BackendUnavailableError(
                message="OpenCV could not open the input video.",
                code="opencv_open_failed",
                suggestions=[
                    "Run ccs inspect on the input file.",
                    "Try re-encoding the video to MP4/H.264.",
                ],
            )

        fps = metadata.fps or capture.get(cv2.CAP_PROP_FPS) or 25.0
        results: list[ReactionResult] = []
        try:
            for event in audio_events:
                frames = _sample_grayscale_frames(
                    cv2=cv2,
                    capture=capture,
                    fps=fps,
                    offsets=[
                        event.start_time - config.sample_window_before,
                        event.start_time,
                        (event.start_time + event.end_time) / 2.0,
                        event.end_time,
                        event.end_time + config.sample_window_after,
                    ],
                )
                motion = _motion_score(cv2, frames)
                reaction_confidence = round(max(0.0, min(0.95, motion * 3.5)), 3)
                results.append(
                    ReactionResult(
                        event_id=event.event_id,
                        start_time=event.start_time,
                        end_time=event.end_time,
                        reaction_confidence=reaction_confidence,
                        reaction_signals={
                            "scene_motion_delta": round(motion, 4),
                            "frame_difference": round(motion, 4),
                        },
                        frames_sampled=len(frames),
                        vision_backend=self.name,
                        debug_info={"fps": fps, "method": "grayscale frame difference"},
                    )
                )
        finally:
            capture.release()
        return results


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise BackendUnavailableError(
            message="The OpenCV vision backend requires opencv-python.",
            code="opencv_not_installed",
            suggestions=[
                "Install vision dependencies: pip install -r requirements-vision.txt",
                "Use --vision-backend mock for deterministic demos/tests.",
            ],
        ) from exc
    return cv2


def _sample_grayscale_frames(cv2, capture, fps: float, offsets: list[float]) -> list[object]:
    frames: list[object] = []
    for seconds in offsets:
        if seconds < 0:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        frames.append(gray)
    return frames


def _motion_score(cv2, frames: list[object]) -> float:
    if len(frames) < 2:
        return 0.0
    scores: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        diff = cv2.absdiff(previous, current)
        mean_score = float(diff.mean()) / 255.0
        changed_fraction = float((diff > 18).sum()) / float(diff.size)
        scores.append(max(mean_score * 8.0, changed_fraction * 5.0))
    return max(scores) if scores else 0.0
