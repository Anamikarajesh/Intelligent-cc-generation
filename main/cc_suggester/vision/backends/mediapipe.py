"""Optional MediaPipe visual reaction backend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.errors import BackendUnavailableError
from cc_suggester.core.types import AudioEventCandidate, ReactionResult, VideoMetadata
from cc_suggester.vision.backends.base import VisionBackend

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def _ensure_model() -> Path:
    import os
    cache_dir = Path(os.path.expanduser("~/.cache/cc_suggester"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "pose_landmarker_lite.task"
    if not model_path.exists():
        import urllib.request
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


class MediaPipeVisionBackend(VisionBackend):
    """Estimate pose-based reaction signals around audio events."""

    name = "mediapipe"
    requires_valid_media = True

    def analyze(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        audio_events: list[AudioEventCandidate],
        config: PipelineConfig,
    ) -> list[ReactionResult]:
        cv2, mp, PoseLandmarker, VisionTaskRunningMode, BaseOptions = _import_dependencies()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise BackendUnavailableError(
                message="OpenCV could not open the input video for MediaPipe analysis.",
                code="mediapipe_video_open_failed",
                suggestions=[
                    "Run ccs inspect on the input file.",
                    "Try --vision-backend opencv to confirm basic video decoding works.",
                ],
            )

        fps = metadata.fps or capture.get(cv2.CAP_PROP_FPS) or 25.0
        results: list[ReactionResult] = []

        from mediapipe.tasks.python.vision import PoseLandmarkerOptions
        model_path = _ensure_model()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            output_segmentation_masks=False,
        )
        pose = PoseLandmarker.create_from_options(options)
        try:
            for event in audio_events:
                frames = _sample_frames(cv2, capture, fps, _event_offsets(event, config))
                landmarks = [_pose_landmarks(cv2, mp, pose, frame) for frame in frames]
                landmarks = [item for item in landmarks if item is not None]
                pose_motion = _landmark_motion(landmarks)
                head_motion = _head_motion(landmarks)
                visibility = len(landmarks) / max(1, len(frames))
                reaction_confidence = round(
                    max(0.0, min(0.95, (pose_motion * 3.0) + (head_motion * 2.0) + (visibility * 0.08))),
                    3,
                )
                results.append(
                    ReactionResult(
                        event_id=event.event_id,
                        start_time=event.start_time,
                        end_time=event.end_time,
                        reaction_confidence=reaction_confidence,
                        reaction_signals={
                            "pose_motion": round(pose_motion, 4),
                            "head_motion": round(head_motion, 4),
                            "pose_visibility": round(visibility, 4),
                        },
                        frames_sampled=len(frames),
                        vision_backend=self.name,
                        debug_info={"fps": fps, "landmark_frames": len(landmarks)},
                    )
                )
        finally:
            pose.close()
            capture.release()
        return results


def _import_dependencies():
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python.vision import PoseLandmarker
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception as exc:
        raise BackendUnavailableError(
            message="The MediaPipe vision backend requires mediapipe and opencv-python.",
            code="mediapipe_not_installed",
            suggestions=[
                "Install vision dependencies: pip install -r requirements-vision.txt",
                "Use --vision-backend opencv for a CPU scene-motion baseline.",
                "Use --vision-backend mock for deterministic demos/tests.",
            ],
            details={"error": str(exc)},
        ) from exc
    return cv2, mp, PoseLandmarker, VisionTaskRunningMode, BaseOptions


def _event_offsets(event: AudioEventCandidate, config: PipelineConfig) -> list[float]:
    midpoint = (event.start_time + event.end_time) / 2.0
    return [
        event.start_time - config.sample_window_before,
        event.start_time,
        midpoint,
        event.end_time,
        event.end_time + config.sample_window_after,
    ]


def _sample_frames(cv2, capture, fps: float, offsets: list[float]) -> list[Any]:
    frames: list[Any] = []
    for seconds in offsets:
        if seconds < 0:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
    return frames


def _pose_landmarks(cv2, mp, pose, frame) -> list[tuple[float, float, float]] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return [(lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks[0]]


def _landmark_motion(frames: list[list[tuple[float, float, float]]]) -> float:
    if len(frames) < 2:
        return 0.0
    indices = [0, 11, 12, 15, 16, 23, 24]
    motions: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        distances = []
        for index in indices:
            if index >= len(previous) or index >= len(current):
                continue
            if previous[index][2] < 0.35 or current[index][2] < 0.35:
                continue
            distances.append(_distance(previous[index], current[index]))
        if distances:
            motions.append(sum(distances) / len(distances))
    return max(motions) if motions else 0.0


def _head_motion(frames: list[list[tuple[float, float, float]]]) -> float:
    if len(frames) < 2:
        return 0.0
    motions: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        if previous[0][2] < 0.35 or current[0][2] < 0.35:
            continue
        motions.append(_distance(previous[0], current[0]))
    return max(motions) if motions else 0.0


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)
