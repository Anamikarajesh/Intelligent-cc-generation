"""Vision backend registry."""

from cc_suggester.vision.backends.base import VisionBackend
from cc_suggester.vision.backends.mediapipe import MediaPipeVisionBackend
from cc_suggester.vision.backends.mock import MockVisionBackend
from cc_suggester.vision.backends.opencv import OpenCvVisionBackend


def get_vision_backend(name: str) -> VisionBackend:
    """Return a visual reaction backend by name."""

    normalized = name.lower().strip()
    if normalized in {"mock", "demo"}:
        return MockVisionBackend()
    if normalized in {"opencv", "cv2"}:
        return OpenCvVisionBackend()
    if normalized == "mediapipe":
        return MediaPipeVisionBackend()
    raise ValueError(
        f"Unknown vision backend '{name}'. Available: mock, opencv, mediapipe. "
        "Planned advanced backends: mmpose, mmaction."
    )
