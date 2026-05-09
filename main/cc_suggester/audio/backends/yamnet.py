"""Optional YAMNet sound event detection backend."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Sequence

from cc_suggester.audio.backends.base import AudioBackend
from cc_suggester.audio.extractor import extract_audio
from cc_suggester.audio.label_mapping import normalize_sound_label
from cc_suggester.audio.wav import load_wav_mono_float32
from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.errors import BackendUnavailableError
from cc_suggester.core.types import AudioEventCandidate, VideoMetadata


DEFAULT_YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16000
YAMNET_FRAME_HOP_SECONDS = 0.48
YAMNET_FRAME_DURATION_SECONDS = 0.96


class YamnetAudioBackend(AudioBackend):
    """Classify non-speech events using TensorFlow Hub YAMNet when installed."""

    name = "yamnet"
    requires_audio_file = True
    requires_valid_media = True

    def detect(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        config: PipelineConfig,
    ) -> list[AudioEventCandidate]:
        tf, hub, np = _import_dependencies()
        audio_path = _audio_path_for(video_path, config)
        waveform = load_wav_mono_float32(audio_path, target_sample_rate=YAMNET_SAMPLE_RATE)
        if not waveform:
            return []

        model_handle = config.yamnet_model or os.environ.get("CCS_YAMNET_MODEL") or DEFAULT_YAMNET_HANDLE
        try:
            model = hub.load(model_handle)
        except Exception as exc:
            raise BackendUnavailableError(
                message=f"YAMNet model could not be loaded from: {model_handle}",
                code="yamnet_model_load_failed",
                suggestions=[
                    "Use --audio-backend dsp for an offline CPU baseline.",
                    "Set CCS_YAMNET_MODEL to a local TensorFlow Hub YAMNet model directory.",
                    "Ensure internet/model cache access is available if using the default TF Hub handle.",
                ],
                details={"model_handle": model_handle, "error": str(exc)},
            ) from exc

        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        try:
            scores, _embeddings, _spectrogram = model(waveform_tensor)
        except Exception as exc:
            raise BackendUnavailableError(
                message="YAMNet inference failed.",
                code="yamnet_inference_failed",
                suggestions=[
                    "Verify the input audio is mono 16 kHz WAV or extractable video audio.",
                    "Try --audio-backend dsp to confirm audio extraction works.",
                ],
                details={"error": str(exc)},
            ) from exc

        class_names = _load_class_names(model, config.yamnet_class_map_path)
        scores_array = scores.numpy() if hasattr(scores, "numpy") else np.asarray(scores)
        return _events_from_scores(
            scores_array=scores_array,
            class_names=class_names,
            audio_path=audio_path,
            config=config,
        )


def _audio_path_for(video_path: Path, config: PipelineConfig) -> Path:
    if config.sidecar_audio_path is not None:
        return Path(config.sidecar_audio_path)
    if video_path.suffix.lower() == ".wav":
        return video_path
    run_dir = config.run_dir or config.output_dir / video_path.stem
    return extract_audio(video_path, run_dir / "artifacts" / "audio.wav")


def _import_dependencies():
    try:
        import numpy as np  # type: ignore
        import tensorflow as tf  # type: ignore
        import tensorflow_hub as hub  # type: ignore
    except Exception as exc:
        raise BackendUnavailableError(
            message="The YAMNet backend requires TensorFlow, TensorFlow Hub, and NumPy.",
            code="yamnet_dependencies_missing",
            suggestions=[
                "Install audio dependencies: pip install -r requirements-audio.txt",
                "Use --audio-backend dsp for an offline CPU baseline.",
                "Use --audio-backend mock for deterministic demos/tests.",
            ],
            details={"error": str(exc)},
        ) from exc
    return tf, hub, np


def _load_class_names(model: Any, class_map_path: Path | None) -> list[str]:
    path = class_map_path
    if path is None and hasattr(model, "class_map_path"):
        raw_path = model.class_map_path()
        if hasattr(raw_path, "numpy"):
            raw_path = raw_path.numpy()
        if isinstance(raw_path, bytes):
            raw_path = raw_path.decode("utf-8")
        path = Path(str(raw_path))

    if path is None:
        return []

    with Path(path).open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        class_names: list[str] = []
        for row in reader:
            class_names.append(row.get("display_name") or row.get("name") or row.get("label") or "")
        return class_names


def _events_from_scores(
    *,
    scores_array: Sequence[Sequence[float]],
    class_names: list[str],
    audio_path: Path,
    config: PipelineConfig,
) -> list[AudioEventCandidate]:
    events: list[AudioEventCandidate] = []
    for frame_index, frame_scores in enumerate(scores_array):
        scored_classes = _top_scored_classes(frame_scores, class_names, top_k=config.yamnet_top_k)
        event_scores: dict[str, tuple[float, str]] = {}
        for class_name, score in scored_classes:
            if score < config.audio_threshold:
                continue
            event_id = normalize_sound_label(class_name)
            if event_id is None:
                continue
            existing = event_scores.get(event_id)
            if existing is None or score > existing[0]:
                event_scores[event_id] = (score, class_name)

        for event_id, (score, class_name) in event_scores.items():
            start = frame_index * YAMNET_FRAME_HOP_SECONDS
            end = start + YAMNET_FRAME_DURATION_SECONDS
            events.append(
                AudioEventCandidate(
                    event_id=event_id,
                    label=event_id.replace("_", " ").title(),
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    audio_confidence=round(float(score), 3),
                    audio_backend="yamnet",
                    raw_class_name=class_name,
                    debug_info={
                        "audio_path": str(audio_path),
                        "frame_index": frame_index,
                        "yamnet_frame_hop_seconds": YAMNET_FRAME_HOP_SECONDS,
                        "yamnet_frame_duration_seconds": YAMNET_FRAME_DURATION_SECONDS,
                    },
                )
            )
    return events


def _top_scored_classes(
    frame_scores: Sequence[float],
    class_names: list[str],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    indexed = sorted(enumerate(frame_scores), key=lambda item: float(item[1]), reverse=True)
    output: list[tuple[str, float]] = []
    for class_index, score in indexed[:top_k]:
        if class_index < len(class_names):
            class_name = class_names[class_index]
        else:
            class_name = f"class_{class_index}"
        output.append((class_name, float(score)))
    return output
