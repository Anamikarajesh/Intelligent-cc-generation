"""Configuration for pipeline runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


SUPPORTED_LANGUAGES = ("en", "hi", "ta", "te", "bn", "mr", "ml")
SUPPORTED_DEVICES = ("auto", "cpu", "cuda")


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration shared by CLI, UI, and future integrations."""

    language: str = "en"
    device: str = "auto"
    audio_backend: str = "mock"
    vision_backend: str = "mock"
    output_dir: Path = Path("outputs")
    sidecar_audio_path: Path | None = None
    yamnet_model: str | None = None
    yamnet_class_map_path: Path | None = None
    yamnet_top_k: int = 5
    audio_threshold: float = 0.45
    reaction_threshold: float = 0.35
    decision_threshold: float = 0.65
    review_threshold: float = 0.50
    min_event_duration: float = 0.25
    merge_gap: float = 0.40
    sample_window_before: float = 1.0
    sample_window_after: float = 1.0
    write_rejected_to_reports: bool = True
    allow_demo_input: bool = False
    run_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            supported = ", ".join(SUPPORTED_LANGUAGES)
            raise ValueError(f"Unsupported language '{self.language}'. Supported: {supported}")
        if self.device not in SUPPORTED_DEVICES:
            supported = ", ".join(SUPPORTED_DEVICES)
            raise ValueError(f"Unsupported device '{self.device}'. Supported: {supported}")
        self.output_dir = Path(self.output_dir)
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)
        if self.sidecar_audio_path is not None:
            self.sidecar_audio_path = Path(self.sidecar_audio_path)
        if self.yamnet_class_map_path is not None:
            self.yamnet_class_map_path = Path(self.yamnet_class_map_path)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["run_dir"] = str(self.run_dir) if self.run_dir else None
        data["sidecar_audio_path"] = str(self.sidecar_audio_path) if self.sidecar_audio_path else None
        data["yamnet_class_map_path"] = str(self.yamnet_class_map_path) if self.yamnet_class_map_path else None
        return data


def load_config(path: Path) -> PipelineConfig:
    """Load a JSON config file."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig(**payload)


def merge_config(base: PipelineConfig, **overrides: Any) -> PipelineConfig:
    """Return a config with non-None overrides applied."""

    data = base.to_dict()
    data.pop("run_dir", None)
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return PipelineConfig(**data)
