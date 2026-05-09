"""Pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from cc_suggester.audio.backends import get_audio_backend
from cc_suggester.audio.events import smooth_events
from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.diagnostics import run_diagnostics
from cc_suggester.core.errors import BackendUnavailableError, InputNotFoundError
from cc_suggester.core.media import AUDIO_EXTENSIONS, inspect_video, validate_media
from cc_suggester.core.types import AudioEventCandidate, PipelineResult
from cc_suggester.decision.scorer import decide_captions
from cc_suggester.output.csv_report import write_csv_report
from cc_suggester.output.json_report import write_json_report
from cc_suggester.output.srt import write_srt
from cc_suggester.vision.backends import get_vision_backend


def analyze_video(video_path: Path, config: PipelineConfig) -> PipelineResult:
    """Run the full caption suggestion pipeline."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise InputNotFoundError(
            message=f"Input file was not found: {video_path}",
            code="input_not_found",
            suggestions=[
                "Check the path and filename.",
                "Run ccs inspect /path/to/video.mp4 to validate a video file.",
            ],
            details={"input_path": str(video_path)},
        )

    metadata = inspect_video(video_path)
    diagnostics = run_diagnostics(config)
    run_dir = _run_dir(config.output_dir, video_path)
    config.run_dir = run_dir

    try:
        audio_backend = get_audio_backend(config.audio_backend)
        vision_backend = get_vision_backend(config.vision_backend)
    except ValueError as exc:
        raise BackendUnavailableError(
            message=str(exc),
            code="backend_unavailable",
            suggestions=[
                "Use --audio-backend mock and --vision-backend mock for the first scaffold.",
                "Install the optional backend dependencies before selecting advanced backends.",
            ],
        ) from exc

    _validate_sidecar_audio(config)
    if audio_backend.requires_valid_media or vision_backend.requires_valid_media:
        is_audio_only_input = video_path.suffix.lower() in AUDIO_EXTENSIONS
        require_audio = audio_backend.requires_audio_file and config.sidecar_audio_path is None
        validate_media(
            metadata,
            require_video=vision_backend.requires_valid_media or not is_audio_only_input,
            require_audio=require_audio,
            allow_probe_failure=config.allow_demo_input or is_audio_only_input,
        )

    audio_events = smooth_events(audio_backend.detect(video_path, metadata, config), config)
    reactions = vision_backend.analyze(video_path, metadata, audio_events, config)
    suggestions = decide_captions(audio_events, reactions, config)

    result = PipelineResult(
        input_path=video_path,
        output_dir=run_dir,
        metadata=metadata,
        diagnostics=diagnostics,
        audio_events=audio_events,
        reactions=reactions,
        suggestions=suggestions,
        artifacts=_collect_artifacts(run_dir, config),
    )
    result.files = _write_outputs(result, config)
    return result


def detect_audio_events(video_path: Path, config: PipelineConfig) -> dict[str, object]:
    """Run only the audio detection stage and write an audio JSON report."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise InputNotFoundError(
            message=f"Input file was not found: {video_path}",
            code="input_not_found",
            suggestions=["Check the path and run ccs inspect /path/to/video.mp4."],
        )

    metadata = inspect_video(video_path)
    diagnostics = run_diagnostics(config)
    run_dir = _run_dir(config.output_dir, video_path)
    config.run_dir = run_dir

    try:
        audio_backend = get_audio_backend(config.audio_backend)
    except ValueError as exc:
        raise BackendUnavailableError(
            message=str(exc),
            code="backend_unavailable",
            suggestions=["Use --audio-backend mock or --audio-backend dsp."],
        ) from exc

    _validate_sidecar_audio(config)
    if audio_backend.requires_valid_media:
        is_audio_only_input = video_path.suffix.lower() in AUDIO_EXTENSIONS
        require_audio = audio_backend.requires_audio_file and config.sidecar_audio_path is None
        validate_media(
            metadata,
            require_video=not is_audio_only_input,
            require_audio=require_audio,
            allow_probe_failure=config.allow_demo_input or is_audio_only_input,
        )

    events = smooth_events(audio_backend.detect(video_path, metadata, config), config)
    payload: dict[str, object] = {
        "input_path": str(video_path),
        "output_dir": str(run_dir),
        "metadata": metadata.to_dict(),
        "diagnostics": diagnostics.to_dict(),
        "audio_events": [event.to_dict() for event in events],
        "artifacts": {name: str(path) for name, path in _collect_artifacts(run_dir, config).items()},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_json_report(payload, run_dir / "audio_events.json")
    payload["files"] = {"audio_json": str(report_path)}
    return payload


def score_visual_reactions(
    video_path: Path,
    audio_report_path: Path,
    config: PipelineConfig,
) -> dict[str, object]:
    """Run only visual reaction scoring from an existing audio event report."""

    video_path = Path(video_path)
    audio_report_path = Path(audio_report_path)
    if not video_path.exists():
        raise InputNotFoundError(
            message=f"Input file was not found: {video_path}",
            code="input_not_found",
            suggestions=["Check the path and run ccs inspect /path/to/video.mp4."],
        )
    if not audio_report_path.exists():
        raise InputNotFoundError(
            message=f"Audio event report was not found: {audio_report_path}",
            code="audio_report_not_found",
            suggestions=[
                "Run ccs audio first to generate audio_events.json.",
                "Pass a valid path to a pipeline results.json file or audio_events.json file.",
            ],
        )

    metadata = inspect_video(video_path)
    diagnostics = run_diagnostics(config)
    run_dir = _run_dir(config.output_dir, video_path)
    config.run_dir = run_dir

    try:
        vision_backend = get_vision_backend(config.vision_backend)
    except ValueError as exc:
        raise BackendUnavailableError(
            message=str(exc),
            code="backend_unavailable",
            suggestions=["Use --vision-backend mock or --vision-backend opencv."],
        ) from exc

    if vision_backend.requires_valid_media:
        validate_media(
            metadata,
            require_video=True,
            require_audio=False,
            allow_probe_failure=config.allow_demo_input,
        )

    audio_events = _load_audio_events(audio_report_path)
    reactions = vision_backend.analyze(video_path, metadata, audio_events, config)
    payload: dict[str, object] = {
        "input_path": str(video_path),
        "audio_report_path": str(audio_report_path),
        "output_dir": str(run_dir),
        "metadata": metadata.to_dict(),
        "diagnostics": diagnostics.to_dict(),
        "audio_events": [event.to_dict() for event in audio_events],
        "reactions": [reaction.to_dict() for reaction in reactions],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_json_report(payload, run_dir / "vision_reactions.json")
    payload["files"] = {"vision_json": str(report_path)}
    return payload


def export_from_report(report_path: Path, output_path: Path, language: str) -> Path:
    """Export SRT from a JSON report produced by the pipeline."""

    import json

    from cc_suggester.core.types import CaptionSuggestion
    from cc_suggester.decision.labels import caption_for

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    suggestions: list[CaptionSuggestion] = []
    for item in payload.get("suggestions", []):
        suggestions.append(
            CaptionSuggestion(
                event_id=item["event_id"],
                start_time=float(item["start_time"]),
                end_time=float(item["end_time"]),
                audio_confidence=float(item["audio_confidence"]),
                reaction_confidence=float(item["reaction_confidence"]),
                decision_score=float(item["decision_score"]),
                accepted=bool(item["accepted"]),
                reason=str(item["reason"]),
                caption_text=caption_for(str(item["event_id"]), language),
                language=language,
                requires_review=bool(item.get("requires_review", False)),
                debug_info=item.get("debug_info", {}),
            )
        )
    return write_srt(suggestions, output_path)


def _load_audio_events(report_path: Path) -> list[AudioEventCandidate]:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    raw_events = payload.get("audio_events", [])
    events: list[AudioEventCandidate] = []
    for item in raw_events:
        events.append(
            AudioEventCandidate(
                event_id=str(item["event_id"]),
                label=str(item.get("label") or item["event_id"]),
                start_time=float(item["start_time"]),
                end_time=float(item["end_time"]),
                audio_confidence=float(item["audio_confidence"]),
                audio_backend=str(item.get("audio_backend") or "unknown"),
                raw_class_name=item.get("raw_class_name"),
                debug_info=item.get("debug_info", {}),
            )
        )
    return events


def _write_outputs(result: PipelineResult, config: PipelineConfig) -> dict[str, Path]:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "srt": result.output_dir / f"captions.{config.language}.srt",
        "json": result.output_dir / "results.json",
        "csv": result.output_dir / "events.csv",
        "diagnostics": result.output_dir / "diagnostics.json",
        "config": result.output_dir / "config.json",
    }
    result.files = files
    write_srt(result.suggestions, files["srt"])
    write_csv_report(result.suggestions, files["csv"])
    write_json_report(result.to_dict(), files["json"])
    write_json_report(result.diagnostics.to_dict(), files["diagnostics"])
    write_json_report(config.to_dict(), files["config"])
    return files


def _run_dir(output_dir: Path, video_path: Path) -> Path:
    stem = video_path.stem or "video"
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem)
    return Path(output_dir) / safe_stem


def _collect_artifacts(run_dir: Path, config: PipelineConfig) -> dict[str, Path]:
    artifacts = {
        "audio_wav": run_dir / "artifacts" / "audio.wav",
    }
    if config.sidecar_audio_path is not None:
        artifacts["audio_wav"] = Path(config.sidecar_audio_path)
    return {name: path for name, path in artifacts.items() if path.exists()}


def _validate_sidecar_audio(config: PipelineConfig) -> None:
    if config.sidecar_audio_path is None:
        return
    if not Path(config.sidecar_audio_path).exists():
        raise InputNotFoundError(
            message=f"Sidecar audio file was not found: {config.sidecar_audio_path}",
            code="sidecar_audio_not_found",
            suggestions=[
                "Check the --audio-path value.",
                "Generate sample media with python scripts/generate_sample_video.py.",
            ],
            details={"audio_path": str(config.sidecar_audio_path)},
        )
