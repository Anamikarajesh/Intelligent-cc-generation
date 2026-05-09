"""SRT export."""

from __future__ import annotations

from pathlib import Path

from cc_suggester.core.types import CaptionSuggestion


def render_srt(suggestions: list[CaptionSuggestion]) -> str:
    """Render accepted caption suggestions as SRT text."""

    accepted = [item for item in suggestions if item.accepted]
    lines: list[str] = []
    for index, suggestion in enumerate(accepted, start=1):
        lines.extend(
            [
                str(index),
                f"{format_srt_time(suggestion.start_time)} --> {format_srt_time(suggestion.end_time)}",
                suggestion.caption_text,
                "",
            ]
        )
    return "\n".join(lines)


def write_srt(suggestions: list[CaptionSuggestion], output_path: Path) -> Path:
    """Write accepted caption suggestions to an SRT file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_srt(suggestions), encoding="utf-8")
    return output_path


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp."""

    safe_seconds = max(0.0, seconds)
    hours = int(safe_seconds // 3600)
    minutes = int((safe_seconds % 3600) // 60)
    whole_seconds = int(safe_seconds % 60)
    milliseconds = int(round((safe_seconds - int(safe_seconds)) * 1000))
    if milliseconds == 1000:
        milliseconds = 0
        whole_seconds += 1
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
