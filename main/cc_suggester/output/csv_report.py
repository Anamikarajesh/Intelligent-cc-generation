"""CSV review report export."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from cc_suggester.core.types import CaptionSuggestion


FIELDNAMES = [
    "event_id",
    "start_time",
    "end_time",
    "caption_text",
    "language",
    "audio_confidence",
    "reaction_confidence",
    "decision_score",
    "accepted",
    "requires_review",
    "reason",
]


def render_csv_report(suggestions: list[CaptionSuggestion]) -> str:
    """Render a reviewer-friendly CSV report."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES)
    writer.writeheader()
    for suggestion in suggestions:
        writer.writerow(
            {
                "event_id": suggestion.event_id,
                "start_time": suggestion.start_time,
                "end_time": suggestion.end_time,
                "caption_text": suggestion.caption_text,
                "language": suggestion.language,
                "audio_confidence": suggestion.audio_confidence,
                "reaction_confidence": suggestion.reaction_confidence,
                "decision_score": suggestion.decision_score,
                "accepted": suggestion.accepted,
                "requires_review": suggestion.requires_review,
                "reason": suggestion.reason,
            }
        )
    return buffer.getvalue()


def write_csv_report(suggestions: list[CaptionSuggestion], output_path: Path) -> Path:
    """Write a reviewer-friendly CSV report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_csv_report(suggestions), encoding="utf-8")
    return output_path
