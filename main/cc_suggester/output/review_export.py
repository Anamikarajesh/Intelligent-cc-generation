"""Helpers for exporting manually reviewed caption suggestions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cc_suggester.core.types import CaptionSuggestion
from cc_suggester.output.csv_report import render_csv_report, write_csv_report
from cc_suggester.output.json_report import write_json_report
from cc_suggester.output.srt import render_srt, write_srt


VALID_REVIEW_STATUSES = {"accepted", "review", "rejected"}


@dataclass(frozen=True, slots=True)
class ReviewExport:
    """In-memory reviewed export payload for UI download buttons."""

    suggestions: list[CaptionSuggestion]
    srt_text: str
    csv_text: str
    json_text: str


def build_review_export(rows: Sequence[Mapping[str, Any]], language: str) -> ReviewExport:
    """Convert editable review rows into exportable SRT and CSV content."""

    suggestions = suggestions_from_review_rows(rows, language)
    return ReviewExport(
        suggestions=suggestions,
        srt_text=render_srt(suggestions),
        csv_text=render_csv_report(suggestions),
        json_text=json.dumps(
            review_payload(suggestions, language),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def write_review_exports(rows: Sequence[Mapping[str, Any]], output_dir: Path, language: str) -> dict[str, Path]:
    """Write reviewed SRT, CSV, and JSON files to a directory."""

    export = build_review_export(rows, language)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "reviewed_srt": write_srt(export.suggestions, output_dir / f"reviewed_captions.{language}.srt"),
        "reviewed_csv": write_csv_report(export.suggestions, output_dir / "reviewed_events.csv"),
    }
    files["reviewed_json"] = write_json_report(
        review_payload(export.suggestions, language),
        output_dir / "reviewed_results.json",
    )
    return files


def review_payload(suggestions: Sequence[CaptionSuggestion], language: str) -> dict[str, Any]:
    """Build a JSON-serializable reviewed session payload."""

    return {
        "language": language,
        "suggestions": [suggestion.to_dict() for suggestion in suggestions],
        "summary": {
            "total": len(suggestions),
            "accepted": sum(1 for item in suggestions if item.accepted),
            "review": sum(1 for item in suggestions if item.requires_review),
            "rejected": sum(1 for item in suggestions if not item.accepted and not item.requires_review),
        },
    }


def suggestions_from_review_rows(rows: Sequence[Mapping[str, Any]], language: str) -> list[CaptionSuggestion]:
    """Build caption suggestions from Web UI review rows."""

    suggestions: list[CaptionSuggestion] = []
    for fallback_index, row in enumerate(rows, start=1):
        status = _status_for(row)
        caption_text = _string_for(row, ("caption", "caption_text"), default="").strip()
        suggestions.append(
            CaptionSuggestion(
                event_id=_string_for(row, ("event_id",), default=f"event_{fallback_index}"),
                start_time=_float_for(row, ("start", "start_time"), default=0.0),
                end_time=_float_for(row, ("end", "end_time"), default=0.0),
                audio_confidence=_float_for(row, ("audio", "audio_confidence"), default=0.0),
                reaction_confidence=_float_for(row, ("reaction", "reaction_confidence"), default=0.0),
                decision_score=_float_for(row, ("decision", "decision_score"), default=0.0),
                accepted=status == "accepted",
                requires_review=status == "review",
                reason=_reason_for(row, status),
                caption_text=caption_text,
                language=language,
                debug_info={
                    "editor_status": status,
                    "review_index": row.get("index", fallback_index),
                    "source": "review_export",
                },
            )
        )
    return suggestions


def _status_for(row: Mapping[str, Any]) -> str:
    status = _string_for(row, ("status",), default="").strip().lower()
    if not status:
        if bool(row.get("accepted", False)):
            status = "accepted"
        elif bool(row.get("requires_review", False)):
            status = "review"
        else:
            status = "rejected"
    if status not in VALID_REVIEW_STATUSES:
        valid = ", ".join(sorted(VALID_REVIEW_STATUSES))
        raise ValueError(f"Unknown review status '{status}'. Expected one of: {valid}.")
    return status


def _reason_for(row: Mapping[str, Any], status: str) -> str:
    existing = _string_for(row, ("reason",), default="").strip()
    editor_note = f"Editor marked this event as {status}."
    if not existing:
        return editor_note
    if existing.endswith(editor_note):
        return existing
    return f"{existing} {editor_note}"


def _string_for(row: Mapping[str, Any], keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    return default


def _float_for(row: Mapping[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            return default
    return default
