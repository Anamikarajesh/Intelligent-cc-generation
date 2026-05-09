import json

import pytest

from cc_suggester.output.review_export import build_review_export, suggestions_from_review_rows


def test_review_export_uses_edited_statuses_and_caption_text():
    rows = [
        {
            "index": 1,
            "event_id": "horn_honk",
            "start": 1.2,
            "end": 2.4,
            "caption": "[edited horn]",
            "status": "accepted",
            "audio": 0.9,
            "reaction": 0.8,
            "decision": 0.85,
            "reason": "Pipeline accepted this event.",
        },
        {
            "index": 2,
            "event_id": "traffic_noise",
            "start": 5.0,
            "end": 7.0,
            "caption": "[traffic]",
            "status": "rejected",
            "audio": 0.5,
            "reaction": 0.1,
            "decision": 0.2,
            "reason": "Ambient background noise.",
        },
    ]

    export = build_review_export(rows, "en")

    assert len(export.suggestions) == 2
    assert export.suggestions[0].accepted is True
    assert export.suggestions[0].caption_text == "[edited horn]"
    assert export.suggestions[1].accepted is False
    assert export.suggestions[1].requires_review is False
    assert "[edited horn]" in export.srt_text
    assert "[traffic]" not in export.srt_text
    assert "traffic_noise" in export.csv_text
    assert json.loads(export.json_text)["summary"]["accepted"] == 1


def test_review_export_preserves_review_state():
    rows = [
        {
            "event_id": "school_bell",
            "start_time": 10,
            "end_time": 11,
            "caption_text": "[school bell]",
            "status": "review",
        }
    ]

    suggestions = suggestions_from_review_rows(rows, "hi")

    assert suggestions[0].accepted is False
    assert suggestions[0].requires_review is True
    assert suggestions[0].language == "hi"
    assert suggestions[0].debug_info["editor_status"] == "review"


def test_review_export_rejects_unknown_status():
    rows = [{"event_id": "horn_honk", "status": "maybe"}]

    with pytest.raises(ValueError, match="Unknown review status"):
        suggestions_from_review_rows(rows, "en")
