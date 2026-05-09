from cc_suggester.core.types import CaptionSuggestion
from cc_suggester.decision.labels import caption_for
from cc_suggester.output.csv_report import render_csv_report
from cc_suggester.output.srt import format_srt_time, write_srt


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(62.345) == "00:01:02,345"


def test_caption_for_known_language():
    assert caption_for("horn_honk", "hi") == "[हॉर्न बजता है]"
    assert caption_for("impact_sound", "ml") == "[പെട്ടെന്നുള്ള ശബ്ദം]"
    assert caption_for("siren", "ta") == "[சைரன் ஒலிக்கிறது]"


def test_write_srt_only_accepts_accepted(tmp_path):
    suggestions = [
        CaptionSuggestion(
            event_id="horn_honk",
            start_time=1.0,
            end_time=2.0,
            audio_confidence=0.9,
            reaction_confidence=0.8,
            decision_score=0.8,
            accepted=True,
            reason="accepted",
            caption_text="[horn honks]",
            language="en",
        ),
        CaptionSuggestion(
            event_id="background_chatter",
            start_time=3.0,
            end_time=4.0,
            audio_confidence=0.5,
            reaction_confidence=0.1,
            decision_score=0.2,
            accepted=False,
            reason="rejected",
            caption_text="[background chatter]",
            language="en",
        ),
    ]
    output = tmp_path / "captions.srt"
    write_srt(suggestions, output)
    text = output.read_text(encoding="utf-8")
    assert "[horn honks]" in text
    assert "[background chatter]" not in text


def test_render_csv_report_includes_review_flags():
    suggestions = [
        CaptionSuggestion(
            event_id="school_bell",
            start_time=10.0,
            end_time=11.0,
            audio_confidence=0.7,
            reaction_confidence=0.4,
            decision_score=0.6,
            accepted=False,
            requires_review=True,
            reason="borderline",
            caption_text="[school bell rings]",
            language="en",
        )
    ]

    text = render_csv_report(suggestions)

    assert "school_bell" in text
    assert "requires_review" in text
    assert "True" in text
