from pathlib import Path

from cc_suggester.audio.backends.yamnet import _events_from_scores
from cc_suggester.audio.label_mapping import normalize_sound_label
from cc_suggester.core.config import PipelineConfig


def test_normalize_sound_label_common_yamnet_classes():
    assert normalize_sound_label("Vehicle horn, car horn, honking") == "horn_honk"
    assert normalize_sound_label("Glass") == "glass_break"
    assert normalize_sound_label("Applause") == "applause"
    assert normalize_sound_label("Siren") == "siren"
    assert normalize_sound_label("Speech") is None


def test_events_from_yamnet_scores_maps_classes_to_events(tmp_path: Path):
    scores = [
        [0.91, 0.10, 0.05],
        [0.05, 0.82, 0.02],
        [0.02, 0.06, 0.78],
    ]
    class_names = [
        "Vehicle horn, car horn, honking",
        "Glass",
        "Applause",
    ]

    events = _events_from_scores(
        scores_array=scores,
        class_names=class_names,
        audio_path=tmp_path / "audio.wav",
        config=PipelineConfig(audio_backend="yamnet", audio_threshold=0.40, yamnet_top_k=2),
    )

    assert [event.event_id for event in events] == ["horn_honk", "glass_break", "applause"]
    assert events[0].audio_backend == "yamnet"
    assert events[1].start_time == 0.48
    assert events[2].raw_class_name == "Applause"
