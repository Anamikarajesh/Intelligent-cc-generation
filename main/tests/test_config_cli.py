import json

from cc_suggester.cli.app import main
from cc_suggester.core.config import PipelineConfig, load_config, merge_config


def test_load_and_merge_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"language": "hi", "audio_backend": "mock", "vision_backend": "mock"}),
        encoding="utf-8",
    )

    loaded = load_config(config_path)
    merged = merge_config(loaded, language="ml", device="cpu")

    assert loaded.language == "hi"
    assert merged.language == "ml"
    assert merged.device == "cpu"


def test_cli_labels_command(capsys):
    exit_code = main(["labels"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Supported languages" in captured.out
    assert "horn_honk" in captured.out


def test_cli_unknown_command_suggests_analyze(capsys):
    exit_code = main(["analize"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Did you mean: analyze?" in captured.err
