import subprocess
import sys
from pathlib import Path

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.media import inspect_video
from cc_suggester.core.pipeline import analyze_video


def test_real_sample_video_inspect_and_analyze(tmp_path: Path):
    sample_path = tmp_path / "sample_classroom.mp4"
    sidecar_path = sample_path.with_suffix(".wav")
    generator = Path(__file__).resolve().parents[1] / "scripts" / "generate_sample_video.py"

    subprocess.run(
        [sys.executable, str(generator), "--out", str(sample_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = inspect_video(sample_path)
    assert metadata.exists
    assert metadata.has_video is True
    assert metadata.duration is not None

    result = analyze_video(
        sample_path,
        PipelineConfig(
            language="en",
            audio_backend="dsp",
            vision_backend="opencv",
            output_dir=tmp_path / "outputs",
            sidecar_audio_path=sidecar_path,
            audio_threshold=0.40,
        ),
    )

    assert result.files["srt"].exists()
    assert result.files["json"].exists()
    assert result.artifacts["audio_wav"].exists()
    assert result.audio_events
    assert any(suggestion.accepted for suggestion in result.suggestions)
