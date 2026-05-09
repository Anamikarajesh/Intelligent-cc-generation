import subprocess
import sys
from pathlib import Path

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.pipeline import detect_audio_events, score_visual_reactions


def test_score_visual_reactions_from_audio_report(tmp_path: Path):
    sample_path = tmp_path / "sample_classroom.mp4"
    sidecar_path = sample_path.with_suffix(".wav")
    generator = Path(__file__).resolve().parents[1] / "scripts" / "generate_sample_video.py"
    subprocess.run(
        [sys.executable, str(generator), "--out", str(sample_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    audio_payload = detect_audio_events(
        sample_path,
        PipelineConfig(
            audio_backend="dsp",
            sidecar_audio_path=sidecar_path,
            output_dir=tmp_path / "outputs",
            audio_threshold=0.40,
        ),
    )

    vision_payload = score_visual_reactions(
        sample_path,
        Path(audio_payload["files"]["audio_json"]),
        PipelineConfig(
            vision_backend="opencv",
            output_dir=tmp_path / "outputs",
        ),
    )

    assert vision_payload["reactions"]
    assert Path(vision_payload["files"]["vision_json"]).exists()
    assert vision_payload["reactions"][0]["vision_backend"] == "opencv"
