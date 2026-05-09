import math
import wave
from pathlib import Path

from cc_suggester.audio.backends.dsp import DspAudioBackend
from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.types import VideoMetadata


def test_dsp_backend_detects_synthetic_loud_region(tmp_path: Path):
    wav_path = tmp_path / "synthetic.wav"
    _write_synthetic_wav(wav_path)
    backend = DspAudioBackend()
    config = PipelineConfig(audio_backend="dsp", vision_backend="mock", audio_threshold=0.40, run_dir=tmp_path)
    metadata = VideoMetadata(path=wav_path, exists=True, has_audio=True, has_video=False, duration=3.0)

    events = backend.detect(wav_path, metadata, config)

    assert events
    assert events[0].audio_backend == "dsp"
    assert events[0].audio_confidence >= 0.40
    assert events[0].start_time < 1.5
    assert events[0].end_time > 1.0


def _write_synthetic_wav(path: Path) -> None:
    sample_rate = 16000
    samples = []
    for index in range(sample_rate * 3):
        seconds = index / sample_rate
        if 1.0 <= seconds <= 1.45:
            value = int(math.sin(2 * math.pi * 880 * seconds) * 18000)
        else:
            value = int(math.sin(2 * math.pi * 220 * seconds) * 500)
        samples.append(value)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
