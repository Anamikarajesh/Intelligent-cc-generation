"""Small WAV loading helpers shared by audio backends."""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class WavPcm:
    """Decoded mono PCM WAV samples."""

    sample_rate: int
    sample_width: int
    samples: list[int]


def load_wav_mono_pcm(path: Path) -> WavPcm:
    """Load a WAV file as mono integer PCM samples."""

    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())
    samples = _decode_pcm(frames, sample_width, channels)
    return WavPcm(sample_rate=sample_rate, sample_width=sample_width, samples=samples)


def load_wav_mono_float32(path: Path, target_sample_rate: int = 16000) -> list[float]:
    """Load WAV samples normalized to [-1, 1] and resampled if required."""

    wav = load_wav_mono_pcm(path)
    max_amplitude = float(2 ** (8 * wav.sample_width - 1))
    floats = [max(-1.0, min(1.0, sample / max_amplitude)) for sample in wav.samples]
    if wav.sample_rate != target_sample_rate:
        floats = _resample_linear(floats, wav.sample_rate, target_sample_rate)
    return floats


def _decode_pcm(frames: bytes, sample_width: int, channels: int) -> list[int]:
    if sample_width == 1:
        values = [byte - 128 for byte in frames]
    elif sample_width == 2:
        count = len(frames) // 2
        values = list(struct.unpack(f"<{count}h", frames[: count * 2]))
    elif sample_width == 4:
        count = len(frames) // 4
        values = list(struct.unpack(f"<{count}i", frames[: count * 4]))
    elif sample_width == 3:
        values = [_decode_24bit(frames[index : index + 3]) for index in range(0, len(frames) - 2, 3)]
    else:
        return []

    if channels <= 1:
        return values

    mono: list[int] = []
    for index in range(0, len(values) - channels + 1, channels):
        mono.append(int(sum(values[index : index + channels]) / channels))
    return mono


def _decode_24bit(chunk: bytes) -> int:
    padded = chunk + (b"\xff" if chunk[2] & 0x80 else b"\x00")
    return struct.unpack("<i", padded)[0]


def _resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if not samples or source_rate <= 0 or target_rate <= 0:
        return samples
    if source_rate == target_rate:
        return samples

    target_len = max(1, int(len(samples) * target_rate / source_rate))
    if target_len == 1:
        return [samples[0]]

    scale = (len(samples) - 1) / (target_len - 1)
    output: list[float] = []
    for index in range(target_len):
        source_pos = index * scale
        left = int(source_pos)
        right = min(left + 1, len(samples) - 1)
        fraction = source_pos - left
        output.append(samples[left] * (1.0 - fraction) + samples[right] * fraction)
    return output
