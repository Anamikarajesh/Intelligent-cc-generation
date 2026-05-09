"""CPU-friendly DSP audio backend.

This backend performs simple energy/onset style detection from a mono WAV file.
It is not a semantic classifier like YAMNet or PANNs, but it is useful as a real
offline baseline and as a candidate-region generator.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from cc_suggester.audio.backends.base import AudioBackend
from cc_suggester.audio.extractor import extract_audio
from cc_suggester.audio.wav import load_wav_mono_pcm
from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.types import AudioEventCandidate, VideoMetadata


@dataclass(slots=True)
class EnergyWindow:
    """RMS summary for a short audio window."""

    start: float
    end: float
    rms_norm: float


class DspAudioBackend(AudioBackend):
    """Detect non-speech candidate regions using RMS energy windows."""

    name = "dsp"
    requires_audio_file = True
    requires_valid_media = True

    def detect(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        config: PipelineConfig,
    ) -> list[AudioEventCandidate]:
        audio_path = self._audio_path_for(video_path, config)
        windows = _read_energy_windows(audio_path)
        if not windows:
            return []

        values = [window.rms_norm for window in windows]
        median = statistics.median(values)
        peak = max(values)
        adaptive_threshold = max(0.015, median * 3.0, peak * 0.32)

        active = [window for window in windows if window.rms_norm >= adaptive_threshold]
        groups = _group_windows(active, max_gap=0.35)
        events: list[AudioEventCandidate] = []
        for index, group in enumerate(groups, start=1):
            start = group[0].start
            end = group[-1].end
            duration = end - start
            peak_norm = max(window.rms_norm for window in group)
            confidence = _confidence(peak_norm, adaptive_threshold)
            if confidence < config.audio_threshold:
                continue
            event_id = _event_id_for(duration, peak_norm)
            events.append(
                AudioEventCandidate(
                    event_id=event_id,
                    label=event_id.replace("_", " ").title(),
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    audio_confidence=confidence,
                    audio_backend=self.name,
                    raw_class_name="RMS energy candidate",
                    debug_info={
                        "audio_path": str(audio_path),
                        "window_index": index,
                        "rms_peak": round(peak_norm, 6),
                        "rms_median": round(median, 6),
                        "adaptive_threshold": round(adaptive_threshold, 6),
                        "duration": round(duration, 3),
                    },
                )
            )
        return events

    def _audio_path_for(self, video_path: Path, config: PipelineConfig) -> Path:
        if config.sidecar_audio_path is not None:
            return Path(config.sidecar_audio_path)
        if video_path.suffix.lower() == ".wav":
            return video_path
        run_dir = config.run_dir or config.output_dir / video_path.stem
        return extract_audio(video_path, run_dir / "artifacts" / "audio.wav")


def _read_energy_windows(
    audio_path: Path,
    *,
    window_seconds: float = 0.50,
    hop_seconds: float = 0.25,
) -> list[EnergyWindow]:
    wav = load_wav_mono_pcm(audio_path)
    samples = wav.samples
    if not samples:
        return []

    window_samples = max(1, int(wav.sample_rate * window_seconds))
    hop_samples = max(1, int(wav.sample_rate * hop_seconds))
    max_amplitude = float(2 ** (8 * wav.sample_width - 1))

    windows: list[EnergyWindow] = []
    for start_index in range(0, max(0, len(samples) - window_samples + 1), hop_samples):
        chunk = samples[start_index : start_index + window_samples]
        if len(chunk) < window_samples:
            break
        start = start_index / wav.sample_rate
        end = start + window_seconds
        rms = math.sqrt(sum(sample * sample for sample in chunk) / len(chunk))
        windows.append(EnergyWindow(start=start, end=end, rms_norm=rms / max_amplitude))
    return windows


def _group_windows(windows: list[EnergyWindow], max_gap: float) -> list[list[EnergyWindow]]:
    if not windows:
        return []
    groups: list[list[EnergyWindow]] = [[windows[0]]]
    for window in windows[1:]:
        previous = groups[-1][-1]
        if window.start - previous.end <= max_gap:
            groups[-1].append(window)
        else:
            groups.append([window])
    return groups


def _confidence(peak_norm: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    ratio = peak_norm / threshold
    return round(max(0.0, min(0.99, 0.35 + ratio * 0.28)), 3)


def _event_id_for(duration: float, peak_norm: float) -> str:
    if duration <= 0.85 and peak_norm >= 0.08:
        return "impact_sound"
    if duration >= 3.0:
        return "sustained_sound"
    return "loud_sound"
