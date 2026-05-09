"""Audio backend registry."""

from cc_suggester.audio.backends.base import AudioBackend
from cc_suggester.audio.backends.dsp import DspAudioBackend
from cc_suggester.audio.backends.mock import MockAudioBackend
from cc_suggester.audio.backends.unavailable import UnavailableAudioBackend
from cc_suggester.audio.backends.yamnet import YamnetAudioBackend


def get_audio_backend(name: str) -> AudioBackend:
    """Return an audio backend by name."""

    normalized = name.lower().strip()
    if normalized in {"mock", "demo"}:
        return MockAudioBackend()
    if normalized in {"dsp", "energy"}:
        return DspAudioBackend()
    if normalized == "yamnet":
        return YamnetAudioBackend()
    if normalized == "panns":
        return UnavailableAudioBackend("panns", "Install PyTorch PANNs dependencies and add checkpoint loading.")
    if normalized == "ast":
        return UnavailableAudioBackend("ast", "Install AST dependencies and add an AudioSet checkpoint.")
    if normalized == "beats":
        return UnavailableAudioBackend("beats", "Install BEATs dependencies and add model checkpoint loading.")
    if normalized == "clap":
        return UnavailableAudioBackend("clap", "Install CLAP dependencies for open-vocabulary matching.")
    raise ValueError(
        f"Unknown audio backend '{name}'. Available: mock, dsp, yamnet, panns, ast, beats, clap."
    )
