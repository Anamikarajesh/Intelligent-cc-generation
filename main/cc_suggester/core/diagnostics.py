"""Environment and device diagnostics."""

from __future__ import annotations

import platform
import shutil
from typing import Any

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.errors import DeviceUnavailableError
from cc_suggester.core.types import DiagnosticsReport


def _torch_status() -> tuple[bool, bool, str | None]:
    try:
        import torch  # type: ignore
    except Exception:
        return False, False, None

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:
            gpu_name = "CUDA device"
    return True, cuda_available, gpu_name


def run_diagnostics(config: PipelineConfig) -> DiagnosticsReport:
    """Collect environment details and resolve the actual processing device."""

    torch_available, cuda_available, gpu_name = _torch_status()
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    warnings: list[str] = []

    if ffmpeg_path is None:
        warnings.append("ffmpeg was not found; real video/audio extraction will fail.")
    if ffprobe_path is None:
        warnings.append("ffprobe was not found; metadata inspection will be limited.")

    actual_device = "cpu"
    fallback_reason = None
    if config.device == "cuda":
        if not cuda_available:
            details: dict[str, Any] = {
                "torch_available": torch_available,
                "cuda_available": cuda_available,
                "gpu_name": gpu_name,
                "ffmpeg_path": ffmpeg_path,
            }
            raise DeviceUnavailableError(
                message="CUDA was requested, but no usable GPU was detected.",
                code="cuda_unavailable",
                suggestions=[
                    "Retry with --device cpu.",
                    "Run ccs doctor to inspect the environment.",
                    "Install a CUDA-compatible PyTorch build if GPU acceleration is required.",
                ],
                details=details,
            )
        actual_device = "cuda"
    elif config.device == "auto" and cuda_available:
        actual_device = "cuda"
    elif config.device == "auto":
        fallback_reason = "CUDA was not detected; using CPU."

    return DiagnosticsReport(
        python_version=platform.python_version(),
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        selected_device=config.device,
        actual_device=actual_device,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        torch_available=torch_available,
        fallback_reason=fallback_reason,
        warnings=warnings,
    )
