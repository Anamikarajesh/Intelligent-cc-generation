"""Generate tiny deterministic sample media for integration testing.

The script uses Python's standard library to create a synthetic WAV file and
ffmpeg to combine it with a generated test video pattern when ffmpeg is
available. If ffmpeg is missing but OpenCV is installed, it writes a video-only
MP4 plus a sidecar WAV file. The sidecar path can be passed to the CLI with
``--audio-path``.

Usage:
    python scripts/generate_sample_video.py
    python scripts/generate_sample_video.py --out tests/fixtures/sample.mp4
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a sample MP4 with audible non-speech events.")
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/sample_classroom.mp4"))
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wav_path = args.out.with_suffix(".wav")
    _write_synthetic_wav(wav_path, duration=args.duration)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        _write_video_with_ffmpeg(ffmpeg, wav_path, args.out, duration=args.duration)
        print(f"Generated embedded-audio sample video: {args.out}")
    else:
        _write_video_with_opencv(args.out, duration=args.duration)
        print(f"Generated video-only sample: {args.out}")
        print("ffmpeg was not found, so use the sidecar WAV with --audio-path.")
    print(f"Generated source audio: {wav_path}")
    return 0


def _write_synthetic_wav(path: Path, *, duration: float, sample_rate: int = 16000) -> None:
    samples = []
    total_samples = int(sample_rate * duration)
    for index in range(total_samples):
        seconds = index / sample_rate
        base = math.sin(2 * math.pi * 180 * seconds) * 450
        event_one = math.sin(2 * math.pi * 880 * seconds) * 19000 if 1.15 <= seconds <= 1.55 else 0
        event_two = math.sin(2 * math.pi * 1240 * seconds) * 17000 if 3.25 <= seconds <= 3.70 else 0
        value = int(max(-32000, min(32000, base + event_one + event_two)))
        samples.append(value)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


def _write_video_with_ffmpeg(ffmpeg: str, wav_path: Path, out_path: Path, *, duration: float) -> None:
    command = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=640x360:rate=25:duration={duration}",
        "-i",
        str(wav_path),
        "-shortest",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{completed.stderr}")


def _write_video_with_opencv(out_path: Path, *, duration: float) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Neither ffmpeg nor OpenCV video writing is available. "
            "Install ffmpeg, or install opencv-python for sidecar fixture generation."
        ) from exc

    width, height, fps = 640, 360, 25
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit("OpenCV could not open a VideoWriter for the requested output path.")

    total_frames = int(duration * fps)
    for frame_index in range(total_frames):
        t = frame_index / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (235, 238, 230)
        cv2.rectangle(frame, (0, 0), (width, 84), (92, 130, 178), -1)
        cv2.putText(frame, "Demo classroom scene", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 120), (590, 300), (245, 245, 245), -1)
        cv2.rectangle(frame, (72, 145), (210, 278), (88, 120, 150), -1)
        cv2.rectangle(frame, (252, 145), (390, 278), (90, 150, 120), -1)
        cv2.rectangle(frame, (432, 145), (570, 278), (150, 120, 90), -1)
        if 1.15 <= t <= 1.55 or 3.25 <= t <= 3.70:
            cv2.circle(frame, (320, 212), 44, (0, 215, 255), -1)
            cv2.putText(frame, "SOUND EVENT", (230, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 45, 60), 2)
        else:
            cv2.circle(frame, (320, 212), 28, (190, 205, 220), -1)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
