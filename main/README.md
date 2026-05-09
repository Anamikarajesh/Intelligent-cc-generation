# cc-suggester

Python implementation for the Intelligent Closed Caption Suggestion Tool.

This package generates meaningful non-speech closed caption suggestions from video. The current implementation is a runnable foundation: it proves the modular pipeline, CLI, diagnostics, decision engine, multilingual labels, and SRT/JSON/CSV export flow before heavy ML backends are added.

## Current Implementation Status

Implemented now:

- `cc_suggester.core`: pipeline orchestration, config, shared data models, diagnostics, media inspection, friendly errors
- `cc_suggester.audio`: audio backend interface, deterministic mock backend, DSP backend, event smoothing, ffmpeg extraction helper, advanced backend placeholders
- `cc_suggester.vision`: vision backend interface, deterministic mock backend, OpenCV backend, optional MediaPipe pose backend, frame-sampling and reaction helpers
- `cc_suggester.decision`: scoring rules, ambient penalties, multilingual caption glossary
- `cc_suggester.output`: SRT, JSON, CSV, and reviewed export helpers
- `cc_suggester.cli`: `analyze`, `audio`, `inspect`, `doctor`, `export`, `labels`, and `web` commands
- `cc_suggester.ui`: Streamlit editor review client with edited SRT/CSV/session downloads
- `tests`: tests for SRT output, label lookup, config/CLI behavior, DSP detection, and reviewed exports

Not implemented yet:

- Real YAMNet/PANNs/AST/BEATs semantic audio backend
- MediaPipe face-landmark/expression reaction scoring
- Advanced Streamlit timeline editing and persisted review sessions
- Real evaluation dataset and editor feedback loop
- Docker and VLC integration

The full roadmap is documented in [`../docs/implementation-plan.md`](../docs/implementation-plan.md).

## Setup

The current scaffold uses only the Python standard library for the core pipeline.

```bash
cd main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development tests:

```bash
pip install -r requirements-dev.txt
```

For the Web UI:

```bash
pip install -r requirements-ui.txt
```

For the OpenCV vision backend:

```bash
pip install -r requirements-vision.txt
```

`requirements-vision.txt` also includes MediaPipe for the optional pose-based reaction backend.

## CLI Usage

Run diagnostics:

```bash
python -m cc_suggester doctor
```

Inspect a video:

```bash
python -m cc_suggester inspect path/to/video.mp4
```

Run the current mock pipeline:

```bash
python -m cc_suggester analyze path/to/video.mp4 --lang hi --device auto --out outputs/
```

Run the CPU DSP audio baseline:

```bash
python -m cc_suggester analyze path/to/video.mp4 --audio-backend dsp --vision-backend mock --lang en
```

Run only audio detection:

```bash
python -m cc_suggester audio path/to/video.mp4 --audio-backend dsp --out outputs/
```

Run only visual reaction scoring from an audio report:

```bash
python -m cc_suggester vision path/to/video.mp4 outputs/video/audio_events.json --vision-backend opencv
```

Run the optional YAMNet backend after installing audio dependencies:

```bash
pip install -r requirements-audio.txt
python -m cc_suggester audio path/to/video.mp4 --audio-backend yamnet --out outputs/
```

For offline environments, point YAMNet to a local TensorFlow Hub model directory:

```bash
python -m cc_suggester audio path/to/video.mp4 \
  --audio-backend yamnet \
  --yamnet-model /path/to/local/yamnet
```

Export another language from an existing JSON report:

```bash
python -m cc_suggester export outputs/video/results.json --format srt --lang ml
```

Show Web UI guidance:

```bash
python -m cc_suggester web
```

List supported labels:

```bash
python -m cc_suggester labels
```

The installed package will expose the same CLI as `ccs`:

```bash
ccs analyze path/to/video.mp4 --lang hi --device auto
```

## Output Files

Each analysis run creates a directory under `outputs/`:

```text
outputs/
  video-name/
    captions.<lang>.srt
    results.json
    events.csv
    diagnostics.json
    config.json
```

`captions.<lang>.srt` contains only accepted captions. `results.json` and `events.csv` include accepted, rejected, and review-needed candidates for debugging and editor review.

The Streamlit UI can also export reviewed SRT, CSV, and JSON session content from the current editor choices. This means edited caption text and manual accept/reject/review decisions drive the downloaded files.

## Backend Strategy

Backends are intentionally pluggable.

Audio backends implement:

```text
detect(video_path, metadata, config) -> list[AudioEventCandidate]
```

Vision backends implement:

```text
analyze(video_path, metadata, audio_events, config) -> list[ReactionResult]
```

The DSP audio backend and OpenCV vision backend are available as local baselines. YAMNet is implemented as an optional TensorFlow Hub backend and requires `requirements-audio.txt`. MediaPipe is implemented as an optional pose-based reaction backend and requires `requirements-vision.txt`. Mock backends should remain available for tests and demos.

## Verification

Run syntax checks:

```bash
python -m compileall cc_suggester
```

Run tests:

```bash
python -m pytest tests
```

Run CLI smoke checks:

```bash
python -m cc_suggester doctor
python -m cc_suggester analize
python -m cc_suggester analyze README.md --lang hi --device auto --out outputs
python -m cc_suggester export outputs/README/results.json --format srt --lang ml --out outputs/README/captions.ml.srt
python -m cc_suggester labels
python -m cc_suggester vision tests/fixtures/sample_classroom.mp4 outputs/sample_classroom/audio_events.json --vision-backend opencv
```

The `analize` command is intentionally useful as a smoke check for friendly typo suggestions.

## Real Sample Video Fixture

Generate a tiny deterministic MP4 fixture for local integration testing:

```bash
python scripts/generate_sample_video.py
```

Then run:

```bash
python -m cc_suggester inspect tests/fixtures/sample_classroom.mp4
python -m cc_suggester analyze tests/fixtures/sample_classroom.mp4 --audio-backend dsp --vision-backend mock --lang hi
```

If `ffmpeg` is available, the MP4 includes embedded audio. If `ffmpeg` is unavailable but OpenCV is installed, the script writes a video-only MP4 plus a sidecar WAV file:

```bash
python -m cc_suggester analyze tests/fixtures/sample_classroom.mp4 \
  --audio-backend dsp \
  --vision-backend opencv \
  --audio-path tests/fixtures/sample_classroom.wav \
  --lang hi
```

## Immediate Next Sprint

1. Test YAMNet with an installed TensorFlow/TensorFlow Hub environment and a cached/local model.
2. Test MediaPipe in an environment with `requirements-vision.txt` installed and tune pose thresholds.
3. Add face-landmark/expression scoring to the MediaPipe backend.
4. Add more decision-rule and backend dependency tests.
5. Add timeline markers and persisted review sessions to the Streamlit editor.
6. Add evaluation scripts for editor feedback.

After that, add evaluation scripts and package the CPU pipeline with Docker.
