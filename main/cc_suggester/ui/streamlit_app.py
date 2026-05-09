"""Streamlit editor review UI.

Run with:
    streamlit run cc_suggester/ui/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from cc_suggester.core.config import SUPPORTED_DEVICES, SUPPORTED_LANGUAGES, PipelineConfig
from cc_suggester.core.errors import CCSuggesterError
from cc_suggester.core.pipeline import analyze_video
from cc_suggester.output.review_export import build_review_export


def main() -> None:
    st.set_page_config(
        page_title="Intelligent CC Suggestion Tool",
        page_icon="CC",
        layout="wide",
    )
    st.title("Intelligent Closed Caption Suggestion Tool")
    st.caption("Generate and review meaningful non-speech CC suggestions.")

    with st.sidebar:
        st.header("Pipeline")
        uploaded = st.file_uploader("Video file", type=["mp4", "mkv", "mov", "avi", "webm", "wav"])
        language = st.selectbox("Language", SUPPORTED_LANGUAGES, index=0)
        device = st.selectbox("Device", SUPPORTED_DEVICES, index=0)
        audio_backend = st.selectbox("Audio backend", ["mock", "dsp", "yamnet"], index=0)
        vision_backend = st.selectbox("Vision backend", ["mock", "opencv", "mediapipe"], index=0)
        decision_threshold = st.slider("Decision threshold", 0.0, 1.0, 0.65, 0.01)
        review_threshold = st.slider("Review threshold", 0.0, 1.0, 0.50, 0.01)
        allow_demo_input = st.checkbox("Allow demo/non-video input", value=audio_backend == "mock")
        run = st.button("Start Caption", type="primary", use_container_width=True)

    if uploaded is None:
        st.info("Upload a video to begin. Use mock backends for a fast demo, or DSP/OpenCV for real local processing.")
        return

    input_path = _save_upload(uploaded)
    left, right = st.columns([1.5, 1.0], gap="large")
    with left:
        st.subheader("Video Preview")
        if input_path.suffix.lower() != ".wav":
            st.video(str(input_path))
        else:
            st.audio(str(input_path))

    if run:
        config = PipelineConfig(
            language=language,
            device=device,
            audio_backend=audio_backend,
            vision_backend=vision_backend,
            output_dir=Path("outputs"),
            decision_threshold=decision_threshold,
            review_threshold=review_threshold,
            allow_demo_input=allow_demo_input,
        )
        try:
            with st.spinner("Analyzing audio and visual reaction signals..."):
                st.session_state["result"] = analyze_video(input_path, config)
        except CCSuggesterError as exc:
            st.error(exc.message)
            if exc.suggestions:
                st.markdown("**Suggestions**")
                for suggestion in exc.suggestions:
                    st.write(f"- {suggestion}")
            if exc.details:
                with st.expander("Debug details"):
                    st.json(exc.details)
            return

    result = st.session_state.get("result")
    if not result:
        return

    with right:
        st.subheader("Run Summary")
        accepted = [item for item in result.suggestions if item.accepted]
        review = [item for item in result.suggestions if item.requires_review]
        rejected = [item for item in result.suggestions if not item.accepted and not item.requires_review]
        st.metric("Detected events", len(result.audio_events))
        st.metric("Accepted", len(accepted))
        st.metric("Needs review", len(review))
        st.metric("Rejected", len(rejected))
        st.write(f"Device used: `{result.diagnostics.actual_device}`")
        if result.diagnostics.warnings:
            with st.expander("Diagnostics warnings"):
                for warning in result.diagnostics.warnings:
                    st.warning(warning)

    st.subheader("Review Suggestions")
    rows = []
    for index, suggestion in enumerate(result.suggestions, start=1):
        status = "accepted" if suggestion.accepted else "review" if suggestion.requires_review else "rejected"
        row_key = f"{Path(result.input_path).stem}-{index}-{suggestion.event_id}-{suggestion.start_time:.3f}"
        with st.expander(
            f"{index}. {suggestion.caption_text} | {suggestion.start_time:.2f}s-{suggestion.end_time:.2f}s | {status}",
            expanded=index == 1,
        ):
            edited = st.text_input(
                "Caption text",
                value=suggestion.caption_text,
                key=f"caption-{row_key}",
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Audio", f"{suggestion.audio_confidence:.2f}")
            c2.metric("Reaction", f"{suggestion.reaction_confidence:.2f}")
            c3.metric("Decision", f"{suggestion.decision_score:.2f}")
            st.write(suggestion.reason)
            status_choice = st.radio(
                "Editor decision",
                ["accepted", "review", "rejected"],
                index=["accepted", "review", "rejected"].index(status),
                horizontal=True,
                key=f"status-{row_key}",
            )
            rows.append(
                {
                    "index": index,
                    "event_id": suggestion.event_id,
                    "start": suggestion.start_time,
                    "end": suggestion.end_time,
                    "caption": edited,
                    "status": status_choice,
                    "audio": suggestion.audio_confidence,
                    "reaction": suggestion.reaction_confidence,
                    "decision": suggestion.decision_score,
                    "reason": suggestion.reason,
                }
            )

    st.subheader("Exports")
    st.dataframe(rows, use_container_width=True)
    export_language = result.suggestions[0].language if result.suggestions else language
    review_export = build_review_export(rows, export_language)
    reviewed_accepted = sum(1 for item in review_export.suggestions if item.accepted)
    reviewed_review = sum(1 for item in review_export.suggestions if item.requires_review)
    reviewed_rejected = len(review_export.suggestions) - reviewed_accepted - reviewed_review
    srt_name = f"{Path(result.input_path).stem}.reviewed.{export_language}.srt"
    csv_name = f"{Path(result.input_path).stem}.reviewed.events.csv"
    json_name = f"{Path(result.input_path).stem}.reviewed.session.json"

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviewed accepted", reviewed_accepted)
    c2.metric("Still needs review", reviewed_review)
    c3.metric("Rejected", reviewed_rejected)

    export_left, export_middle, export_right = st.columns(3)
    export_left.download_button(
        label="Download Reviewed SRT",
        data=review_export.srt_text.encode("utf-8"),
        file_name=srt_name,
        mime="application/x-subrip",
        type="primary",
        use_container_width=True,
    )
    export_middle.download_button(
        label="Download Reviewed CSV",
        data=review_export.csv_text.encode("utf-8"),
        file_name=csv_name,
        mime="text/csv",
        use_container_width=True,
    )
    export_right.download_button(
        label="Download Review Session",
        data=review_export.json_text.encode("utf-8"),
        file_name=json_name,
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Raw pipeline exports"):
        for name, path in result.files.items():
            if path.exists():
                st.download_button(
                    label=f"Download Original {name.upper()}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    use_container_width=False,
                )


def _save_upload(uploaded) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "cc_suggester_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


if __name__ == "__main__":
    main()
