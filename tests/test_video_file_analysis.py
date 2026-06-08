from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.agent import file_analysis


class _DummyModel:
    def __init__(self, final_text: str = "Final synthesized video report.") -> None:
        self.final_text = final_text
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(content=self.final_text)


class _DummyRuntime:
    def __init__(self, final_text: str = "Final synthesized video report.") -> None:
        self._model = _DummyModel(final_text=final_text)


def test_build_30s_segments_caps_and_aligns() -> None:
    segments = file_analysis._build_30s_segments(duration_seconds=95, max_segments=12)
    assert segments == [(0, 30), (30, 60), (60, 90), (90, 95)]


@pytest.mark.asyncio
async def test_summarize_uploaded_blob_video_uses_segment_pipeline(monkeypatch) -> None:
    runtime = _DummyRuntime(final_text="Synthesized scenes + music summary.")
    seen_segments: list[tuple[int, int]] = []

    async def _fake_estimate(*args: Any, **kwargs: Any) -> int:  # type: ignore[no-untyped-def]
        return 95

    async def _fake_segment(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-untyped-def]
        start = int(kwargs["start_seconds"])
        end = int(kwargs["end_seconds"])
        seen_segments.append((start, end))
        return f"segment {start}-{end}"

    monkeypatch.setattr(file_analysis, "_estimate_video_duration_seconds", _fake_estimate)
    monkeypatch.setattr(file_analysis, "_analyze_video_segment", _fake_segment)

    summary = await file_analysis.summarize_uploaded_blob(
        runtime,
        filename="clip.mp4",
        mime_type="video/mp4",
        kind="video",
        raw_bytes=b"video-bytes",
        caption="concert night",
        question="describe scenes and music",
    )

    assert "Synthesized scenes + music summary." in summary
    assert seen_segments == [(0, 30), (30, 60), (60, 90), (90, 95)]


@pytest.mark.asyncio
async def test_summarize_uploaded_blob_video_large_file_returns_guard_message() -> None:
    runtime = _DummyRuntime()
    too_large = b"x" * (file_analysis._VIDEO_INLINE_MAX_BYTES + 1)

    summary = await file_analysis.summarize_uploaded_blob(
        runtime,
        filename="large.mov",
        mime_type="video/quicktime",
        kind="video",
        raw_bytes=too_large,
    )

    assert "too large for inline video analysis" in summary.lower()


@pytest.mark.asyncio
async def test_video_synthesis_prompt_requests_transcript_and_concise_happenings() -> None:
    runtime = _DummyRuntime(final_text="ok")
    out = await file_analysis._synthesize_video_segments(
        runtime,
        filename="clip.mp4",
        mime_type="video/mp4",
        caption="city night drive",
        question="what vibe does this give?",
        segment_notes=["00:00-00:30 ..."],
    )

    assert out == "ok"
    # First message is SystemMessage, second message carries the user prompt text.
    human_text = str(getattr(runtime._model.calls[-1][1], "content", ""))
    assert "1) Transcript" in human_text
    assert "2) What happens" in human_text
    assert "Keep non-transcript description concise" in human_text
