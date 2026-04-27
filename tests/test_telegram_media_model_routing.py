from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.agent import file_analysis


class _RecordingModel:
    def __init__(self, label: str) -> None:
        self.label = label


class _RoutingRuntime:
    def __init__(self) -> None:
        self._model = _RecordingModel("main")
        self._telegram_media_model = _RecordingModel("telegram-media")
        self.model_name = "deepseek/deepseek-v4-pro"
        self._telegram_media_model_name = "google/gemini-3-flash-preview"
        self.calls: list[dict[str, Any]] = []

    async def ainvoke_model(
        self,
        model: Any,
        messages: list[Any],
        *,
        model_name: str | None = None,
        stable_prefix_count: int = 0,
        call_context: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append(
            {
                "model": getattr(model, "label", None),
                "model_name": model_name,
                "messages": messages,
                "stable_prefix_count": stable_prefix_count,
                "call_context": call_context,
            }
        )
        return SimpleNamespace(content="Media summary from Gemini.")


@pytest.mark.asyncio
async def test_summarize_uploaded_blob_image_uses_telegram_media_model() -> None:
    runtime = _RoutingRuntime()

    summary = await file_analysis.summarize_uploaded_blob(
        runtime,
        filename="photo.jpg",
        mime_type="image/jpeg",
        kind="photo",
        raw_bytes=b"\xff\xd8\xff" + (b"x" * 64),
        caption="receipt photo",
    )

    assert summary == "Media summary from Gemini."
    assert runtime.calls
    assert runtime.calls[-1]["model"] == "telegram-media"
    assert runtime.calls[-1]["model_name"] == "google/gemini-3-flash-preview"


@pytest.mark.asyncio
async def test_summarize_uploaded_blob_audio_uses_transcript_path(monkeypatch) -> None:
    runtime = _RoutingRuntime()

    async def _fake_transcribe(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-untyped-def]
        _ = (args, kwargs)
        return "Booked for tomorrow at 4pm."

    monkeypatch.setattr(file_analysis, "transcribe_audio_blob", _fake_transcribe)

    summary = await file_analysis.summarize_uploaded_blob(
        runtime,
        filename="note.mp3",
        mime_type="audio/mpeg",
        kind="audio",
        raw_bytes=b"audio-bytes",
    )

    assert "Transcript: Booked for tomorrow at 4pm." in summary


@pytest.mark.asyncio
async def test_summarize_uploaded_blob_video_note_uses_video_path(monkeypatch) -> None:
    runtime = _RoutingRuntime()

    async def _fake_video_summary(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-untyped-def]
        _ = (args, kwargs)
        return "Video note summary from Gemini."

    monkeypatch.setattr(file_analysis, "_summarize_video_blob", _fake_video_summary)

    summary = await file_analysis.summarize_uploaded_blob(
        runtime,
        filename="circle.mp4",
        mime_type="video/mp4",
        kind="video_note",
        raw_bytes=b"video-bytes",
    )

    assert summary == "Video note summary from Gemini."


@pytest.mark.asyncio
async def test_transcribe_audio_blob_uses_telegram_media_model_name(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "hello from audio",
                        }
                    }
                ]
            }

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            _ = (exc_type, exc, tb)
            return None

        async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, Any]) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse()

    runtime = SimpleNamespace(
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-v4-pro",
        _telegram_media_model_name="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(file_analysis.httpx, "AsyncClient", _FakeAsyncClient)

    transcript = await file_analysis.transcribe_audio_blob(
        runtime,
        filename="voice.ogg",
        mime_type="audio/ogg",
        kind="voice",
        raw_bytes=b"audio-bytes",
    )

    assert transcript == "hello from audio"
    assert captured["json"]["model"] == "google/gemini-3-flash-preview"
    instruction = captured["json"]["messages"][0]["content"][0]["text"]
    assert "Transcribe all spoken or clearly heard speech" in instruction
    assert "Return plain text transcript only" in instruction
