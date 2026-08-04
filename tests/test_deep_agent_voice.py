from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

import opentulpa.deep_agent.voice as voice_module
from opentulpa.deep_agent.voice import (
    DEFAULT_TRANSCRIPTION_MODEL,
    OpenRouterAudioTranscriber,
    convert_audio_to_mp3,
)


def test_convert_audio_to_mp3_uses_bounded_trusted_ffmpeg_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, stdout=b"mp3-bytes", stderr=b"")

    monkeypatch.setattr(voice_module.subprocess, "run", run)

    result = convert_audio_to_mp3(b"telegram-ogg", max_output_bytes=100)

    assert result == b"mp3-bytes"
    assert captured["argv"][0] == "ffmpeg"
    assert captured["input"] == b"telegram-ogg"
    assert captured["timeout"] == 90
    assert "pipe:0" in captured["argv"]
    assert "pipe:1" in captured["argv"]
    size_index = captured["argv"].index("-fs")
    assert captured["argv"][size_index + 1] == "101"


@pytest.mark.asyncio
async def test_openrouter_transcriber_sends_mp3_to_gemini_and_returns_text() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello from the voice note."}}]},
        )

    def converter(raw_bytes: bytes, *, max_output_bytes: int) -> bytes:
        assert raw_bytes == b"telegram-ogg"
        assert max_output_bytes == 1_000
        return b"normalized-mp3"

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transcriber = OpenRouterAudioTranscriber(
        api_key="private-openrouter-key",
        base_url="https://openrouter.ai/api/v1",
        max_mp3_bytes=1_000,
        client=http,
        converter=converter,
    )

    result = await transcriber.transcribe(b"telegram-ogg")

    assert result.mp3_bytes == b"normalized-mp3"
    assert result.transcript == "Hello from the voice note."
    assert requests[0].url.path == "/api/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer private-openrouter-key"
    payload = __import__("json").loads(requests[0].content)
    assert payload["model"] == DEFAULT_TRANSCRIPTION_MODEL
    audio = payload["messages"][0]["content"][1]
    assert audio == {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(b"normalized-mp3").decode("ascii"),
            "format": "mp3",
        },
    }
    assert "Return only the transcript" in payload["messages"][0]["content"][0]["text"]
    await http.aclose()


def test_runtime_image_installs_ffmpeg_for_voice_normalization() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert "curl ffmpeg git" in dockerfile
