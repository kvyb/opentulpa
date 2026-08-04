from __future__ import annotations

import asyncio
import base64
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import opentulpa.deep_agent.voice as voice_module
from opentulpa.deep_agent.voice import (
    DEFAULT_TRANSCRIPTION_MODEL,
    AudioTranscription,
    AudioTranscriptionError,
    OpenRouterAudioTranscriber,
    VoiceAttachmentProcessor,
    build_openrouter_audio_transcriber,
    convert_audio_to_mp3,
)


class _Resolver:
    def __init__(self, record: dict[str, Any], raw_bytes: bytes | None = b"source-audio") -> None:
        self.record = record
        self.raw_bytes = raw_bytes

    def get_file(self, tenant_id: str, file_id: str) -> dict[str, Any] | None:
        assert tenant_id == "tenant-1"
        assert file_id == "voice-1"
        return self.record

    def read_file_bytes(self, tenant_id: str, file_id: str) -> bytes | None:
        assert tenant_id == "tenant-1"
        assert file_id == "voice-1"
        return self.raw_bytes


class _Transcriber:
    def __init__(self, *, delay: float = 0) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.inputs: list[bytes] = []

    async def transcribe(self, raw_bytes: bytes) -> AudioTranscription:
        self.inputs.append(raw_bytes)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return AudioTranscription(mp3_bytes=b"mp3", transcript="hello")

    async def aclose(self) -> None:
        return None


def _context() -> Any:
    return SimpleNamespace(tenant_id="tenant-1", run_kind="owner")


def _record(*, kind: str, mime_type: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "mime_type": mime_type,
        "original_filename": "voice-note.ogg",
    }


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
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.DEVNULL
    assert "capture_output" not in captured
    assert "pipe:0" in captured["argv"]
    assert "pipe:1" in captured["argv"]
    size_index = captured["argv"].index("-fs")
    assert captured["argv"][size_index + 1] == "101"


def test_convert_audio_to_mp3_rejects_failed_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=None)

    monkeypatch.setattr(voice_module.subprocess, "run", run)

    with pytest.raises(AudioTranscriptionError, match="conversion failed"):
        convert_audio_to_mp3(b"corrupt-audio", max_output_bytes=100)


@pytest.mark.parametrize(
    ("kind", "mime_type"),
    [
        ("voice", "application/octet-stream"),
        ("audio", "application/octet-stream"),
        ("document", "audio/wav"),
    ],
)
@pytest.mark.asyncio
async def test_voice_processor_accepts_kind_or_audio_mime_type(
    kind: str,
    mime_type: str,
) -> None:
    uploads: list[list[tuple[str, bytes]]] = []

    def upload(_context: Any, files: list[tuple[str, bytes]]) -> tuple[set[str], set[str]]:
        uploads.append(files)
        return {path for path, _content in files}, set()

    transcriber = _Transcriber()
    processor = VoiceAttachmentProcessor(
        attachment_resolver=_Resolver(_record(kind=kind, mime_type=mime_type)),
        transcriber=transcriber,
        workspace_uploader=upload,
    )

    notes = await processor.process(
        context=_context(),
        file_ids=("voice-1",),
        run_id="run-1",
    )

    assert transcriber.inputs == [b"source-audio"]
    assert uploads[0][0][0].endswith("/voice-note.mp3")
    assert "Transcript of the received audio:\n<audio-transcript>\nhello" in notes[0]


@pytest.mark.asyncio
async def test_voice_processor_caps_concurrent_work() -> None:
    def upload(_context: Any, files: list[tuple[str, bytes]]) -> tuple[set[str], set[str]]:
        return {path for path, _content in files}, set()

    transcriber = _Transcriber(delay=0.02)
    processor = VoiceAttachmentProcessor(
        attachment_resolver=_Resolver(_record(kind="voice", mime_type="audio/ogg")),
        transcriber=transcriber,
        workspace_uploader=upload,
        max_concurrency=2,
    )

    await asyncio.gather(
        *(
            processor.process(
                context=_context(),
                file_ids=("voice-1",),
                run_id=f"run-{index}",
            )
            for index in range(6)
        )
    )

    assert transcriber.peak == 2


@pytest.mark.asyncio
async def test_voice_processor_degrades_when_transcription_fails() -> None:
    uploads: list[list[tuple[str, bytes]]] = []

    class FailingTranscriber(_Transcriber):
        async def transcribe(self, raw_bytes: bytes) -> AudioTranscription:
            del raw_bytes
            raise AudioTranscriptionError("provider unavailable")

    def upload(_context: Any, files: list[tuple[str, bytes]]) -> tuple[set[str], set[str]]:
        uploads.append(files)
        return set(), {path for path, _content in files}

    processor = VoiceAttachmentProcessor(
        attachment_resolver=_Resolver(_record(kind="voice", mime_type="audio/ogg")),
        transcriber=FailingTranscriber(),
        workspace_uploader=upload,
    )

    notes = await processor.process(
        context=_context(),
        file_ids=("voice-1",),
        run_id="run-1",
    )

    assert notes == ("Received audio file voice-1; transcription was unavailable.",)
    assert uploads == []


@pytest.mark.asyncio
async def test_voice_processor_degrades_when_source_bytes_are_unavailable() -> None:
    uploads: list[list[tuple[str, bytes]]] = []

    def upload(_context: Any, files: list[tuple[str, bytes]]) -> tuple[set[str], set[str]]:
        uploads.append(files)
        return set(), {path for path, _content in files}

    processor = VoiceAttachmentProcessor(
        attachment_resolver=_Resolver(
            _record(kind="voice", mime_type="audio/ogg"),
            raw_bytes=None,
        ),
        transcriber=_Transcriber(),
        workspace_uploader=upload,
    )

    notes = await processor.process(
        context=_context(),
        file_ids=("voice-1",),
        run_id="run-1",
    )

    assert notes == ("Received audio file voice-1; transcription was unavailable.",)
    assert uploads == []


@pytest.mark.asyncio
async def test_voice_processor_keeps_transcript_when_workspace_upload_fails() -> None:
    def upload(_context: Any, files: list[tuple[str, bytes]]) -> tuple[set[str], set[str]]:
        return set(), {path for path, _content in files}

    processor = VoiceAttachmentProcessor(
        attachment_resolver=_Resolver(_record(kind="voice", mime_type="audio/ogg")),
        transcriber=_Transcriber(),
        workspace_uploader=upload,
    )

    notes = await processor.process(
        context=_context(),
        file_ids=("voice-1",),
        run_id="run-1",
    )

    assert "its MP3 could not be stored in sandbox" in notes[0]
    assert "<audio-transcript>\nhello\n</audio-transcript>" in notes[0]
    assert "Received audio file:" not in notes[0]


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


@pytest.mark.parametrize(
    ("status_code", "response_payload"),
    [
        (500, {"error": {"message": "unavailable"}}),
        (200, {"choices": []}),
        (200, {"choices": [{"message": {"content": ""}}]}),
    ],
)
@pytest.mark.asyncio
async def test_openrouter_transcriber_rejects_provider_failures(
    status_code: int,
    response_payload: dict[str, Any],
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=response_payload)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transcriber = OpenRouterAudioTranscriber(
        api_key="private-openrouter-key",
        base_url="https://openrouter.ai/api/v1",
        max_mp3_bytes=1_000,
        client=http,
        converter=lambda _raw, *, max_output_bytes: b"mp3",
    )

    with pytest.raises(AudioTranscriptionError):
        await transcriber.transcribe(b"telegram-ogg")

    await http.aclose()


def test_openrouter_transcriber_is_disabled_for_other_provider(caplog: pytest.LogCaptureFixture) -> None:
    transcriber = build_openrouter_audio_transcriber(
        api_key="other-provider-key",
        base_url="https://api.example.com/v1",
        max_mp3_bytes=1_000,
    )

    assert transcriber is None
    assert "configured inference endpoint is not OpenRouter" in caplog.text


def test_runtime_image_installs_ffmpeg_for_voice_normalization() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert "curl ffmpeg git" in dockerfile
