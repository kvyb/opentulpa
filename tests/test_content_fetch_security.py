from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import httpx
import pytest

from opentulpa.integrations.content_fetch import (
    ContentFetchError,
    ContentFetchService,
    Crawl4AIContentExtractor,
)


class _Resolver:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> Sequence[str]:
        self.calls.append((hostname, port))
        answer = self.answers[hostname]
        if answer and isinstance(answer[0], (list, tuple)):
            return cast(list[Sequence[str]], answer).pop(0)
        return cast(Sequence[str], answer)


class _SlowResolver:
    async def resolve(self, hostname: str, port: int) -> Sequence[str]:
        _ = hostname, port
        await asyncio.sleep(1)
        return ["93.184.216.34"]


class _ExpandingExtractor:
    def extract(
        self,
        *,
        body: bytes,
        content_type: str,
        charset: str,
        url: str,
    ) -> tuple[str, str | None]:
        del body, content_type, charset, url
        return "expanded" * 1_000, "Title"


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: Sequence[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


async def _fetch_error(service: ContentFetchService, url: str) -> ContentFetchError:
    with pytest.raises(ContentFetchError) as captured:
        await service.fetch(url)
    return captured.value


@pytest.mark.asyncio
async def test_fetch_pins_validated_ip_and_extracts_bounded_html_text() -> None:
    resolver = _Resolver({"example.com": ["93.184.216.34"]})
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=(
                b"<html><head><title>Example</title><style>hidden</style></head>"
                b"<body><h1>Hello</h1><script>secret()</script><p>World</p></body></html>"
            ),
        )

    service = ContentFetchService(
        resolver=resolver,
        transport=httpx.MockTransport(handler),
        max_bytes=2_000,
    )
    result = await service.fetch("https://example.com/research?q=one#fragment")

    assert result.url == "https://example.com/research?q=one"
    assert result.content_type == "text/html"
    assert result.charset == "utf-8"
    assert result.title == "Example"
    assert "Hello" in result.text
    assert "World" in result.text
    assert "hidden" not in result.text
    assert "secret" not in result.text
    assert result.redirects == 0
    assert resolver.calls == [("example.com", 443)]
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "example.com"
    assert requests[0].extensions["sni_hostname"] == "example.com"
    assert "authorization" not in requests[0].headers
    assert "cookie" not in requests[0].headers


def test_crawl4ai_adapter_transforms_already_fetched_html_offline() -> None:
    pytest.importorskip("crawl4ai")
    text, title = Crawl4AIContentExtractor().extract(
        body=b"<html><head><title>Docs</title></head><body><h1>Hello</h1><a href='/a'>A</a></body></html>",
        content_type="text/html",
        charset="utf-8",
        url="https://example.com/docs",
    )

    assert title == "Docs"
    assert "Hello" in text
    assert "https://example.com/a" in text


@pytest.mark.asyncio
async def test_extractor_output_is_bounded_after_safe_fetch() -> None:
    service = ContentFetchService(
        resolver=_Resolver({"example.com": ["93.184.216.34"]}),
        extractor=_ExpandingExtractor(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<p>small response</p>",
            )
        ),
        max_text_characters=100,
    )

    result = await service.fetch("https://example.com/")

    assert len(result.text) == 100


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("ftp://example.com/file", "invalid_scheme"),
        ("https://user:password@example.com/", "url_credentials"),
        ("http://@example.com/", "url_credentials"),
        ("https://example.com:8443/", "forbidden_port"),
        ("https://metadata.google.internal/latest", "forbidden_target"),
        ("http://localhost/", "forbidden_target"),
        ("http://127.0.0.1/", "forbidden_target"),
        ("http://169.254.169.254/latest", "forbidden_target"),
        ("http://[::1]/", "forbidden_target"),
        ("http://[fe80::1]/", "forbidden_target"),
        ("http://[::ffff:127.0.0.1]/", "forbidden_target"),
        ("http://[64:ff9b::7f00:1]/", "forbidden_target"),
    ],
)
async def test_rejects_unsafe_url_shapes_before_transport(url: str, code: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="unsafe")

    service = ContentFetchService(
        resolver=_Resolver({"example.com": ["93.184.216.34"]}),
        transport=httpx.MockTransport(handler),
    )
    error = await _fetch_error(service, url)
    assert error.code == code
    assert calls == 0


@pytest.mark.asyncio
async def test_rejects_any_private_or_reserved_dns_answer_before_transport() -> None:
    resolver = _Resolver(
        {
            "mixed.example": ["93.184.216.34", "10.0.0.5"],
            "reserved.example": ["192.0.2.10"],
        }
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="unsafe")

    service = ContentFetchService(resolver=resolver, transport=httpx.MockTransport(handler))
    mixed = await _fetch_error(service, "https://mixed.example/")
    reserved = await _fetch_error(service, "https://reserved.example/")

    assert mixed.code == "forbidden_target"
    assert reserved.code == "forbidden_target"
    assert calls == 0


@pytest.mark.asyncio
async def test_redirect_re_resolves_host_and_blocks_dns_rebinding() -> None:
    resolver = _Resolver(
        {"rebind.example": [["93.184.216.34"], ["127.0.0.1"]]}
    )
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(302, headers={"location": "/private"})

    service = ContentFetchService(resolver=resolver, transport=httpx.MockTransport(handler))
    error = await _fetch_error(service, "https://rebind.example/start")

    assert error.code == "forbidden_target"
    assert resolver.calls == [("rebind.example", 443), ("rebind.example", 443)]
    assert requests == 1


@pytest.mark.asyncio
async def test_redirect_to_link_local_metadata_target_is_blocked_before_second_request() -> None:
    resolver = _Resolver({"public.example": ["93.184.216.34"]})
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            302,
            headers={"location": "http://169.254.169.254/latest/meta-data"},
        )

    service = ContentFetchService(resolver=resolver, transport=httpx.MockTransport(handler))
    error = await _fetch_error(service, "https://public.example/start")

    assert error.code == "forbidden_target"
    assert requests == 1


@pytest.mark.asyncio
async def test_streaming_body_limit_rejects_decompressed_oversize_content() -> None:
    resolver = _Resolver({"large.example": ["93.184.216.34"]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_ChunkStream([b"12345", b"67890", b"overflow"]),
        )

    service = ContentFetchService(
        resolver=resolver,
        transport=httpx.MockTransport(handler),
        max_bytes=10,
    )
    error = await _fetch_error(service, "https://large.example/data")
    assert error.code == "response_too_large"


@pytest.mark.asyncio
async def test_content_length_limit_rejects_without_reading_body() -> None:
    resolver = _Resolver({"large.example": ["93.184.216.34"]})
    stream = _ChunkStream([b"should not be read"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-length": "500"},
            stream=stream,
        )

    service = ContentFetchService(
        resolver=resolver,
        transport=httpx.MockTransport(handler),
        max_bytes=100,
    )
    error = await _fetch_error(service, "https://large.example/data")
    assert error.code == "response_too_large"


@pytest.mark.asyncio
async def test_redirect_limit_and_content_type_are_fail_closed() -> None:
    resolver = _Resolver({"redirect.example": ["93.184.216.34"]})

    def redirects(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    limited = ContentFetchService(
        resolver=resolver,
        transport=httpx.MockTransport(redirects),
        max_redirects=1,
    )
    redirect_error = await _fetch_error(limited, "https://redirect.example/start")
    assert redirect_error.code == "redirect_limit"

    binary_resolver = _Resolver({"binary.example": ["93.184.216.34"]})

    def binary(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=b"binary",
        )

    binary_service = ContentFetchService(
        resolver=binary_resolver,
        transport=httpx.MockTransport(binary),
    )
    content_error = await _fetch_error(binary_service, "https://binary.example/file")
    assert content_error.code == "unsupported_content_type"


@pytest.mark.asyncio
async def test_total_and_read_timeouts_are_explicit_retryable_errors() -> None:
    total_service = ContentFetchService(
        resolver=_SlowResolver(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/plain"})
        ),
        total_timeout_seconds=0.001,
    )
    total_error = await _fetch_error(total_service, "https://slow.example/")
    assert total_error.code == "total_timeout"
    assert total_error.retryable is True

    resolver = _Resolver({"slow.example": ["93.184.216.34"]})

    def read_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    read_service = ContentFetchService(
        resolver=resolver,
        transport=httpx.MockTransport(read_timeout),
    )
    read_error = await _fetch_error(read_service, "https://slow.example/")
    assert read_error.code == "fetch_timeout"
    assert read_error.retryable is True
