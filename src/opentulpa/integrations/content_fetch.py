"""Bounded, DNS-pinned content fetching for model-requested research URLs."""

from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import json
import re
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_TEXTUAL_APPLICATION_TYPES = {
    "application/atom+xml",
    "application/json",
    "application/ld+json",
    "application/rss+xml",
    "application/xhtml+xml",
    "application/xml",
}
_BLOCKED_HOSTS = {
    "instance-data",
    "instance-data.ec2.internal",
    "localhost",
    "metadata",
    "metadata.google.internal",
}
_BLOCKED_HOST_SUFFIXES = (
    ".home",
    ".internal",
    ".lan",
    ".local",
    ".localdomain",
    ".localhost",
)
_CONTROL_OR_SPACE_RE = re.compile(r"[\x00-\x20\x7f]")
_BLOCKED_IPV6_TRANSLATION_NETWORKS = (
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)


class ContentFetchError(Exception):
    """Safe application error returned for rejected or failed fetches."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class HostResolver(Protocol):
    async def resolve(self, hostname: str, port: int) -> Sequence[str]: ...


class ContentExtractor(Protocol):
    def extract(
        self,
        *,
        body: bytes,
        content_type: str,
        charset: str,
        url: str,
    ) -> tuple[str, str | None]: ...


class SocketHostResolver:
    """Resolve TCP addresses without applying platform search-domain shortcuts."""

    async def resolve(self, hostname: str, port: int) -> Sequence[str]:
        loop = asyncio.get_running_loop()
        try:
            records = await loop.getaddrinfo(
                hostname,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
                flags=socket.AI_ADDRCONFIG,
            )
        except OSError as exc:
            raise ContentFetchError(
                "dns_resolution_failed",
                "The destination hostname could not be resolved.",
                retryable=True,
            ) from exc
        addresses: list[str] = []
        for record in records:
            address = str(record[4][0])
            if address not in addresses:
                addresses.append(address)
        return addresses


@dataclass(frozen=True, slots=True)
class ContentFetchResult:
    url: str
    status_code: int
    content_type: str
    charset: str
    title: str | None
    text: str
    bytes_read: int
    redirects: int

    def to_payload(self) -> dict[str, object]:
        return {
            "url": self.url,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "charset": self.charset,
            "title": self.title,
            "text": self.text,
            "bytes_read": self.bytes_read,
            "redirects": self.redirects,
        }


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    logical_url: httpx.URL
    hostname: str
    port: int
    address: str

    @property
    def pinned_url(self) -> httpx.URL:
        return self.logical_url.copy_with(host=self.address)

    @property
    def host_header(self) -> str:
        return self.logical_url.netloc.decode("ascii")


class _HTMLTextExtractor(HTMLParser):
    _BREAK_TAGS = {
        "article",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "td",
        "th",
        "tr",
    }
    _SKIP_TAGS = {"canvas", "noscript", "script", "style", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._title_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        lowered = tag.lower()
        if lowered in self._SKIP_TAGS:
            self._skip_depth += 1
        if lowered == "title":
            self._title_depth += 1
        if not self._skip_depth and lowered in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title" and self._title_depth:
            self._title_depth -= 1
        if lowered in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if not self._skip_depth and lowered in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
        self.parts.append(data)

    def result(self) -> tuple[str, str | None]:
        text = _normalize_text("".join(self.parts))
        title = " ".join("".join(self.title_parts).split()).strip() or None
        return text, title


def _normalize_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in str(value or "").replace("\x00", "").splitlines()]
    output: list[str] = []
    for line in lines:
        if line:
            output.append(line)
        elif output and output[-1]:
            output.append("")
    while output and not output[-1]:
        output.pop()
    return "\n".join(output)


def _decode_body(body: bytes, charset: str) -> str:
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _extract_text(body: bytes, content_type: str, charset: str) -> tuple[str, str | None]:
    decoded = _decode_body(body, charset)
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = _HTMLTextExtractor()
        parser.feed(decoded)
        parser.close()
        return parser.result()
    if content_type in {"application/json", "application/ld+json"}:
        try:
            parsed = json.loads(decoded)
        except (TypeError, ValueError):
            return _normalize_text(decoded), None
        return json.dumps(parsed, ensure_ascii=False, indent=2), None
    return _normalize_text(decoded), None


class PlainTextExtractor:
    """Deterministic fallback extractor with no optional parser dependency."""

    def extract(
        self,
        *,
        body: bytes,
        content_type: str,
        charset: str,
        url: str,
    ) -> tuple[str, str | None]:
        del url
        return _extract_text(body, content_type, charset)


class Crawl4AIContentExtractor:
    """Use Crawl4AI only as an offline HTML-to-Markdown transformer."""

    def extract(
        self,
        *,
        body: bytes,
        content_type: str,
        charset: str,
        url: str,
    ) -> tuple[str, str | None]:
        fallback_text, title = _extract_text(body, content_type, charset)
        if content_type not in {"text/html", "application/xhtml+xml"}:
            return fallback_text, title
        decoded = _decode_body(body, charset)
        from crawl4ai.markdown_generation_strategy import (  # type: ignore[import-untyped]
            DefaultMarkdownGenerator,
        )

        generated = DefaultMarkdownGenerator().generate_markdown(
            decoded,
            base_url=url,
            citations=True,
        )
        markdown = str(getattr(generated, "markdown_with_citations", "") or "").strip()
        if not markdown or markdown.startswith("Error "):
            return fallback_text, title
        references = str(getattr(generated, "references_markdown", "") or "").strip()
        if references:
            markdown = f"{markdown}\n\n{references}"
        return markdown, title


def default_content_extractor() -> ContentExtractor:
    """Use the optional rich parser when installed, otherwise stay dependency-free."""

    if importlib.util.find_spec("crawl4ai") is not None:
        return Crawl4AIContentExtractor()
    return PlainTextExtractor()


def is_forbidden_hostname(hostname: str) -> bool:
    normalized = str(hostname or "").lower().rstrip(".")
    return bool(
        not normalized
        or "%" in normalized
        or normalized in _BLOCKED_HOSTS
        or normalized.endswith(_BLOCKED_HOST_SUFFIXES)
    )


def is_forbidden_address(address: str) -> bool:
    if "%" in address:
        return True
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return True
    if isinstance(parsed, ipaddress.IPv6Address):
        if parsed.ipv4_mapped is not None:
            parsed = parsed.ipv4_mapped
        elif (
            parsed.sixtofour is not None
            or parsed.teredo is not None
            or any(parsed in network for network in _BLOCKED_IPV6_TRANSLATION_NETWORKS)
        ):
            return True
    return bool(
        not parsed.is_global
        or parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


class ContentFetchService:
    """Fetch textual content after validating and pinning every redirect target."""

    def __init__(
        self,
        *,
        resolver: HostResolver | None = None,
        extractor: ContentExtractor | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        allowed_http_ports: Sequence[int] = (80,),
        allowed_https_ports: Sequence[int] = (443,),
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 10.0,
        total_timeout_seconds: float = 20.0,
        max_bytes: int = 2_000_000,
        max_text_characters: int = 2_000_000,
        max_redirects: int = 5,
    ) -> None:
        self._resolver = resolver or SocketHostResolver()
        self._extractor = extractor or PlainTextExtractor()
        self._transport = transport
        self._allowed_ports = {
            "http": frozenset(int(port) for port in allowed_http_ports),
            "https": frozenset(int(port) for port in allowed_https_ports),
        }
        if not all(1 <= port <= 65_535 for ports in self._allowed_ports.values() for port in ports):
            raise ValueError("allowed ports must be between 1 and 65535")
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0 or total_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if max_text_characters <= 0:
            raise ValueError("max_text_characters must be positive")
        if max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        self._connect_timeout_seconds = connect_timeout_seconds
        self._read_timeout_seconds = read_timeout_seconds
        self._total_timeout_seconds = total_timeout_seconds
        self._max_bytes = max_bytes
        self._max_text_characters = max_text_characters
        self._max_redirects = max_redirects

    async def fetch(self, url: str) -> ContentFetchResult:
        try:
            async with asyncio.timeout(self._total_timeout_seconds):
                return await self._fetch_within_deadline(url)
        except TimeoutError as exc:
            raise ContentFetchError(
                "total_timeout",
                "The content fetch exceeded its total time limit.",
                retryable=True,
            ) from exc

    async def _fetch_within_deadline(self, url: str) -> ContentFetchResult:
        current_url = str(url or "").strip()
        redirects = 0
        transport = self._transport or httpx.AsyncHTTPTransport(
            verify=True,
            trust_env=False,
            retries=0,
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
        )
        timeout = httpx.Timeout(
            connect=self._connect_timeout_seconds,
            read=self._read_timeout_seconds,
            write=self._connect_timeout_seconds,
            pool=self._connect_timeout_seconds,
        )
        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                while True:
                    target = await self._resolve_target(current_url)
                    headers = {
                        "Accept": "text/html,text/plain,application/json,application/xml;q=0.9,*/*;q=0.1",
                        "Accept-Encoding": "identity",
                        "Host": target.host_header,
                        "User-Agent": "OpenTulpa-ContentFetch/2",
                    }
                    async with client.stream(
                        "GET",
                        target.pinned_url,
                        headers=headers,
                        extensions={"sni_hostname": target.hostname},
                    ) as response:
                        if response.status_code in _REDIRECT_STATUSES:
                            location = str(response.headers.get("location", "") or "").strip()
                            if not location:
                                raise ContentFetchError(
                                    "invalid_redirect",
                                    "The remote server returned a redirect without a destination.",
                                )
                            if redirects >= self._max_redirects:
                                raise ContentFetchError(
                                    "redirect_limit",
                                    "The content fetch exceeded its redirect limit.",
                                )
                            redirects += 1
                            current_url = urljoin(str(target.logical_url), location)
                            continue
                        if response.status_code < 200 or response.status_code >= 300:
                            raise ContentFetchError(
                                "http_status",
                                f"The remote server returned HTTP {response.status_code}.",
                                retryable=response.status_code >= 500,
                            )
                        content_encoding = str(
                            response.headers.get("content-encoding", "") or ""
                        ).strip().lower()
                        if content_encoding not in {"", "identity"}:
                            raise ContentFetchError(
                                "unsupported_content_encoding",
                                "The remote resource ignored the required identity encoding.",
                            )
                        content_type = self._content_type(response)
                        content_length = self._content_length(response)
                        if content_length is not None and content_length > self._max_bytes:
                            raise ContentFetchError(
                                "response_too_large",
                                "The remote content exceeds the configured size limit.",
                            )
                        chunks: list[bytes] = []
                        bytes_read = 0
                        async for chunk in response.aiter_bytes():
                            bytes_read += len(chunk)
                            if bytes_read > self._max_bytes:
                                raise ContentFetchError(
                                    "response_too_large",
                                    "The remote content exceeds the configured size limit.",
                                )
                            chunks.append(chunk)
                        body = b"".join(chunks)
                        charset = response.charset_encoding or "utf-8"
                        try:
                            text, title = await asyncio.to_thread(
                                self._extractor.extract,
                                body=body,
                                content_type=content_type,
                                charset=charset,
                                url=str(target.logical_url),
                            )
                        except Exception as exc:
                            raise ContentFetchError(
                                "content_decode_failed",
                                "The remote textual content could not be decoded.",
                            ) from exc
                        text = str(text or "")[: self._max_text_characters]
                        return ContentFetchResult(
                            url=str(target.logical_url),
                            status_code=response.status_code,
                            content_type=content_type,
                            charset=charset,
                            title=title,
                            text=text,
                            bytes_read=bytes_read,
                            redirects=redirects,
                        )
        except ContentFetchError:
            raise
        except httpx.TimeoutException as exc:
            raise ContentFetchError(
                "fetch_timeout",
                "The remote server timed out.",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ContentFetchError(
                "network_error",
                "The remote content could not be fetched.",
                retryable=True,
            ) from exc

    async def _resolve_target(self, raw_url: str) -> _ResolvedTarget:
        if not raw_url or len(raw_url) > 8_192 or _CONTROL_OR_SPACE_RE.search(raw_url):
            raise ContentFetchError("invalid_url", "A valid HTTP or HTTPS URL is required.")
        if "\\" in raw_url:
            raise ContentFetchError("invalid_url", "A valid HTTP or HTTPS URL is required.")
        try:
            parsed_input = urlsplit(raw_url)
        except ValueError as exc:
            raise ContentFetchError("invalid_url", "A valid HTTP or HTTPS URL is required.") from exc
        if parsed_input.username is not None or parsed_input.password is not None:
            raise ContentFetchError("url_credentials", "URLs containing user credentials are not allowed.")
        try:
            logical_url = httpx.URL(raw_url).copy_with(fragment=None)
        except (TypeError, ValueError) as exc:
            raise ContentFetchError("invalid_url", "A valid HTTP or HTTPS URL is required.") from exc
        scheme = logical_url.scheme.lower()
        if scheme not in {"http", "https"}:
            raise ContentFetchError("invalid_scheme", "Only HTTP and HTTPS URLs are allowed.")
        if logical_url.userinfo:
            raise ContentFetchError("url_credentials", "URLs containing user credentials are not allowed.")
        hostname = logical_url.raw_host.decode("ascii", errors="strict").lower().rstrip(".")
        if not hostname or "%" in hostname:
            raise ContentFetchError("invalid_url", "A valid destination hostname is required.")
        if is_forbidden_hostname(hostname):
            raise ContentFetchError("forbidden_target", "The destination address is not allowed.")
        port = logical_url.port or (443 if scheme == "https" else 80)
        if port not in self._allowed_ports[scheme]:
            raise ContentFetchError("forbidden_port", "The destination port is not allowed.")

        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            try:
                resolved = await self._resolver.resolve(hostname, port)
            except ContentFetchError:
                raise
            except Exception as exc:
                raise ContentFetchError(
                    "dns_resolution_failed",
                    "The destination hostname could not be resolved.",
                    retryable=True,
                ) from exc
            addresses = [str(value).strip() for value in resolved]
        else:
            addresses = [str(literal)]
        if not addresses:
            raise ContentFetchError(
                "dns_resolution_failed",
                "The destination hostname could not be resolved.",
                retryable=True,
            )
        if any(is_forbidden_address(address) for address in addresses):
            raise ContentFetchError("forbidden_target", "The destination address is not allowed.")
        return _ResolvedTarget(
            logical_url=logical_url,
            hostname=hostname,
            port=port,
            address=addresses[0],
        )

    @staticmethod
    def _content_type(response: httpx.Response) -> str:
        raw = str(response.headers.get("content-type", "") or "").strip().lower()
        content_type = raw.split(";", 1)[0].strip()
        if content_type.startswith("text/") or content_type in _TEXTUAL_APPLICATION_TYPES:
            return content_type
        raise ContentFetchError(
            "unsupported_content_type",
            "The remote resource is not a supported textual content type.",
        )

    @staticmethod
    def _content_length(response: httpx.Response) -> int | None:
        raw = str(response.headers.get("content-length", "") or "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value >= 0 else None


__all__ = [
    "ContentFetchError",
    "ContentFetchResult",
    "ContentFetchService",
    "HostResolver",
    "SocketHostResolver",
]
