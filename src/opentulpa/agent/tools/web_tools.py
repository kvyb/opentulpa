"""Web tools."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from opentulpa.agent.file_analysis import summarize_uploaded_blob
from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.tools.core_tools import _crawl4ai_extract
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.agent.utils import extract_html_title as _extract_html_title
from opentulpa.agent.utils import html_to_text as _html_to_text
from opentulpa.integrations.web_search import get_web_search_backend_name

ExaSearchType = Literal["auto", "fast", "neural", "deep"]
ExaCategory = Literal[
    "company",
    "research paper",
    "news",
    "pdf",
    "github",
    "tweet",
    "personal site",
    "financial report",
    "people",
]


def register_web_tools(runtime: Any) -> dict[str, Any]:
    def _web_search_request_payload(query: str, **optional_args: object) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        payload.update({key: value for key, value in optional_args.items() if value is not None})
        return payload

    async def _run_web_search_payload(payload: dict[str, Any]) -> Any:
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/web_search",
            json_body=payload,
            timeout=90.0,
        )
        if r.status_code != 200:
            return {"error": "web_search request failed"}
        return r.json().get("result", "No result.")

    if get_web_search_backend_name() == "exa":
        @tool
        async def web_search(
            query: str,
            search_type: ExaSearchType | None = None,
            category: ExaCategory | None = None,
        ) -> Any:
            """
            Search the web with Exa.

            Optional args: search_type (auto, fast, neural, deep) and category
            (news, research paper, github, pdf, company, people, personal site,
            financial report, tweet). Exa returns 20 raw results by default.
            """
            payload = _web_search_request_payload(
                query,
                search_type=search_type,
                category=category,
            )
            return await _run_web_search_payload(payload)

    else:

        @tool
        async def web_search(query: str) -> Any:
            """Search the web for current information."""
            return await _run_web_search_payload(_web_search_request_payload(query))

    async def _fetch_remote_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
        target: str = "url",
    ) -> Any:
        raw_url = str(url or "").strip()
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            return {"error": "url must start with http:// or https://"}

        safe_max_chars = max(2000, min(int(max_chars), 120000))
        try:
            async with httpx.AsyncClient(
                timeout=45.0,
                follow_redirects=True,
                headers={"User-Agent": "OpenTulpa/0.1 (+content-fetch)"},
            ) as client:
                resp = await client.get(raw_url)
        except Exception as exc:
            return {"error": f"link fetch failed: {exc}"}

        if resp.status_code >= 400:
            return {"error": f"link fetch failed: HTTP {resp.status_code}"}

        ctype = str(resp.headers.get("content-type", "")).split(";")[0].strip().lower()
        final_url = str(resp.url)
        text_content = ""
        title: str | None = None
        mode = "text"
        is_image = ctype.startswith("image/")
        is_pdf = ctype == "application/pdf" or final_url.lower().endswith(".pdf")
        is_docx = (
            ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or final_url.lower().endswith(".docx")
        )
        file_like = is_image or is_pdf or is_docx

        safe_target = str(target or "url").strip().lower()
        if safe_target == "url" and file_like:
            return {
                "error": (
                    "URL points to a file-like resource (image/pdf/docx). "
                    "Use fetch_file_content instead."
                ),
                "url": final_url,
                "content_type": ctype or "unknown",
            }
        if safe_target == "file" and not file_like:
            return {
                "error": (
                    "URL does not point to supported file-like content (image/pdf/docx). "
                    "Use fetch_url_content instead."
                ),
                "url": final_url,
                "content_type": ctype or "unknown",
            }

        try:
            if is_image:
                mode = "image_vision"
                if use_vision_for_images:
                    vision = await runtime._model.ainvoke(
                        [
                            SystemMessage(
                                content=(
                                    "Describe the image and extract all readable text. "
                                    "If it is a screenshot/document, summarize key points."
                                )
                            ),
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": "Analyze this image URL."},
                                    {"type": "image_url", "image_url": {"url": final_url}},
                                ]
                            ),
                        ]
                    )
                    text_content = _content_to_text(getattr(vision, "content", "")).strip()
            elif is_pdf:
                mode = "pdf_llm"
                text_content = await summarize_uploaded_blob(
                    runtime,
                    filename=final_url.rsplit("/", 1)[-1] or "document.pdf",
                    mime_type=ctype or "application/pdf",
                    kind="document",
                    raw_bytes=resp.content,
                    question=(
                        "Extract key information from this PDF and provide a concise but complete "
                        "summary with important facts, entities, dates, and actions."
                    ),
                )
            elif is_docx:
                mode = "docx_llm"
                text_content = await summarize_uploaded_blob(
                    runtime,
                    filename=final_url.rsplit("/", 1)[-1] or "document.docx",
                    mime_type=ctype
                    or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    kind="document",
                    raw_bytes=resp.content,
                    question=(
                        "Extract key information from this DOCX and provide a concise but complete "
                        "summary with important facts, entities, dates, and actions."
                    ),
                )
            else:
                mode = "web_text"
                raw_text = resp.text
                if "html" in ctype or "<html" in raw_text.lower():
                    crawled_text, crawled_title, crawl_error = await _crawl4ai_extract(final_url)
                    if crawled_text:
                        mode = "web_text_crawl4ai"
                        title = crawled_title or _extract_html_title(raw_text)
                        text_content = crawled_text
                    else:
                        mode = "web_text_fallback"
                        title = _extract_html_title(raw_text)
                        text_content = _html_to_text(raw_text)
                        if crawl_error:
                            text_content = (
                                f"[crawl4ai_fallback_reason] {crawl_error}\n\n{text_content}"
                            )
                else:
                    text_content = raw_text
        except Exception as exc:
            return {
                "error": f"content extraction failed: {exc}",
                "url": final_url,
                "content_type": ctype or "unknown",
            }

        normalized = re.sub(r"\n{3,}", "\n\n", str(text_content or "").strip())
        truncated = len(normalized) > safe_max_chars
        return {
            "url": final_url,
            "content_type": ctype or "unknown",
            "mode": mode,
            "title": title,
            "char_count": len(normalized),
            "truncated": truncated,
            "text": normalized[:safe_max_chars],
        }

    @tool
    async def fetch_url_content(url: str, max_chars: int = 40000) -> Any:
        """Fetch and extract web page/text/JSON content from a URL."""
        return await _fetch_remote_content(
            url=url,
            max_chars=max_chars,
            use_vision_for_images=False,
            target="url",
        )

    @tool
    async def fetch_file_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
    ) -> Any:
        """Fetch and analyze file-like URL content (image/pdf/docx)."""
        return await _fetch_remote_content(
            url=url,
            max_chars=max_chars,
            use_vision_for_images=use_vision_for_images,
            target="file",
        )

    return {
        "web_search": web_search,
        "fetch_url_content": fetch_url_content,
        "fetch_file_content": fetch_file_content,
    }
