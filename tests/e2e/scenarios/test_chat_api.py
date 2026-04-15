from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.api.routes import wake_search as wake_search_routes

TEST_USER_ID = "test_user_e2e"
pytestmark = [pytest.mark.e2e]


class _WeatherStormRuntime:
    def __init__(self, *, artifact_root: Path) -> None:
        self.calls: list[dict[str, Any]] = []
        self._app: Any | None = None
        self._artifact_root = artifact_root
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self.produced_artifacts: list[Path] = []
        self.search_query: str = ""
        self.search_started_at: float | None = None
        self.free_api_started_at: float | None = None
        self.free_api_completed_at: float | None = None

    def bind_app(self, app: Any) -> None:
        self._app = app

    async def _run_internal_web_search(self, query: str) -> dict[str, Any]:
        if self._app is None:
            raise RuntimeError("app not bound")
        self.search_started_at = time.monotonic()
        self.search_query = query
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/internal/web_search", json={"query": query})
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result", {})
        return result if isinstance(result, dict) else {"answer": str(result)}

    async def _call_free_weather_api(self) -> dict[str, Any]:
        self.free_api_started_at = time.monotonic()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": 14.60,
                    "longitude": 120.98,
                    "hourly": "precipitation_probability,windspeed_10m",
                    "forecast_days": 1,
                    "timezone": "UTC",
                },
            )
        response.raise_for_status()
        self.free_api_completed_at = time.monotonic()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _write_report_artifact(self, *, report_text: str, thread_id: str) -> Path:
        safe_thread = str(thread_id or "chat").replace("/", "_")
        artifact_path = self._artifact_root / f"{safe_thread}_pacific_storm_report.md"
        artifact_path.write_text(report_text, encoding="utf-8")
        self.produced_artifacts.append(artifact_path)
        return artifact_path

    async def ainvoke_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str = "interactive",
        include_pending_context: bool = True,
        recursion_limit_override: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "thread_id": thread_id,
                "customer_id": customer_id,
                "text": text,
                "turn_mode": turn_mode,
                "include_pending_context": include_pending_context,
                "recursion_limit_override": recursion_limit_override,
            }
        )
        query = (
            "recent weather storms in the pacific ocean in the last 7 days "
            "with key impacts and links"
        )
        search_result = await self._run_internal_web_search(query)
        weather_payload = await self._call_free_weather_api()

        answer = str(search_result.get("answer", "")).strip()
        source_count = int(search_result.get("source_count", 0) or 0)
        hourly = weather_payload.get("hourly", {}) if isinstance(weather_payload, dict) else {}
        precip = hourly.get("precipitation_probability", []) if isinstance(hourly, dict) else []
        wind = hourly.get("windspeed_10m", []) if isinstance(hourly, dict) else []
        precip_peak = max(precip) if isinstance(precip, list) and precip else 0
        wind_peak = max(wind) if isinstance(wind, list) and wind else 0

        report_lines = [
            "Pacific Weather Storm Report",
            "",
            f"Customer: {customer_id}",
            f"Thread: {thread_id}",
            "",
            f"Search source count: {source_count}",
            f"Search summary: {answer[:500] if answer else 'No summary returned.'}",
            "",
            "Free API snapshot (Open-Meteo):",
            f"- Peak hourly precipitation probability: {precip_peak}",
            f"- Peak hourly wind speed (10m): {wind_peak}",
        ]
        report_text = "\n".join(report_lines).strip()
        artifact_path = self._write_report_artifact(report_text=report_text, thread_id=thread_id)
        return f"{report_text}\n\nArtifact: {artifact_path}"


@pytest.fixture()
def chat_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, _WeatherStormRuntime]:
    async def _fake_run_web_search(query: str) -> dict[str, Any]:
        return {
            "answer": f"Tracked Pacific storm activity for query: {query}.",
            "source_count": 3,
            "sources": [
                {"url": "https://example.com/storm-1", "domain": "example.com"},
                {"url": "https://example.com/storm-2", "domain": "example.com"},
            ],
            "model": "test-double",
        }

    original_get = httpx.AsyncClient.get

    async def _fake_get(self: httpx.AsyncClient, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        if str(url).startswith("https://api.open-meteo.com/v1/forecast"):
            request = httpx.Request("GET", str(url), params=kwargs.get("params"))
            return httpx.Response(
                status_code=200,
                request=request,
                json={
                    "hourly": {
                        "precipitation_probability": [20, 55, 80, 35],
                        "windspeed_10m": [12.3, 22.8, 31.1, 18.4],
                    }
                },
            )
        return await original_get(self, url, *args, **kwargs)

    monkeypatch.setattr(wake_search_routes, "run_web_search", _fake_run_web_search)
    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

    runtime = _WeatherStormRuntime(artifact_root=tmp_path / "artifacts")
    app = create_app(agent_runtime=runtime)
    runtime.bind_app(app)
    with TestClient(app) as client:
        yield client, runtime


def test_internal_chat_rejects_missing_fields(chat_client: tuple[TestClient, _WeatherStormRuntime]) -> None:
    client, _ = chat_client
    response = client.post("/internal/chat", json={"customer_id": TEST_USER_ID})
    assert response.status_code == 400
    assert "customer_id and text are required" in response.json()["detail"]


def test_internal_chat_e2e_weather_report_uses_search_and_free_api(
    chat_client: tuple[TestClient, _WeatherStormRuntime],
) -> None:
    client, runtime = chat_client
    thread_id = "e2e-weather-thread-001"
    request_started_at = time.monotonic()

    response = client.post(
        "/internal/chat",
        json={
            "customer_id": TEST_USER_ID,
            "thread_id": thread_id,
            "text": (
                "Please search recent weather storms in the Pacific and produce a report "
                "using a free weather API."
            ),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["customer_id"] == TEST_USER_ID
    assert payload["thread_id"] == thread_id
    assert "Pacific Weather Storm Report" in payload["text"]
    assert "Search source count: 3" in payload["text"]
    assert "Free API snapshot (Open-Meteo)" in payload["text"]

    assert runtime.search_started_at is not None
    assert runtime.search_started_at - request_started_at <= 30.0
    assert "weather storms in the pacific" in runtime.search_query.lower()
    assert runtime.free_api_started_at is not None
    assert runtime.free_api_completed_at is not None
    assert runtime.free_api_completed_at >= runtime.free_api_started_at
    assert len(runtime.produced_artifacts) == 1
    artifact_path = runtime.produced_artifacts[0]
    assert artifact_path.exists()
    artifact_text = artifact_path.read_text(encoding="utf-8")
    assert "Pacific Weather Storm Report" in artifact_text

    # Clean up all produced artifacts at test end.
    for path in runtime.produced_artifacts:
        path.unlink(missing_ok=True)
    assert all(not p.exists() for p in runtime.produced_artifacts)
