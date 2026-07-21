from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.bootstrap.host import InMemoryReleaseHost
from opentulpa.bootstrap.models import ActivationStatus, ReleaseRecord
from opentulpa.bootstrap.recovery import RecoveryService, create_recovery_router
from opentulpa.bootstrap.store import BootstrapStore
from opentulpa.bootstrap.supervisor import BootstrapSupervisor, SupervisorPolicy


def _release(name: str, character: str) -> ReleaseRecord:
    return ReleaseRecord(
        id=f"release_{name}",
        candidate_id=f"candidate_{name}",
        source_commit=character * 40,
        artifact_digest=f"sha256:{character * 64}",
        manifest_digest=f"sha256:{character * 64}",
        entrypoint=("python", "-m", f"release_{name}"),
    )


def _policy() -> SupervisorPolicy:
    return SupervisorPolicy(
        stage_probe_attempts=1,
        production_probe_attempts=1,
        probe_interval_seconds=0,
        probation_seconds=0,
        probation_probe_interval_seconds=1,
    )


@pytest.mark.asyncio
async def test_recovery_router_can_install_first_release(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    supervisor = BootstrapSupervisor(
        store=store,
        host=InMemoryReleaseHost(),
        policy=_policy(),
    )
    await supervisor.start()
    token = "r" * 32
    app = FastAPI()
    app.include_router(create_recovery_router(RecoveryService(supervisor), recovery_token=token))

    with TestClient(app) as client:
        response = client.post(
            "/bootstrap/v1/releases/initial",
            headers={"Authorization": f"Bearer {token}"},
            json={"release": _release("blue", "a").model_dump(mode="json")},
        )

    assert response.status_code == 201
    assert response.json()["release_id"] == "release_blue"
    assert store.get_state().serving_release_id == "release_blue"
    assert store.get_state().ingress_paused is False


@pytest.mark.asyncio
async def test_recovery_service_can_restore_previous_release(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    supervisor = BootstrapSupervisor(
        store=store,
        host=InMemoryReleaseHost(),
        policy=_policy(),
    )
    await supervisor.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await supervisor.install_initial(blue)
    deploy = await supervisor.request_activation(green, origin=None)
    assert (await supervisor.activate(deploy.id)).status is ActivationStatus.ACTIVE
    recovery = RecoveryService(supervisor)

    queued = await recovery.request_rollback(reason="normal UI is unavailable")
    completed = await recovery.wait(queued.id)

    assert completed.status is ActivationStatus.ACTIVE
    assert store.get_state().serving_release_id == blue.id


@pytest.mark.asyncio
async def test_recovery_router_uses_separate_auth_and_survives_safe_mode(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    supervisor = BootstrapSupervisor(
        store=store,
        host=InMemoryReleaseHost(),
        policy=_policy(),
    )
    await supervisor.start()
    await supervisor.install_initial(_release("blue", "a"))
    token = "r" * 32
    app = FastAPI()
    app.include_router(create_recovery_router(RecoveryService(supervisor), recovery_token=token))

    with TestClient(app) as client:
        assert client.get("/bootstrap/v1/status").status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        status_response = client.get("/bootstrap/v1/status", headers=headers)
        assert status_response.status_code == 200
        assert status_response.json()["state"]["serving_release_id"] == "release_blue"
        assert client.get("/recovery").status_code == 404
        assert client.get("/recovery/index.html").status_code == 404
        green = _release("green", "b")
        queued = client.post(
            "/bootstrap/v1/activations",
            headers=headers,
            json={
                "release": green.model_dump(mode="json"),
                "reason": "approved from recovery",
                "start": False,
            },
        )
        assert queued.status_code == 202
        activation_id = queued.json()["id"]
        activation = client.get(
            f"/bootstrap/v1/activations/{activation_id}",
            headers=headers,
        )
        assert activation.json()["status"] == "queued"
        cancelled = client.post(
            f"/bootstrap/v1/activations/{activation_id}/cancel",
            headers=headers,
        )
        assert cancelled.json()["status"] == "cancelled"
        safe = client.post("/bootstrap/v1/safe-mode", headers=headers)
        assert safe.status_code == 202
        restarted = client.post("/bootstrap/v1/restart", headers=headers)
        assert restarted.status_code == 202

    state = store.get_state()
    assert state.safe_mode is False
    assert state.serving_release_id == "release_blue"
    assert state.ingress_paused is False


@pytest.mark.asyncio
async def test_recovery_is_cli_only_and_all_control_apis_require_auth(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    supervisor = BootstrapSupervisor(
        store=store,
        host=InMemoryReleaseHost(),
        policy=_policy(),
    )
    await supervisor.start()
    await supervisor.install_initial(_release("blue", "a"))
    token = "recovery-token-" + "r" * 32
    app = FastAPI()
    app.include_router(
        create_recovery_router(RecoveryService(supervisor), recovery_token=token)
    )

    protected_routes = (
        ("GET", "/bootstrap/v1/status"),
        ("POST", "/bootstrap/v1/activations"),
        ("POST", "/bootstrap/v1/releases/initial"),
        ("GET", "/bootstrap/v1/activations/activation_test"),
        ("POST", "/bootstrap/v1/rollback"),
        ("POST", "/bootstrap/v1/activations/activation_test/cancel"),
        ("POST", "/bootstrap/v1/safe-mode"),
        ("POST", "/bootstrap/v1/restart"),
    )

    with TestClient(app) as client:
        assert client.get("/recovery").status_code == 404
        assert client.get("/recovery/anything").status_code == 404
        for method, path in protected_routes:
            response = client.request(method, path)
            assert response.status_code == 401, path
            assert token not in response.text
        invalid = client.get(
            "/bootstrap/v1/status",
            headers={"Authorization": f"Bearer {token}-invalid"},
        )
        for browser_headers in (
            {"Origin": "https://owner.example"},
            {"Referer": "https://owner.example/recovery"},
            {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"},
        ):
            rejected = client.get(
                "/bootstrap/v1/status",
                headers={"Authorization": f"Bearer {token}", **browser_headers},
            )
            assert rejected.status_code == 403
            assert token not in rejected.text

    assert invalid.status_code == 401
    assert token not in invalid.text
