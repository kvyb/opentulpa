from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from opentulpa.api import app as app_module
from opentulpa.api.app import create_app
from opentulpa.api.tulpa_loader import TulpaRouterLoader


def test_tulpa_loader_mounts_internal_router(tmp_path) -> None:
    project_root = tmp_path / "project"
    package_dir = project_root / "tulpa_stuff"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text('"""tulpas."""\n', encoding="utf-8")
    (package_dir / "hook.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/health')\n"
        "async def health():\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    internal = APIRouter()
    loader = TulpaRouterLoader(
        project_root=project_root,
        mount_router=internal,
    )
    result = loader.reload()

    app = FastAPI()
    app.include_router(internal, prefix="/tulpa")

    with TestClient(app) as client:
        assert client.get("/tulpa/hook/health").status_code == 200

    assert result["loaded"] == ["hook"]
    assert set(result) == {"ok", "loaded", "warnings", "errors", "mount_prefix"}


def test_tulpa_loader_skips_non_router_scripts_without_import_side_effects(tmp_path) -> None:
    project_root = tmp_path / "project"
    package_dir = project_root / "tulpa_stuff"
    marker_path = project_root / "side_effect.txt"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text('"""tulpas."""\n', encoding="utf-8")
    (package_dir / "script.py").write_text(
        f"from pathlib import Path\nPath({str(marker_path)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    (package_dir / "hook.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/health')\n"
        "async def health():\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    loader = TulpaRouterLoader(
        project_root=project_root,
        mount_router=APIRouter(),
    )
    result = loader.reload()

    assert result["loaded"] == ["hook"]
    assert result["errors"] == []
    assert result["warnings"] == []
    assert not marker_path.exists()


def test_tulpa_loader_skips_router_modules_with_top_level_side_effects(tmp_path) -> None:
    project_root = tmp_path / "project"
    package_dir = project_root / "tulpa_stuff"
    marker_path = project_root / "side_effect.txt"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text('"""tulpas."""\n', encoding="utf-8")
    (package_dir / "unsafe_hook.py").write_text(
        "from fastapi import APIRouter\n"
        "from pathlib import Path\n"
        "router = APIRouter()\n"
        f"Path({str(marker_path)!r}).write_text('imported')\n"
        "@router.get('/health')\n"
        "async def health():\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    loader = TulpaRouterLoader(
        project_root=project_root,
        mount_router=APIRouter(),
    )
    result = loader.reload()

    assert result["loaded"] == []
    assert result["errors"] == []
    assert result["warnings"] == [
        {
            "module": "unsafe_hook",
            "warning": "import safety guard: unsafe top-level statement at line 4: Expr",
        }
    ]
    assert not marker_path.exists()


def test_create_app_mounts_internal_tulpa_router_on_startup(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    package_dir = project_root / "tulpa_stuff"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text('"""tulpas."""\n', encoding="utf-8")
    (package_dir / "hook.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/health')\n"
        "async def health():\n"
        "    return {'ok': True, 'via': 'startup'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)

    app = create_app()

    with TestClient(app) as client:
        response = client.get("/tulpa/hook/health")
        assert response.status_code == 200
        assert response.json()["via"] == "startup"


def test_internal_tulpa_reload_remounts_new_internal_routes(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    package_dir = project_root / "tulpa_stuff"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text('"""tulpas."""\n', encoding="utf-8")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)

    app = create_app()

    with TestClient(app) as client:
        assert client.get("/tulpa/hook/health").status_code == 404
        (package_dir / "hook.py").write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/health')\n"
            "async def health():\n"
            "    return {'ok': True, 'via': 'reload'}\n",
            encoding="utf-8",
        )
        reload_response = client.post("/internal/tulpa/reload")
        assert reload_response.status_code == 200
        response = client.get("/tulpa/hook/health")
        assert response.status_code == 200
        assert response.json()["via"] == "reload"
