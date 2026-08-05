from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from opentulpa.evolution.dependency_resolver import (
    _NORMALIZE_SITE_SCRIPT,
    DependencyResolutionError,
    DependencyResolverPolicy,
    TrustedDependencyResolver,
)
from opentulpa.evolution.process import BoundedProcessResult

_IMAGE = "sha256:" + "a" * 64
_WHEEL_HASH = "b" * 64


def _pyproject(*, dependency: str = "demo>=1") -> str:
    return (
        "[project]\n"
        "name = 'opentulpa'\n"
        "version = '0.1.0'\n"
        "requires-python = '>=3.12'\n"
        f"dependencies = [{dependency!r}]\n"
        "[project.optional-dependencies]\n"
        "evaluation = ['pytest>=8', 'ruff>=0.9', 'mypy>=1.0']\n"
        "[build-system]\n"
        "requires = ['hatchling==1.27.0']\n"
        "build-backend = 'hatchling.build'\n"
    )


def _lock(*, registry: str = "https://pypi.org/simple") -> bytes:
    return (
        "version = 1\n"
        "requires-python = '>=3.12'\n"
        "[[package]]\n"
        "name = 'demo'\n"
        "version = '1.0.0'\n"
        f"source = {{ registry = {registry!r} }}\n"
        "wheels = [{ url = "
        "'https://files.pythonhosted.org/packages/demo-1.0.0-py3-none-any.whl', "
        f"hash = 'sha256:{_WHEEL_HASH}', size = 300 }}]\n"
        "[[package]]\n"
        "name = 'opentulpa'\n"
        "version = '0.1.0'\n"
        "source = { editable = '.' }\n"
    ).encode()


class _ResolverRunner:
    def __init__(
        self,
        *,
        registry: str = "https://pypi.org/simple",
        volume_root: Path | None = None,
    ) -> None:
        self.registry = registry
        self.volume_root = volume_root
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        max_output_bytes: int,
        timeout_cleanup: Callable[[], None] | None = None,
    ) -> BoundedProcessResult:
        del cwd, timeout_seconds, max_output_bytes, timeout_cleanup
        call = tuple(argv)
        self.calls.append(call)
        if call[1:3] == ("image", "inspect"):
            return BoundedProcessResult(
                returncode=0,
                output=json.dumps({"Id": _IMAGE, "Config": {"Env": ["PATH=/usr/local/bin"]}}).encode(),
                truncated=False,
                timed_out=False,
            )
        mount = next(value for value in call if value.startswith(("type=bind,", "type=volume,")))
        if mount.startswith("type=bind,"):
            staging = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/resolution"))
        else:
            assert self.volume_root is not None
            subpath = mount.rpartition("volume-subpath=")[2]
            staging = self.volume_root / subpath
        if "lock" in call:
            (staging / "uv.lock").write_bytes(_lock(registry=self.registry))
        elif "export" in call:
            (staging / "requirements.txt").write_text(
                f"demo==1.0.0 \\\n    --hash=sha256:{_WHEEL_HASH}\n",
                encoding="utf-8",
            )
        elif "download" in call:
            with zipfile.ZipFile(
                staging / "wheelhouse" / "demo-1.0.0-py3-none-any.whl",
                "w",
            ) as archive:
                metadata_info = zipfile.ZipInfo(
                    "demo-1.0.0.dist-info/METADATA",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                archive.writestr(
                    metadata_info,
                    "Metadata-Version: 2.1\nName: demo\nVersion: 1.0.0\n",
                )
                wheel_info = zipfile.ZipInfo(
                    "demo-1.0.0.dist-info/WHEEL",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                archive.writestr(
                    wheel_info,
                    "Wheel-Version: 1.0\nTag: py3-none-any\n",
                )
        elif "install" in call:
            site = staging / "site"
            (site / "demo").mkdir(parents=True)
            (site / "demo" / "__init__.py").write_text("VERSION = '1.0.0'\n", encoding="utf-8")
            metadata = site / "demo-1.0.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: demo\nVersion: 1.0.0\n",
                encoding="utf-8",
            )
            (metadata / "RECORD").write_text(str(staging), encoding="utf-8")
            (metadata / "uv_cache.json").write_text(str(staging), encoding="utf-8")
        elif _NORMALIZE_SITE_SCRIPT in call:
            for name in ("RECORD", "uv_cache.json"):
                (staging / "site" / "demo-1.0.0.dist-info" / name).unlink()
        else:
            raise AssertionError(f"unexpected resolver command: {call!r}")
        return BoundedProcessResult(
            returncode=0,
            output=b"",
            truncated=False,
            timed_out=False,
        )


def _resolver(tmp_path: Path, runner: _ResolverRunner) -> TrustedDependencyResolver:
    trusted = tmp_path / "trusted-pyproject.toml"
    trusted_lock = tmp_path / "trusted-uv.lock"
    trusted.write_text(_pyproject(), encoding="utf-8")
    trusted_lock.write_bytes(_lock())
    return TrustedDependencyResolver(
        policy=DependencyResolverPolicy(
            bases_root=tmp_path / "bases",
            state_root=tmp_path / "state",
            trusted_pyproject=trusted,
            trusted_lock=trusted_lock,
            resolver_image_digest=_IMAGE,
        ),
        runner=runner,
    )


@pytest.mark.asyncio
async def test_resolver_publishes_sealed_content_addressed_inputs(tmp_path: Path) -> None:
    runner = _ResolverRunner()
    resolver = _resolver(tmp_path, runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text(
        _pyproject(dependency="demo>=1.0"),
        encoding="utf-8",
    )

    first = await resolver.resolve(workspace)
    second = await resolver.resolve(workspace)

    assert first == second
    assert resolver.base_for_lock(first.lock_sha256) == first
    assert first.root.name == first.id
    assert first.lock_sha256 == hashlib.sha256(first.lock_path.read_bytes()).hexdigest()
    assert stat.S_IMODE(first.root.stat().st_mode) == 0o555
    assert stat.S_IMODE(first.wheelhouse.stat().st_mode) == 0o555
    assert stat.S_IMODE(first.dependency_site.stat().st_mode) == 0o555
    assert {path.name for path in first.wheelhouse.iterdir()} == {
        "demo-1.0.0-py3-none-any.whl"
    }
    inventory = json.loads((first.root / "inventory.json").read_text(encoding="utf-8"))
    assert inventory == [
        {
            "filename": "demo-1.0.0-py3-none-any.whl",
            "name": "demo",
            "sha256": inventory[0]["sha256"],
            "size": inventory[0]["size"],
            "version": "1.0.0",
        }
    ]
    resolver_calls = [call for call in runner.calls if len(call) > 1 and call[1] == "run"]
    assert len(resolver_calls) == 10
    assert all("--network=bridge" in call for call in resolver_calls)
    assert all(not any("TOKEN" in value or "KEY=" in value for value in call) for call in resolver_calls)
    assert any("--no-sources" in call for call in resolver_calls)
    assert any("--only-binary=:all:" in call and "--require-hashes" in call for call in resolver_calls)
    assert any("--target" in call and "/resolution/site" in call for call in resolver_calls)
    assert any(_NORMALIZE_SITE_SCRIPT in call for call in resolver_calls)
    assert not (first.dependency_site / "demo-1.0.0.dist-info" / "RECORD").exists()
    assert not (first.dependency_site / "demo-1.0.0.dist-info" / "uv_cache.json").exists()
    next(first.wheelhouse.iterdir()).chmod(0o644)
    assert resolver.base_for_lock(first.lock_sha256) is None


@pytest.mark.asyncio
async def test_resolver_mounts_only_staging_subpath_from_named_volume(tmp_path: Path) -> None:
    volume_root = tmp_path / "resolver-volume"
    runner = _ResolverRunner(volume_root=volume_root)
    trusted = tmp_path / "trusted-pyproject.toml"
    trusted_lock = tmp_path / "trusted-uv.lock"
    trusted.write_text(_pyproject(), encoding="utf-8")
    trusted_lock.write_bytes(_lock())
    resolver = TrustedDependencyResolver(
        policy=DependencyResolverPolicy(
            bases_root=volume_root / "bases",
            state_root=volume_root / "state",
            trusted_pyproject=trusted,
            trusted_lock=trusted_lock,
            resolver_image_digest=_IMAGE,
            container_volume_name="opentulpa-resolver-e2e",
            container_volume_root=volume_root,
        ),
        runner=runner,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text(_pyproject(), encoding="utf-8")

    await resolver.resolve(workspace)

    resolver_calls = [call for call in runner.calls if len(call) > 1 and call[1] == "run"]
    mounts = [next(call[index + 1] for index, value in enumerate(call) if value == "--mount") for call in resolver_calls]
    assert resolver_calls
    assert all(
        mount.startswith(
            "type=volume,src=opentulpa-resolver-e2e,dst=/resolution,"
            "volume-subpath=state/resolve-"
        )
        for mount in mounts
    )
    assert all("type=bind" not in mount for mount in mounts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dependency",
    [
        "demo @ https://example.com/demo.whl",
        "demo @ file:///tmp/demo.whl",
    ],
)
async def test_resolver_rejects_direct_dependency_sources(
    tmp_path: Path,
    dependency: str,
) -> None:
    runner = _ResolverRunner()
    resolver = _resolver(tmp_path, runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text(
        _pyproject(dependency=dependency),
        encoding="utf-8",
    )

    with pytest.raises(DependencyResolutionError, match="direct or invalid"):
        await resolver.resolve(workspace)

    assert runner.calls == []


@pytest.mark.asyncio
async def test_resolver_rejects_non_dependency_metadata_changes(tmp_path: Path) -> None:
    runner = _ResolverRunner()
    resolver = _resolver(tmp_path, runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    changed = _pyproject().replace("version = '0.1.0'", "version = '9.9.9'")
    (workspace / "pyproject.toml").write_text(changed, encoding="utf-8")

    with pytest.raises(DependencyResolutionError, match="only project dependency"):
        await resolver.resolve(workspace)

    assert runner.calls == []


@pytest.mark.asyncio
async def test_resolver_rejects_lock_that_escaped_trusted_index(tmp_path: Path) -> None:
    runner = _ResolverRunner(registry="https://packages.example.com/simple")
    resolver = _resolver(tmp_path, runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text(_pyproject(), encoding="utf-8")

    with pytest.raises(DependencyResolutionError, match="escaped the trusted index"):
        await resolver.resolve(workspace)

    assert not any((tmp_path / "bases").iterdir())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_oci_resolver_builds_offline_dependency_base(tmp_path: Path) -> None:
    image = os.environ.get("OPENTULPA_TEST_DEPENDENCY_RESOLVER_IMAGE", "").strip()
    if not image:
        pytest.skip("set OPENTULPA_TEST_DEPENDENCY_RESOLVER_IMAGE to an immutable local image ID")
    trusted = tmp_path / "trusted-pyproject.toml"
    trusted_lock = tmp_path / "trusted-uv.lock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def project(version: str) -> str:
        return (
            "[project]\n"
            "name = 'resolver-fixture'\n"
            "version = '0.1.0'\n"
            "requires-python = '>=3.12'\n"
            f"dependencies = ['idna=={version}']\n"
            "[build-system]\n"
            "requires = ['hatchling==1.27.0']\n"
            "build-backend = 'hatchling.build'\n"
        )

    trusted.write_text(project("3.10"), encoding="utf-8")
    trusted_lock.write_text(
        "version = 1\nrequires-python = '>=3.12'\n",
        encoding="utf-8",
    )
    (workspace / "pyproject.toml").write_text(project("3.11"), encoding="utf-8")
    resolver = TrustedDependencyResolver(
        policy=DependencyResolverPolicy(
            bases_root=tmp_path / "bases",
            state_root=tmp_path / "state",
            trusted_pyproject=trusted,
            trusted_lock=trusted_lock,
            resolver_image_digest=image,
            extras=(),
            uid=os.getuid(),
            gid=os.getgid(),
        )
    )

    resolved = await resolver.resolve(workspace)
    repeated = await resolver.resolve(workspace)

    inventory = json.loads((resolved.root / "inventory.json").read_text(encoding="utf-8"))
    assert repeated == resolved
    assert [(item["name"], item["version"]) for item in inventory] == [("idna", "3.11")]
    assert (resolved.dependency_site / "idna" / "__init__.py").is_file()
    assert resolver.base_for_lock(resolved.lock_sha256) == resolved
