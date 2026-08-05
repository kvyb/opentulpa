"""Trusted dependency resolution into sealed, content-addressed runtime inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tomllib
import zipfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from packaging.utils import canonicalize_name

from opentulpa.evolution.generation import canonical_json_bytes
from opentulpa.evolution.process import BoundedProcessResult, run_bounded_process

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_PACKAGE_VALUE_RE = re.compile(r"[^\x00\r\n]{1,2000}\Z")
_WHEEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,299}\.whl\Z")
_VOLUME_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RESOLVER_VERSION = "oci-uv-wheelhouse-v3"
_NORMALIZE_SITE_SCRIPT = (
    "import pathlib,sys;"
    "root=pathlib.Path(sys.argv[1]);"
    "targets=(*root.glob('*.dist-info/RECORD'),*root.glob('*.dist-info/uv_cache.json'));"
    "[path.unlink() for path in targets]"
)


class DependencyResolutionError(RuntimeError):
    """Sanitized resolver failure safe to return through the evolution API."""


class DependencyResolver(Protocol):
    async def resolve(self, workspace: Path) -> ResolvedDependencyBase: ...

    def base_for_lock(self, lock_sha256: str) -> ResolvedDependencyBase | None: ...


class ResolverCommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        max_output_bytes: int,
        timeout_cleanup: Callable[[], None] | None = None,
    ) -> BoundedProcessResult: ...


def _run_resolver_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_cleanup: Callable[[], None] | None = None,
) -> BoundedProcessResult:
    return run_bounded_process(
        argv,
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        timeout_cleanup=timeout_cleanup,
    )


@dataclass(frozen=True, slots=True)
class DependencyResolverPolicy:
    """Immutable resolver image and bounded publication policy."""

    bases_root: Path
    state_root: Path
    trusted_pyproject: Path
    trusted_lock: Path
    resolver_image_digest: str
    package_index: str = "https://pypi.org/simple"
    container_cli: str = "docker"
    container_volume_name: str | None = None
    container_volume_root: Path | None = None
    uv_executable: str = "/usr/local/bin/uv"
    python_executable: str = "/usr/local/bin/python3"
    python_version: str = "3.12"
    extras: tuple[str, ...] = ("evaluation",)
    uid: int = 65_533
    gid: int = 65_533
    timeout_seconds: int = 1_800
    max_output_bytes: int = 1_000_000
    max_lock_bytes: int = 20 * 1024 * 1024
    max_wheelhouse_bytes: int = 1024 * 1024 * 1024
    max_site_bytes: int = 2 * 1024 * 1024 * 1024
    max_site_entries: int = 200_000
    max_wheels: int = 2_000

    def __post_init__(self) -> None:
        if not _DIGEST_RE.fullmatch(self.resolver_image_digest):
            raise ValueError("resolver image must be an immutable local OCI image ID")
        if Path(self.container_cli).name not in {"docker", "podman"}:
            raise ValueError("resolver container CLI is invalid")
        if (self.container_volume_name is None) != (self.container_volume_root is None):
            raise ValueError("resolver container volume configuration is incomplete")
        if self.container_volume_name is not None:
            if _VOLUME_NAME_RE.fullmatch(self.container_volume_name) is None:
                raise ValueError("resolver container volume name is invalid")
            volume_root = self.container_volume_root
            assert volume_root is not None
            volume_root = volume_root.expanduser().absolute()
            for path in (self.bases_root, self.state_root):
                try:
                    path.expanduser().absolute().relative_to(volume_root)
                except ValueError as exc:
                    raise ValueError("resolver paths must be inside its container volume") from exc
            object.__setattr__(self, "container_volume_root", volume_root)
        for executable in (self.uv_executable, self.python_executable):
            if not executable.startswith("/") or "\x00" in executable:
                raise ValueError("resolver executable is invalid")
        parsed = urlsplit(self.package_index)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.rstrip("/").endswith("/simple")
        ):
            raise ValueError("resolver package index must be a public HTTPS simple index")
        object.__setattr__(self, "package_index", self.package_index.rstrip("/"))
        canonical_extras = tuple(sorted(set(self.extras)))
        if canonical_extras != self.extras or any(
            re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", extra) is None
            for extra in self.extras
        ):
            raise ValueError("resolver extras must be sorted, unique canonical names")
        if self.uid < 1 or self.gid < 1 or self.uid == 65_532 or self.gid == 65_532:
            raise ValueError("resolver identity is invalid")
        if not 60 <= self.timeout_seconds <= 86_400:
            raise ValueError("resolver timeout is invalid")
        if self.max_output_bytes < 1_024 or self.max_lock_bytes < 1_024:
            raise ValueError("resolver output limits are invalid")
        if self.max_wheelhouse_bytes < 1024 * 1024 or not 1 <= self.max_wheels <= 10_000:
            raise ValueError("resolver wheelhouse limits are invalid")
        if self.max_site_bytes < 1024 * 1024 or self.max_site_entries < 100:
            raise ValueError("resolver dependency site limits are invalid")


@dataclass(frozen=True, slots=True)
class ResolvedDependencyBase:
    """Verified immutable inputs used to evaluate and build one dependency lock."""

    id: str
    root: Path
    lock_sha256: str
    requirements_sha256: str
    wheelhouse_sha256: str
    inventory_sha256: str
    pyproject_sha256: str
    site_sha256: str
    resolver_fingerprint: str

    @property
    def lock_path(self) -> Path:
        return self.root / "uv.lock"

    @property
    def requirements_path(self) -> Path:
        return self.root / "requirements.txt"

    @property
    def wheelhouse(self) -> Path:
        return self.root / "wheelhouse"

    @property
    def dependency_site(self) -> Path:
        return self.root / "site"


class TrustedDependencyResolver:
    """Resolve dependency-only metadata in a credential-free fixed-command OCI worker."""

    def __init__(
        self,
        *,
        policy: DependencyResolverPolicy,
        runner: ResolverCommandRunner = _run_resolver_process,
    ) -> None:
        self._policy = policy
        self._runner = runner
        self._bases_root = self._secure_directory(policy.bases_root, create=True, mode=0o711)
        self._state_root = self._secure_directory(policy.state_root, create=True, mode=0o700)
        if self._bases_root == self._state_root or self._is_relative_to(
            self._state_root, self._bases_root
        ):
            raise ValueError("resolver state must be outside dependency bases")
        self._trusted_pyproject = self._regular_file(policy.trusted_pyproject, "trusted pyproject")
        self._trusted_lock = self._regular_file(policy.trusted_lock, "trusted dependency lock")
        try:
            self._trusted_document = tomllib.loads(
                self._trusted_pyproject.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError("trusted pyproject is invalid") from exc
        self._fingerprint = self._resolver_fingerprint()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def resolve(self, workspace: Path) -> ResolvedDependencyBase:
        import asyncio

        return await asyncio.to_thread(self._resolve, workspace)

    def base_for_lock(self, lock_sha256: str) -> ResolvedDependencyBase | None:
        if not _HASH_RE.fullmatch(str(lock_sha256 or "")):
            raise ValueError("dependency lock hash is invalid")
        matches: list[ResolvedDependencyBase] = []
        for path in self._bases_root.iterdir():
            if not _HASH_RE.fullmatch(path.name):
                continue
            try:
                base = self._open_existing(path)
            except DependencyResolutionError:
                continue
            if base.lock_sha256 == lock_sha256:
                matches.append(base)
        if len(matches) > 1:
            raise DependencyResolutionError("dependency lock resolves to ambiguous bases")
        return matches[0] if matches else None

    def _resolve(self, workspace: Path) -> ResolvedDependencyBase:
        source_root = workspace.expanduser()
        if source_root.is_symlink():
            raise DependencyResolutionError("dependency proposal workspace is invalid")
        try:
            source_root = source_root.resolve(strict=True)
        except OSError as exc:
            raise DependencyResolutionError("dependency proposal workspace is invalid") from exc
        pyproject = self._regular_file(source_root / "pyproject.toml", "candidate pyproject")
        pyproject_bytes = pyproject.read_bytes()
        if len(pyproject_bytes) > self._policy.max_lock_bytes:
            raise DependencyResolutionError("candidate pyproject exceeds its byte limit")
        trusted_lock_bytes = self._trusted_lock.read_bytes()
        if not trusted_lock_bytes or len(trusted_lock_bytes) > self._policy.max_lock_bytes:
            raise DependencyResolutionError("trusted dependency lock exceeds its byte limit")
        document = self._validated_dependency_proposal(pyproject_bytes)
        staging = self._state_root / f"resolve-{uuid4().hex}"
        staging.mkdir(mode=0o700)
        try:
            (staging / "wheelhouse").mkdir(mode=0o700)
            self._write_file(staging / "pyproject.toml", pyproject_bytes)
            self._write_file(staging / "uv.lock", trusted_lock_bytes)
            self._assign_worker(staging)
            self._verify_resolver_image()
            self._run_resolver_command(
                staging,
                (
                    self._policy.uv_executable,
                    "lock",
                    "--no-sources",
                    "--index",
                    self._policy.package_index,
                    "--python",
                    self._policy.python_version,
                ),
                label="dependency lock resolution failed",
            )
            export = [
                self._policy.uv_executable,
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--no-header",
                "--output-file",
                "/resolution/requirements.txt",
            ]
            for extra in self._policy.extras:
                export.extend(("--extra", extra))
            self._run_resolver_command(
                staging,
                tuple(export),
                label="dependency requirement export failed",
            )
            self._run_resolver_command(
                staging,
                (
                    self._policy.python_executable,
                    "-I",
                    "-m",
                    "pip",
                    "download",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--index-url",
                    self._policy.package_index,
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--dest",
                    "/resolution/wheelhouse",
                    "--requirement",
                    "/resolution/requirements.txt",
                ),
                label="dependency wheel download failed",
            )
            self._run_resolver_command(
                staging,
                (
                    self._policy.uv_executable,
                    "pip",
                    "install",
                    "--target",
                    "/resolution/site",
                    "--offline",
                    "--no-index",
                    "--find-links",
                    "/resolution/wheelhouse",
                    "--link-mode=copy",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--strict",
                    "--requirements",
                    "/resolution/requirements.txt",
                ),
                label="dependency evaluation site installation failed",
            )
            self._run_resolver_command(
                staging,
                (
                    self._policy.python_executable,
                    "-I",
                    "-c",
                    _NORMALIZE_SITE_SCRIPT,
                    "/resolution/site",
                ),
                label="dependency evaluation site normalization failed",
            )
            return self._publish(
                staging,
                pyproject_sha256=hashlib.sha256(pyproject_bytes).hexdigest(),
                project_name=self._project_name(document),
            )
        except DependencyResolutionError:
            raise
        except (OSError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
            raise DependencyResolutionError("dependency resolution output was invalid") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _validated_dependency_proposal(self, raw: bytes) -> dict[str, Any]:
        try:
            document = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise DependencyResolutionError("candidate pyproject is invalid") from exc
        trusted = deepcopy(self._trusted_document)
        proposed = deepcopy(document)
        self._validate_dependency_values(proposed)
        for value in (trusted, proposed):
            project = value.get("project")
            if not isinstance(project, dict):
                raise DependencyResolutionError("candidate project metadata is incomplete")
            project.pop("dependencies", None)
            project.pop("optional-dependencies", None)
        if proposed != trusted:
            raise DependencyResolutionError(
                "dependency resolver accepts only project dependency field changes"
            )
        return document

    def _validate_dependency_values(self, document: Mapping[str, Any]) -> None:
        project = document.get("project")
        if not isinstance(project, dict):
            raise DependencyResolutionError("candidate project metadata is incomplete")
        groups: list[Any] = [project.get("dependencies", [])]
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            raise DependencyResolutionError("candidate optional dependencies are invalid")
        groups.extend(optional.values())
        for dependencies in groups:
            if not isinstance(dependencies, list) or any(
                not isinstance(item, str)
                or _PACKAGE_VALUE_RE.fullmatch(item) is None
                or "@" in item
                or "://" in item
                for item in dependencies
            ):
                raise DependencyResolutionError(
                    "dependency proposal contains a direct or invalid package reference"
                )
        tool = document.get("tool", {})
        uv = tool.get("uv", {}) if isinstance(tool, dict) else {}
        if isinstance(uv, dict) and any(
            key in uv
            for key in (
                "sources",
                "index",
                "extra-index-url",
                "find-links",
                "allow-insecure-host",
                "keyring-provider",
                "workspace",
            )
        ):
            raise DependencyResolutionError("candidate package source configuration is forbidden")

    def _verify_resolver_image(self) -> None:
        result = self._runner(
            (
                self._policy.container_cli,
                "image",
                "inspect",
                self._policy.resolver_image_digest,
                "--format",
                "{{json .}}",
            ),
            cwd=self._state_root,
            timeout_seconds=30,
            max_output_bytes=256 * 1024,
        )
        try:
            payload: Any = json.loads(result.output.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DependencyResolutionError("resolver image inspection failed") from exc
        config = payload.get("Config") if isinstance(payload, dict) else None
        image_environment = config.get("Env") if isinstance(config, dict) else None
        if (
            result.returncode != 0
            or result.timed_out
            or result.truncated
            or not isinstance(payload, dict)
            or str(payload.get("Id") or "").lower() != self._policy.resolver_image_digest
            or not isinstance(image_environment, list)
            or any(
                not isinstance(value, str)
                or (
                    bool(value.partition("=")[2])
                    and (
                        any(
                            marker in value.partition("=")[0].casefold()
                            for marker in (
                                "token",
                                "password",
                                "secret",
                                "credential",
                                "api_key",
                            )
                        )
                        or value.partition("=")[0].casefold()
                        in {
                            "pip_index_url",
                            "pip_extra_index_url",
                            "pip_trusted_host",
                            "uv_index",
                            "uv_default_index",
                            "uv_insecure_host",
                        }
                    )
                )
                for value in image_environment
            )
        ):
            raise DependencyResolutionError("resolver image identity is unavailable")

    def _run_resolver_command(self, staging: Path, command: tuple[str, ...], *, label: str) -> None:
        name = f"opentulpa-resolver-{uuid4().hex}"
        mount = f"type=bind,src={staging},dst=/resolution"
        if self._policy.container_volume_name is not None:
            volume_root = self._policy.container_volume_root
            assert volume_root is not None
            try:
                subpath = staging.relative_to(volume_root).as_posix()
            except ValueError as exc:
                raise DependencyResolutionError("resolver staging path escaped its volume") from exc
            mount = (
                f"type=volume,src={self._policy.container_volume_name},dst=/resolution,"
                f"volume-subpath={subpath}"
            )
        argv = (
            self._policy.container_cli,
            "run",
            "--rm",
            "--name",
            name,
            "--pull=never",
            "--network=bridge",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--pids-limit=256",
            "--memory=2g",
            "--memory-swap=2g",
            f"--user={self._policy.uid}:{self._policy.gid}",
            "--env=HOME=/tmp",
            "--env=PIP_CONFIG_FILE=/dev/null",
            f"--env=UV_DEFAULT_INDEX={self._policy.package_index}",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=512m,mode=1777",
            "--mount",
            mount,
            "--workdir=/resolution",
            f"--entrypoint={command[0]}",
            self._policy.resolver_image_digest,
            *command[1:],
        )
        result = self._runner(
            argv,
            cwd=self._state_root,
            timeout_seconds=self._policy.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            timeout_cleanup=lambda: self._remove_container(name),
        )
        if result.returncode != 0 or result.timed_out or result.truncated:
            raise DependencyResolutionError(label)

    def _remove_container(self, name: str) -> None:
        try:
            subprocess.run(
                (self._policy.container_cli, "rm", "--force", name),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return

    def _publish(
        self,
        staging: Path,
        *,
        pyproject_sha256: str,
        project_name: str,
    ) -> ResolvedDependencyBase:
        lock = self._output_file(staging / "uv.lock", "resolved dependency lock")
        requirements = self._output_file(
            staging / "requirements.txt", "resolved dependency requirements"
        )
        lock_bytes = lock.read_bytes()
        requirements_bytes = requirements.read_bytes()
        if not lock_bytes or len(lock_bytes) > self._policy.max_lock_bytes:
            raise DependencyResolutionError("resolved dependency lock exceeds its byte limit")
        if not requirements_bytes or len(requirements_bytes) > self._policy.max_lock_bytes:
            raise DependencyResolutionError("resolved requirements exceed their byte limit")
        inventory, wheels = self._wheel_inventory(staging / "wheelhouse")
        required_packages = self._validate_requirements(requirements_bytes)
        inventory_packages = {canonicalize_name(str(item["name"])) for item in inventory}
        if (
            len(inventory_packages) != len(inventory)
            or not inventory_packages
            or not inventory_packages.issubset(required_packages)
        ):
            raise DependencyResolutionError("resolved wheelhouse does not match requirements")
        self._validate_lock(
            lock_bytes,
            project_name=project_name,
            required_packages=required_packages,
        )
        inventory_bytes = canonical_json_bytes(inventory)
        wheelhouse_sha256 = self._wheelhouse_digest(wheels)
        site_sha256 = self._site_digest(staging / "site")
        identity: dict[str, Any] = {
            "format_version": 1,
            "resolver_version": _RESOLVER_VERSION,
            "resolver_fingerprint": self._fingerprint,
            "resolver_image_digest": self._policy.resolver_image_digest,
            "package_index": self._policy.package_index,
            "python_version": self._policy.python_version,
            "extras": list(self._policy.extras),
            "pyproject_sha256": pyproject_sha256,
            "project_name": project_name,
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "requirements_sha256": hashlib.sha256(requirements_bytes).hexdigest(),
            "wheelhouse_sha256": wheelhouse_sha256,
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "site_sha256": site_sha256,
        }
        base_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        manifest = {**identity, "base_id": base_id}
        destination = self._bases_root / base_id
        if destination.exists():
            return self._open_existing(destination, manifest)
        publication = self._bases_root / f".publish-{uuid4().hex}"
        publication.mkdir(mode=0o700)
        try:
            wheelhouse = publication / "wheelhouse"
            wheelhouse.mkdir(mode=0o700)
            self._copy_file(lock, publication / "uv.lock")
            self._copy_file(requirements, publication / "requirements.txt")
            for source in wheels:
                self._copy_file(source, wheelhouse / source.name)
            shutil.copytree(staging / "site", publication / "site", symlinks=False)
            self._write_file(publication / "inventory.json", inventory_bytes)
            self._write_file(publication / "manifest.json", canonical_json_bytes(manifest))
            self._write_file(publication / "COMPLETE", b"")
            self._seal(publication)
            try:
                publication.rename(destination)
            except FileExistsError:
                return self._open_existing(destination, manifest)
            publication = Path()
            return self._open_existing(destination, manifest)
        finally:
            if publication != Path():
                shutil.rmtree(publication, ignore_errors=True)

    def _open_existing(
        self,
        root: Path,
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> ResolvedDependencyBase:
        root_metadata = root.lstat()
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) & 0o222
        ):
            raise DependencyResolutionError("resolved dependency base is mutable")
        manifest_path = self._sealed_file(root / "manifest.json", "dependency base manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DependencyResolutionError("dependency base manifest is invalid") from exc
        if (
            (expected_manifest is not None and manifest != dict(expected_manifest))
            or not self._sealed_file(root / "COMPLETE", "dependency base marker").is_file()
            or manifest.get("base_id") != root.name
            or manifest.get("resolver_fingerprint") != self._fingerprint
        ):
            raise DependencyResolutionError("dependency base identity changed")
        lock = self._sealed_file(root / "uv.lock", "dependency base lock")
        requirements = self._sealed_file(
            root / "requirements.txt", "dependency base requirements"
        )
        inventory_path = self._sealed_file(
            root / "inventory.json", "dependency base inventory"
        )
        lock_bytes = lock.read_bytes()
        requirements_bytes = requirements.read_bytes()
        inventory, wheels = self._wheel_inventory(root / "wheelhouse", sealed=True)
        required_packages = self._validate_requirements(requirements_bytes)
        inventory_packages = {canonicalize_name(str(item["name"])) for item in inventory}
        if (
            len(inventory_packages) != len(inventory)
            or not inventory_packages
            or not inventory_packages.issubset(required_packages)
        ):
            raise DependencyResolutionError("resolved wheelhouse does not match requirements")
        self._validate_lock(
            lock_bytes,
            project_name=str(manifest.get("project_name") or ""),
            required_packages=required_packages,
        )
        inventory_bytes = canonical_json_bytes(inventory)
        if (
            self._file_sha256(lock) != manifest.get("lock_sha256")
            or self._file_sha256(requirements) != manifest.get("requirements_sha256")
            or self._file_sha256(inventory_path) != manifest.get("inventory_sha256")
            or inventory_path.read_bytes() != inventory_bytes
            or self._wheelhouse_digest(wheels) != manifest.get("wheelhouse_sha256")
            or self._site_digest(root / "site", sealed=True) != manifest.get("site_sha256")
        ):
            raise DependencyResolutionError("dependency base content changed")
        return ResolvedDependencyBase(
            id=str(manifest["base_id"]),
            root=root,
            lock_sha256=str(manifest["lock_sha256"]),
            requirements_sha256=str(manifest["requirements_sha256"]),
            wheelhouse_sha256=str(manifest["wheelhouse_sha256"]),
            inventory_sha256=str(manifest["inventory_sha256"]),
            pyproject_sha256=str(manifest["pyproject_sha256"]),
            site_sha256=str(manifest["site_sha256"]),
            resolver_fingerprint=str(manifest["resolver_fingerprint"]),
        )

    def _validate_lock(
        self,
        raw: bytes,
        *,
        project_name: str,
        required_packages: set[str],
    ) -> None:
        try:
            document = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise DependencyResolutionError("resolved dependency lock is invalid") from exc
        packages = document.get("package")
        if document.get("version") != 1 or not isinstance(packages, list) or not packages:
            raise DependencyResolutionError("resolved dependency lock is incomplete")
        locked_packages: set[str] = set()
        for package in packages:
            if not isinstance(package, dict):
                raise DependencyResolutionError("resolved dependency lock package is invalid")
            source = package.get("source")
            name = str(package.get("name") or "")
            if not isinstance(source, dict):
                raise DependencyResolutionError("resolved dependency source is invalid")
            if name == project_name and source.get("editable") == ".":
                continue
            if source != {"registry": self._policy.package_index}:
                raise DependencyResolutionError("resolved dependency escaped the trusted index")
            normalized_name = canonicalize_name(name)
            locked_packages.add(normalized_name)
            sdist = package.get("sdist")
            if sdist is not None:
                self._validate_locked_artifact(sdist, label="source archive")
            wheels = package.get("wheels")
            if normalized_name in required_packages and (not isinstance(wheels, list) or not wheels):
                raise DependencyResolutionError("resolved dependency has no binary wheel")
            if wheels is None:
                continue
            if not isinstance(wheels, list):
                raise DependencyResolutionError("resolved dependency wheels are invalid")
            for wheel in wheels:
                self._validate_locked_artifact(wheel, label="wheel")
        if not required_packages.issubset(locked_packages):
            raise DependencyResolutionError("resolved requirements are absent from the lock")

    def _validate_locked_artifact(self, artifact: Any, *, label: str) -> None:
        if not isinstance(artifact, dict):
            raise DependencyResolutionError(f"resolved dependency {label} is invalid")
        parsed = urlsplit(str(artifact.get("url") or ""))
        digest = str(artifact.get("hash") or "")
        if (
            parsed.scheme != "https"
            or parsed.hostname
            not in {"files.pythonhosted.org", urlsplit(self._policy.package_index).hostname}
            or not _DIGEST_RE.fullmatch(digest)
        ):
            raise DependencyResolutionError(f"resolved dependency {label} source is invalid")

    @staticmethod
    def _validate_requirements(raw: bytes) -> set[str]:
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise DependencyResolutionError("resolved requirements are invalid") from exc
        lowered = text.casefold()
        if any(
            token in lowered
            for token in ("http://", "https://", "file:", "--index", "--find-links", "-e ")
        ) or "--hash=sha256:" not in lowered:
            raise DependencyResolutionError("resolved requirements are not hash-locked registry inputs")
        packages: set[str] = set()
        for line in text.splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value.startswith("--hash=sha256:"):
                digest = value.removeprefix("--hash=sha256:").removesuffix(" \\").strip()
                if not _HASH_RE.fullmatch(digest):
                    raise DependencyResolutionError("resolved requirement hash is invalid")
                continue
            match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)==[^\s;\\]+(?:\s|;|\\|$)", value)
            if match is None:
                raise DependencyResolutionError("resolved requirement entry is invalid")
            packages.add(canonicalize_name(match.group(1)))
        if not packages:
            raise DependencyResolutionError("resolved requirements are empty")
        return packages

    def _wheel_inventory(
        self,
        wheelhouse: Path,
        *,
        sealed: bool = False,
    ) -> tuple[list[dict[str, Any]], tuple[Path, ...]]:
        if wheelhouse.is_symlink() or not wheelhouse.is_dir():
            raise DependencyResolutionError("resolved wheelhouse is invalid")
        root_metadata = wheelhouse.lstat()
        if sealed and (
            root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) & 0o222
        ):
            raise DependencyResolutionError("resolved wheelhouse is mutable")
        inventory: list[dict[str, Any]] = []
        wheels: list[Path] = []
        total = 0
        for path in sorted(wheelhouse.iterdir()):
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or _WHEEL_NAME_RE.fullmatch(path.name) is None
                or (
                    sealed
                    and (
                        metadata.st_uid != os.geteuid()
                        or stat.S_IMODE(metadata.st_mode) & 0o222
                    )
                )
            ):
                raise DependencyResolutionError("resolved wheelhouse contains an unsafe entry")
            total += metadata.st_size
            wheels.append(path)
            if total > self._policy.max_wheelhouse_bytes or len(wheels) > self._policy.max_wheels:
                raise DependencyResolutionError("resolved wheelhouse exceeds its limits")
            with zipfile.ZipFile(path) as archive:
                metadata_names = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA") and name.count("/") == 1
                ]
                if len(metadata_names) != 1:
                    raise DependencyResolutionError("resolved wheel metadata is ambiguous")
                metadata_info = archive.getinfo(metadata_names[0])
                if metadata_info.file_size > 1024 * 1024:
                    raise DependencyResolutionError("resolved wheel metadata exceeds its limit")
                package = BytesParser().parsebytes(archive.read(metadata_info))
            name = str(package.get("Name") or "").strip()
            version = str(package.get("Version") or "").strip()
            if not name or not version or not _PACKAGE_VALUE_RE.fullmatch(name + version):
                raise DependencyResolutionError("resolved wheel metadata is incomplete")
            inventory.append(
                {
                    "filename": path.name,
                    "name": name,
                    "sha256": self._file_sha256(path),
                    "size": metadata.st_size,
                    "version": version,
                }
            )
        if not wheels:
            raise DependencyResolutionError("resolved wheelhouse is empty")
        return inventory, tuple(wheels)

    @staticmethod
    def _wheelhouse_digest(wheels: tuple[Path, ...]) -> str:
        values = {path.name: TrustedDependencyResolver._file_sha256(path) for path in wheels}
        return hashlib.sha256(canonical_json_bytes(values)).hexdigest()

    def _site_digest(self, root: Path, *, sealed: bool = False) -> str:
        if root.is_symlink() or not root.is_dir():
            raise DependencyResolutionError("resolved dependency site is invalid")
        root_metadata = root.lstat()
        if sealed and (
            root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) & 0o222
        ):
            raise DependencyResolutionError("resolved dependency site is mutable")
        entries: list[dict[str, Any]] = []
        total = 0
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_names.sort()
            for name in (*directory_names, *sorted(file_names)):
                path = Path(directory) / name
                metadata = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISLNK(metadata.st_mode) or (
                    sealed
                    and (
                        metadata.st_uid != os.geteuid()
                        or stat.S_IMODE(metadata.st_mode) & 0o222
                    )
                ):
                    raise DependencyResolutionError("resolved dependency site contains a link")
                if stat.S_ISDIR(metadata.st_mode):
                    entries.append({"kind": "directory", "path": relative})
                else:
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise DependencyResolutionError(
                            "resolved dependency site contains an unsafe entry"
                        )
                    total += metadata.st_size
                    entries.append(
                        {
                            "kind": "file",
                            "path": relative,
                            "sha256": self._file_sha256(path),
                            "size": metadata.st_size,
                        }
                    )
                    if total > self._policy.max_site_bytes:
                        raise DependencyResolutionError(
                            "resolved dependency site exceeds its byte limit"
                        )
                if len(entries) > self._policy.max_site_entries:
                    raise DependencyResolutionError("resolved dependency site has too many entries")
        if not entries:
            raise DependencyResolutionError("resolved dependency site is empty")
        return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()

    def _resolver_fingerprint(self) -> str:
        payload = {
            "version": _RESOLVER_VERSION,
            "image": self._policy.resolver_image_digest,
            "index": self._policy.package_index,
            "python": self._policy.python_version,
            "extras": list(self._policy.extras),
            "trusted_pyproject_sha256": self._file_sha256(self._trusted_pyproject),
            "trusted_lock_sha256": self._file_sha256(self._trusted_lock),
        }
        return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"

    @staticmethod
    def _project_name(document: Mapping[str, Any]) -> str:
        project = document.get("project")
        name = str(project.get("name") or "") if isinstance(project, dict) else ""
        if not name:
            raise DependencyResolutionError("candidate project name is unavailable")
        return re.sub(r"[-_.]+", "-", name).casefold()

    @staticmethod
    def _secure_directory(path: Path, *, create: bool, mode: int) -> Path:
        candidate = path.expanduser().absolute()
        if create:
            candidate.mkdir(parents=True, exist_ok=True, mode=mode)
        metadata = candidate.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError("dependency resolver control directory is unsafe")
        candidate.chmod(mode)
        return candidate

    @staticmethod
    def _regular_file(path: Path, label: str) -> Path:
        candidate = path.expanduser().absolute()
        metadata = candidate.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o022
        ):
            raise ValueError(f"{label} is unsafe")
        return candidate

    @staticmethod
    def _output_file(path: Path, label: str) -> Path:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise DependencyResolutionError(f"{label} is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DependencyResolutionError(f"{label} is unsafe")
        return path

    @staticmethod
    def _sealed_file(path: Path, label: str) -> Path:
        candidate = TrustedDependencyResolver._output_file(path, label)
        metadata = candidate.lstat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o222:
            raise DependencyResolutionError(f"{label} is mutable")
        return candidate

    def _assign_worker(self, root: Path) -> None:
        if os.geteuid() != 0:
            return
        for directory, directory_names, file_names in os.walk(root):
            os.chown(directory, self._policy.uid, self._policy.gid)
            for name in (*directory_names, *file_names):
                os.chown(Path(directory) / name, self._policy.uid, self._policy.gid)

    @staticmethod
    def _seal(root: Path) -> None:
        if os.geteuid() == 0:
            for directory, directory_names, file_names in os.walk(root):
                os.chown(directory, 0, 0)
                for name in (*directory_names, *file_names):
                    os.chown(Path(directory) / name, 0, 0)
        for directory, _, file_names in os.walk(root, topdown=False):
            for name in file_names:
                (Path(directory) / name).chmod(0o444)
            Path(directory).chmod(0o555)

    @staticmethod
    def _write_file(path: Path, content: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)

    @staticmethod
    def _copy_file(source: Path, destination: Path) -> None:
        TrustedDependencyResolver._write_file(destination, source.read_bytes())

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True


__all__ = [
    "DependencyResolutionError",
    "DependencyResolver",
    "DependencyResolverPolicy",
    "ResolvedDependencyBase",
    "TrustedDependencyResolver",
]
