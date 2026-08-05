"""Tenant-owned repository workspace lifecycle and publishing."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import httpx
from deepagents.backends.protocol import SandboxBackendProtocol

from opentulpa.core.ids import new_short_id
from opentulpa.repositories.models import (
    RepositoryWorkspace,
    RepositoryWorkspaceStatus,
    utc_now,
)
from opentulpa.repositories.providers import (
    RepositoryProviderRegistry,
    RepositorySandboxError,
)
from opentulpa.repositories.store import RepositoryWorkspaceStore

_GIT_REF_RE = re.compile(r"^(?![./])(?!.*(?:\.\.|//|@\{|\\))[\w./-]{1,250}(?<![./])$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_IDENTITY_RE = re.compile(r"^(.*) <([^<>]+)> ([0-9]+) ([+-][0-9]{4})$")
_MAX_STATUS_LINES = 100
_MAX_PUBLISH_FILES = 200
_MAX_PUBLISH_BYTES = 50 * 1024 * 1024
_StringList = list[str]
_GitTreePayload = list[dict[str, Any]]


class GitHubAPIProxy(Protocol):
    def request(
        self,
        *,
        tenant_id: str,
        method: Literal["GET", "POST", "PATCH", "DELETE"],
        endpoint: str,
        body: object | None = None,
    ) -> tuple[int, Any]: ...


@dataclass(frozen=True, slots=True)
class _GitTreeEntry:
    mode: str
    type: str
    sha: str


class RepositoryWorkspaceError(RuntimeError):
    """Sanitized repository workspace error."""


class RepositoryWorkspaceNotFoundError(RepositoryWorkspaceError):
    pass


class RepositoryWorkspaceConflictError(RepositoryWorkspaceError):
    pass


class RepositoryPublishError(RepositoryWorkspaceError):
    pass


class RepositoryWorkspaceService:
    """Own isolated checkouts and publish verified commits without model-sized payloads."""

    def __init__(
        self,
        *,
        store: RepositoryWorkspaceStore,
        providers: RepositoryProviderRegistry,
        github_token_resolver: Callable[[str, str], str | None],
        github_api_proxy: GitHubAPIProxy | None = None,
        http_client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self._store = store
        self._providers = providers
        self._github_token_resolver = github_token_resolver
        self._github_api_proxy = github_api_proxy
        self._http_client_factory = http_client_factory or (
            lambda: httpx.Client(timeout=httpx.Timeout(30.0))
        )
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _normalize_repository_url(value: str) -> tuple[str, str, str]:
        raw = str(value or "").strip()
        parsed = urlsplit(raw)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in {"github.com", "www.github.com"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise RepositoryWorkspaceError(
                "repository_url must be an HTTPS github.com repository URL"
            )
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 2:
            raise RepositoryWorkspaceError("repository_url must identify owner/repository")
        owner = parts[0]
        repository = parts[1].removesuffix(".git")
        if not owner or not repository or repository in {".", ".."}:
            raise RepositoryWorkspaceError("repository_url is invalid")
        normalized = urlunsplit(("https", "github.com", f"/{owner}/{repository}.git", "", ""))
        return normalized, owner, repository

    @staticmethod
    def _ref(value: str, *, field: str) -> str:
        ref = str(value or "").strip()
        if not _GIT_REF_RE.fullmatch(ref) or ref.endswith(".lock"):
            raise RepositoryWorkspaceError(f"{field} is not a valid Git ref")
        return ref

    def _lock(self, workspace_id: str) -> asyncio.Lock:
        return self._locks.setdefault(workspace_id, asyncio.Lock())

    def _owned(self, *, tenant_id: str, workspace_id: str) -> RepositoryWorkspace:
        workspace = self._store.get(tenant_id=tenant_id, workspace_id=workspace_id)
        if workspace is None:
            raise RepositoryWorkspaceNotFoundError("repository workspace not found")
        return workspace

    async def open(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        repository_url: str,
        base_ref: str = "main",
        branch: str | None = None,
        provider: str | None = None,
    ) -> RepositoryWorkspace:
        normalized_url, _, _ = self._normalize_repository_url(repository_url)
        normalized_base = self._ref(base_ref, field="base_ref")
        workspace_id = new_short_id("repo", suffix_chars=10)
        normalized_branch = self._ref(
            branch or f"opentulpa/{workspace_id}",
            field="branch",
        )
        selected = self._providers.select(tenant_id=tenant_id, requested=provider)
        now = utc_now()
        workspace = RepositoryWorkspace(
            id=workspace_id,
            tenant_id=tenant_id,
            repository_url=normalized_url,
            provider=selected.name,
            base_ref=normalized_base,
            branch=normalized_branch,
            status=RepositoryWorkspaceStatus.CREATING,
            created_at=now,
            updated_at=now,
            last_used_at=now,
        )
        self._store.create(workspace)
        github_token = self._github_token_resolver(tenant_id, "github.read")
        try:
            provider_id = await asyncio.to_thread(
                selected.create,
                workspace,
                github_token=github_token,
            )
            workspace = workspace.model_copy(update={"provider_workspace_id": provider_id})
            backend = await asyncio.to_thread(selected.backend, workspace)
            await self._run(
                backend,
                "git checkout -b "
                f"{self._shell_quote(normalized_branch)} && "
                "git config user.name OpenTulpa && "
                "git config user.email opentulpa@localhost && "
                f"git remote set-url origin {self._shell_quote(normalized_url)} && "
                "(git config --local --unset-all credential.helper >/dev/null 2>&1 || true) && "
                'rm -f "$HOME/.git-credentials"',
                timeout=120,
            )
            base_sha = await self._git_sha(backend, "HEAD")
            updated = utc_now()
            workspace = workspace.model_copy(
                update={
                    "base_sha": base_sha,
                    "head_sha": base_sha,
                    "status": RepositoryWorkspaceStatus.READY,
                    "updated_at": updated,
                    "last_used_at": updated,
                }
            )
            self._store.update(workspace)
            self._store.bind(
                tenant_id=tenant_id,
                thread_id=thread_id,
                workspace_id=workspace.id,
                bound_at=updated,
            )
            return workspace
        except Exception as exc:
            if workspace.provider_workspace_id:
                with suppress(Exception):
                    await asyncio.to_thread(selected.stop, workspace)
            failed = workspace.model_copy(
                update={
                    "status": RepositoryWorkspaceStatus.FAILED,
                    "last_error": self._public_error(exc),
                    "updated_at": utc_now(),
                    "last_used_at": utc_now(),
                }
            )
            self._store.update(failed)
            if isinstance(exc, RepositoryWorkspaceError | RepositorySandboxError):
                raise
            raise RepositoryWorkspaceError("repository workspace could not be created") from exc

    async def list(
        self,
        *,
        tenant_id: str,
        include_closed: bool = False,
    ) -> list[RepositoryWorkspace]:
        statuses = None
        if not include_closed:
            statuses = (
                RepositoryWorkspaceStatus.CREATING,
                RepositoryWorkspaceStatus.READY,
                RepositoryWorkspaceStatus.PUBLISHED,
            )
        return await asyncio.to_thread(
            self._store.list,
            tenant_id=tenant_id,
            statuses=statuses,
        )

    async def active(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> RepositoryWorkspace | None:
        return await asyncio.to_thread(
            self._store.active,
            tenant_id=tenant_id,
            thread_id=thread_id,
        )

    def backend_for_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> SandboxBackendProtocol | None:
        workspace = self._store.active(tenant_id=tenant_id, thread_id=thread_id)
        if workspace is None or workspace.status not in {
            RepositoryWorkspaceStatus.READY,
            RepositoryWorkspaceStatus.PUBLISHED,
        }:
            return None
        return self._providers.for_workspace(workspace).backend(workspace)

    async def status(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        workspace = (
            self._owned(tenant_id=tenant_id, workspace_id=workspace_id)
            if workspace_id
            else await self.active(tenant_id=tenant_id, thread_id=thread_id)
        )
        if workspace is None:
            raise RepositoryWorkspaceNotFoundError("no repository workspace is active")
        data = workspace.model_dump(mode="json")
        if workspace.status not in {
            RepositoryWorkspaceStatus.READY,
            RepositoryWorkspaceStatus.PUBLISHED,
        }:
            return data
        backend = await asyncio.to_thread(
            self._providers.for_workspace(workspace).backend,
            workspace,
        )
        head_sha = await self._git_sha(backend, "HEAD")
        status = await self._run(backend, "git status --porcelain=v1", timeout=60)
        changes = [line[:500] for line in status.splitlines()[:_MAX_STATUS_LINES]]
        ahead_raw = await self._run(
            backend,
            f"git rev-list --count {workspace.base_sha}..HEAD",
            timeout=60,
        )
        now = utc_now()
        workspace = workspace.model_copy(
            update={"head_sha": head_sha, "updated_at": now, "last_used_at": now}
        )
        self._store.update(workspace)
        data = workspace.model_dump(mode="json")
        data.update(
            {
                "clean": not changes,
                "changes": changes,
                "changes_truncated": len(status.splitlines()) > _MAX_STATUS_LINES,
                "commits_ahead": int(ahead_raw.strip() or "0"),
            }
        )
        return data

    async def import_verified_patch(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str,
        patch: bytes,
        expected_sha256: str,
        message: str,
    ) -> dict[str, Any]:
        expected = str(expected_sha256 or "").strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RepositoryWorkspaceError("expected patch digest is invalid")
        if not patch or len(patch) > 20 * 1024 * 1024 or b"\x00" in patch:
            raise RepositoryWorkspaceError("contribution patch is invalid")
        if hashlib.sha256(patch).hexdigest() != expected:
            raise RepositoryWorkspaceError("contribution patch failed digest validation")
        safe_message = " ".join(str(message or "").split())[:500]
        if not safe_message:
            raise RepositoryWorkspaceError("contribution commit message is required")
        workspace = self._owned(tenant_id=tenant_id, workspace_id=workspace_id)
        if workspace.status is not RepositoryWorkspaceStatus.READY:
            raise RepositoryWorkspaceError("repository workspace is not ready")

        async with self._lock(workspace.id):
            provider = self._providers.for_workspace(workspace)
            backend = await asyncio.to_thread(provider.backend, workspace)
            if (await self._run(backend, "git status --porcelain=v1", timeout=60)).strip():
                raise RepositoryWorkspaceConflictError("repository workspace is not clean")
            if await self._git_sha(backend, "HEAD") != workspace.base_sha:
                raise RepositoryWorkspaceConflictError("repository workspace base changed")
            patch_name = f".opentulpa-contribution-{expected}.patch"
            patch_path = f"/workspace/{patch_name}"
            uploaded = await asyncio.to_thread(
                backend.upload_files,
                [(patch_path, patch)],
            )
            if len(uploaded) != 1 or uploaded[0].error:
                raise RepositoryWorkspaceError("contribution patch could not be transferred")
            try:
                await self._run(
                    backend,
                    "git apply --index --whitespace=error-all -- "
                    f"{self._shell_quote(patch_name)} && "
                    "git diff --cached --check && "
                    f"rm -- {self._shell_quote(patch_name)} && "
                    "git commit --no-gpg-sign --no-verify -m "
                    f"{self._shell_quote(safe_message)}",
                    timeout=300,
                )
            except Exception:
                with suppress(Exception):
                    await self._run(
                        backend,
                        f"rm -f -- {self._shell_quote(patch_name)}",
                        timeout=60,
                    )
                raise
            head_sha = await self._git_sha(backend, "HEAD")
            parent_sha = await self._git_sha(backend, "HEAD^")
            if parent_sha != workspace.base_sha or head_sha == workspace.base_sha:
                raise RepositoryWorkspaceError("contribution commit lineage is invalid")
            now = utc_now()
            workspace = workspace.model_copy(
                update={
                    "head_sha": head_sha,
                    "updated_at": now,
                    "last_used_at": now,
                }
            )
            self._store.update(workspace)
            self._store.bind(
                tenant_id=tenant_id,
                thread_id=thread_id,
                workspace_id=workspace.id,
                bound_at=now,
            )
            return {
                "workspace_id": workspace.id,
                "repository_url": workspace.repository_url.removesuffix(".git"),
                "base_ref": workspace.base_ref,
                "base_sha": workspace.base_sha,
                "branch": workspace.branch,
                "head_sha": head_sha,
                "patch_sha256": expected,
                "clean": True,
            }

    async def close(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str | None = None,
    ) -> RepositoryWorkspace:
        workspace = (
            self._owned(tenant_id=tenant_id, workspace_id=workspace_id)
            if workspace_id
            else await self.active(tenant_id=tenant_id, thread_id=thread_id)
        )
        if workspace is None:
            raise RepositoryWorkspaceNotFoundError("no repository workspace is active")
        async with self._lock(workspace.id):
            await asyncio.to_thread(
                self._providers.for_workspace(workspace).stop,
                workspace,
            )
            now = utc_now()
            workspace = workspace.model_copy(
                update={
                    "status": RepositoryWorkspaceStatus.STOPPED,
                    "updated_at": now,
                    "last_used_at": now,
                }
            )
            self._store.update(workspace)
            self._store.unbind(tenant_id=tenant_id, thread_id=thread_id)
        return workspace

    async def publish(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str | None,
        expected_head_sha: str,
        title: str,
        body: str,
        draft: bool,
    ) -> dict[str, Any]:
        expected = str(expected_head_sha or "").strip().casefold()
        if not _SHA_RE.fullmatch(expected):
            raise RepositoryPublishError("expected_head_sha must be a full Git commit SHA")
        safe_title = str(title or "").strip()
        if not safe_title or len(safe_title) > 256:
            raise RepositoryPublishError("pull request title is invalid")
        safe_body = str(body or "")
        if len(safe_body) > 20_000:
            raise RepositoryPublishError("pull request body is too long")
        workspace = (
            self._owned(tenant_id=tenant_id, workspace_id=workspace_id)
            if workspace_id
            else await self.active(tenant_id=tenant_id, thread_id=thread_id)
        )
        if workspace is None:
            raise RepositoryWorkspaceNotFoundError("no repository workspace is active")
        if workspace.status not in {
            RepositoryWorkspaceStatus.READY,
            RepositoryWorkspaceStatus.PUBLISHED,
        }:
            raise RepositoryPublishError("repository workspace is not ready")

        async with self._lock(workspace.id):
            provider = self._providers.for_workspace(workspace)
            backend = await asyncio.to_thread(provider.backend, workspace)
            head_sha = await self._git_sha(backend, "HEAD")
            if head_sha != expected:
                raise RepositoryWorkspaceConflictError(
                    "repository HEAD changed; inspect status and approve the current commit"
                )
            if head_sha == workspace.base_sha:
                raise RepositoryPublishError("repository workspace has no commits to publish")
            current_branch = (
                await self._run(backend, "git branch --show-current", timeout=60)
            ).strip()
            if current_branch != workspace.branch:
                raise RepositoryPublishError("repository is not on the recorded workspace branch")
            remote_url = (await self._run(backend, "git remote get-url origin", timeout=60)).strip()
            try:
                normalized_remote, _, _ = self._normalize_repository_url(remote_url)
            except RepositoryWorkspaceError as exc:
                raise RepositoryPublishError("repository origin changed") from exc
            if normalized_remote != workspace.repository_url:
                raise RepositoryPublishError("repository origin changed")
            dirty = await self._run(backend, "git status --porcelain=v1", timeout=60)
            if dirty.strip():
                raise RepositoryPublishError(
                    "repository workspace has uncommitted changes; commit and verify them first"
                )
            ancestor = await backend.aexecute(
                f"git merge-base --is-ancestor {workspace.base_sha} HEAD",
                timeout=60,
            )
            if ancestor.exit_code != 0:
                raise RepositoryPublishError("repository branch is not based on the recorded base")
            token = self._github_token_resolver(tenant_id, "github.write")
            if token:
                await asyncio.to_thread(provider.push, workspace, github_token=token)
                pull_request_url = await asyncio.to_thread(
                    self._open_pull_request,
                    workspace,
                    token=token,
                    title=safe_title,
                    body=safe_body,
                    draft=bool(draft),
                )
                credential_source = "token"
            elif self._github_api_proxy is not None:
                pull_request_url = await self._publish_via_github_proxy(
                    tenant_id=tenant_id,
                    backend=backend,
                    workspace=workspace,
                    expected_head_sha=expected,
                    title=safe_title,
                    body=safe_body,
                    draft=bool(draft),
                )
                credential_source = "composio"
            else:
                raise RepositoryPublishError(
                    "GitHub publishing is not configured. Connect GitHub through "
                    "Composio or paste a fine-grained GITHUB_TOKEN assignment."
                )
            now = utc_now()
            workspace = workspace.model_copy(
                update={
                    "head_sha": head_sha,
                    "status": RepositoryWorkspaceStatus.PUBLISHED,
                    "pull_request_url": pull_request_url,
                    "updated_at": now,
                    "last_used_at": now,
                }
            )
            self._store.update(workspace)
            return {
                "workspace_id": workspace.id,
                "repository_url": workspace.repository_url.removesuffix(".git"),
                "base_ref": workspace.base_ref,
                "branch": workspace.branch,
                "head_sha": head_sha,
                "pull_request_url": pull_request_url,
                "draft": bool(draft),
                "credential_source": credential_source,
            }

    async def _publish_via_github_proxy(
        self,
        *,
        tenant_id: str,
        backend: SandboxBackendProtocol,
        workspace: RepositoryWorkspace,
        expected_head_sha: str,
        title: str,
        body: str,
        draft: bool,
    ) -> str:
        base_sha = str(workspace.base_sha or "").strip().casefold()
        if not _SHA_RE.fullmatch(base_sha):
            raise RepositoryPublishError("repository base commit is unavailable")
        commit_count_raw = await self._run(
            backend,
            f"git rev-list --count {self._shell_quote(base_sha)}..HEAD",
            timeout=60,
        )
        try:
            commit_count = int(commit_count_raw.strip())
        except ValueError as exc:
            raise RepositoryPublishError("repository returned an invalid commit count") from exc
        if commit_count != 1:
            raise RepositoryPublishError(
                "Composio publishing currently requires one verified commit; "
                "squash the workspace branch and retry"
            )

        _, owner, repository = self._normalize_repository_url(workspace.repository_url)
        prefix = f"/repos/{quote(owner, safe='')}/{quote(repository, safe='')}"
        changed_paths = await self._changed_paths(
            backend,
            base_sha=base_sha,
        )
        base_tree_sha = await self._git_sha(
            backend,
            f"{base_sha}^{{tree}}",
        )
        head_tree_sha = await self._git_sha(backend, "HEAD^{tree}")
        entries: _GitTreePayload = []
        total_bytes = 0

        for path in changed_paths:
            current = await self._tree_entry(backend, ref="HEAD", path=path)
            previous = await self._tree_entry(
                backend,
                ref=base_sha,
                path=path,
            )
            if current is None:
                if previous is not None:
                    entries.append(
                        {
                            "path": path,
                            "mode": previous.mode,
                            "type": previous.type,
                            "sha": None,
                        }
                    )
                continue
            if current.type == "commit":
                entries.append(
                    {
                        "path": path,
                        "mode": current.mode,
                        "type": current.type,
                        "sha": current.sha,
                    }
                )
                continue
            if current.type != "blob":
                raise RepositoryPublishError("repository contains an unsupported Git object")
            content = await self._blob_content(
                backend,
                path=path,
                entry=current,
            )
            total_bytes += len(content)
            if total_bytes > _MAX_PUBLISH_BYTES:
                raise RepositoryPublishError("repository publish exceeds the byte limit")
            blob_sha = self._git_blob_sha(content)
            if blob_sha != current.sha:
                raise RepositoryPublishError("repository file bytes changed during publishing")
            status, payload = await self._github_proxy_request(
                tenant_id=tenant_id,
                method="POST",
                endpoint=f"{prefix}/git/blobs",
                body={
                    "content": base64.b64encode(content).decode("ascii"),
                    "encoding": "base64",
                },
            )
            if status != 201 or self._response_sha(payload) != blob_sha:
                raise RepositoryPublishError("GitHub did not preserve a repository blob")
            entries.append(
                {
                    "path": path,
                    "mode": current.mode,
                    "type": current.type,
                    "sha": blob_sha,
                }
            )

        if head_tree_sha != base_tree_sha:
            status, payload = await self._github_proxy_request(
                tenant_id=tenant_id,
                method="POST",
                endpoint=f"{prefix}/git/trees",
                body={"base_tree": base_tree_sha, "tree": entries},
            )
            if status != 201 or self._response_sha(payload) != head_tree_sha:
                raise RepositoryPublishError("GitHub did not preserve the repository tree")

        commit = await self._commit_payload(
            backend,
            expected_head_sha=expected_head_sha,
            expected_tree_sha=head_tree_sha,
            expected_parent_sha=base_sha,
        )
        status, payload = await self._github_proxy_request(
            tenant_id=tenant_id,
            method="POST",
            endpoint=f"{prefix}/git/commits",
            body=commit,
        )
        if status != 201 or self._response_sha(payload) != expected_head_sha:
            raise RepositoryPublishError("GitHub did not preserve the approved commit")

        branch_path = quote(workspace.branch, safe="/")
        status, payload = await self._github_proxy_request(
            tenant_id=tenant_id,
            method="GET",
            endpoint=f"{prefix}/git/ref/heads/{branch_path}",
        )
        if status == 404:
            status, payload = await self._github_proxy_request(
                tenant_id=tenant_id,
                method="POST",
                endpoint=f"{prefix}/git/refs",
                body={
                    "ref": f"refs/heads/{workspace.branch}",
                    "sha": expected_head_sha,
                },
            )
            remote_sha = (
                str(payload.get("object", {}).get("sha") or "")
                if isinstance(payload, dict) and isinstance(payload.get("object"), dict)
                else ""
            )
            if status != 201 or remote_sha != expected_head_sha:
                raise RepositoryPublishError("GitHub could not create the repository branch")
        elif status == 200:
            remote_sha = (
                str(payload.get("object", {}).get("sha") or "")
                if isinstance(payload, dict) and isinstance(payload.get("object"), dict)
                else ""
            )
            if remote_sha != expected_head_sha:
                raise RepositoryPublishError("GitHub branch already exists at a different commit")
        else:
            raise RepositoryPublishError("GitHub could not inspect the repository branch")

        status, payload = await self._github_proxy_request(
            tenant_id=tenant_id,
            method="POST",
            endpoint=f"{prefix}/pulls",
            body={
                "title": title,
                "head": workspace.branch,
                "base": workspace.base_ref,
                "body": body,
                "draft": draft,
            },
        )
        if status == 422:
            query = urlencode(
                {
                    "state": "open",
                    "head": f"{owner}:{workspace.branch}",
                    "base": workspace.base_ref,
                }
            )
            status, payload = await self._github_proxy_request(
                tenant_id=tenant_id,
                method="GET",
                endpoint=f"{prefix}/pulls?{query}",
            )
            if status == 200 and isinstance(payload, list) and payload:
                payload = payload[0]
            else:
                raise RepositoryPublishError("GitHub could not create the pull request")
        elif status != 201:
            raise RepositoryPublishError("GitHub could not create the pull request")
        url = str(payload.get("html_url") or "") if isinstance(payload, dict) else ""
        if not url.startswith("https://github.com/"):
            raise RepositoryPublishError("GitHub returned an invalid pull request response")
        return url

    async def _github_proxy_request(
        self,
        *,
        tenant_id: str,
        method: Literal["GET", "POST", "PATCH", "DELETE"],
        endpoint: str,
        body: object | None = None,
    ) -> tuple[int, Any]:
        if self._github_api_proxy is None:
            raise RepositoryPublishError("Composio GitHub publishing is unavailable")
        try:
            return await asyncio.to_thread(
                self._github_api_proxy.request,
                tenant_id=tenant_id,
                method=method,
                endpoint=endpoint,
                body=body,
            )
        except Exception as exc:
            raise RepositoryPublishError(
                "Composio GitHub publishing failed. Configure Composio, connect one "
                "GitHub account, and retry"
            ) from exc

    @classmethod
    async def _changed_paths(
        cls,
        backend: SandboxBackendProtocol,
        *,
        base_sha: str,
    ) -> _StringList:
        raw = await cls._git_binary_output(
            backend,
            f"git diff --no-renames --name-only -z {cls._shell_quote(base_sha)} HEAD",
        )
        if not raw:
            return []
        values = raw.removesuffix(b"\0").split(b"\0")
        if len(values) > _MAX_PUBLISH_FILES:
            raise RepositoryPublishError("repository publish has too many changed files")
        paths: _StringList = []
        for value in values:
            try:
                path = value.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise RepositoryPublishError("repository contains a non-UTF-8 path") from exc
            parsed = PurePosixPath(path)
            if not path or path.startswith("/") or ".." in parsed.parts or str(parsed) != path:
                raise RepositoryPublishError("repository contains an unsafe path")
            paths.append(path)
        return paths

    @classmethod
    async def _tree_entry(
        cls,
        backend: SandboxBackendProtocol,
        *,
        ref: str,
        path: str,
    ) -> _GitTreeEntry | None:
        raw = await cls._git_binary_output(
            backend,
            f"git ls-tree -z {cls._shell_quote(ref)} -- {cls._shell_quote(path)}",
        )
        if not raw:
            return None
        if raw.count(b"\0") != 1:
            raise RepositoryPublishError("repository returned an invalid tree entry")
        metadata, returned_path = raw.removesuffix(b"\0").split(b"\t", 1)
        try:
            mode, object_type, sha = metadata.decode("ascii").split(" ", 2)
            decoded_path = returned_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RepositoryPublishError("repository returned an invalid tree entry") from exc
        if (
            decoded_path != path
            or mode not in {"100644", "100755", "120000", "160000"}
            or object_type not in {"blob", "commit"}
            or not _SHA_RE.fullmatch(sha)
        ):
            raise RepositoryPublishError("repository returned an invalid tree entry")
        return _GitTreeEntry(mode=mode, type=object_type, sha=sha)

    @classmethod
    async def _blob_content(
        cls,
        backend: SandboxBackendProtocol,
        *,
        path: str,
        entry: _GitTreeEntry,
    ) -> bytes:
        if entry.mode == "120000":
            return await cls._git_binary_output(
                backend,
                f"git cat-file blob {cls._shell_quote(entry.sha)}",
            )
        responses = await asyncio.to_thread(
            backend.download_files,
            [f"/workspace/{path}"],
        )
        if len(responses) != 1 or responses[0].error is not None:
            raise RepositoryPublishError("repository file could not be read")
        return bytes(responses[0].content or b"")

    @classmethod
    async def _commit_payload(
        cls,
        backend: SandboxBackendProtocol,
        *,
        expected_head_sha: str,
        expected_tree_sha: str,
        expected_parent_sha: str,
    ) -> dict[str, Any]:
        raw = await cls._git_binary_output(
            backend,
            f"git cat-file commit {cls._shell_quote(expected_head_sha)}",
        )
        try:
            raw_headers, raw_message = raw.split(b"\n\n", 1)
            lines = raw_headers.decode("utf-8", errors="strict").splitlines()
        except (UnicodeDecodeError, ValueError) as exc:
            raise RepositoryPublishError("repository commit is not supported") from exc
        if any(line.startswith((" ", "\t")) for line in lines):
            raise RepositoryPublishError("signed or extended commits require a direct GitHub token")
        headers: dict[str, _StringList] = {}
        for line in lines:
            key, separator, value = line.partition(" ")
            if not separator:
                raise RepositoryPublishError("repository commit is not supported")
            headers.setdefault(key, []).append(value)
        if set(headers) != {"tree", "parent", "author", "committer"}:
            raise RepositoryPublishError("signed or extended commits require a direct GitHub token")
        if headers["tree"] != [expected_tree_sha] or headers["parent"] != [expected_parent_sha]:
            raise RepositoryPublishError("repository commit lineage changed")
        try:
            message = raw_message.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RepositoryPublishError("non-UTF-8 commits require a direct GitHub token") from exc
        return {
            "message": message,
            "tree": expected_tree_sha,
            "parents": [expected_parent_sha],
            "author": cls._git_identity_payload(headers["author"][0]),
            "committer": cls._git_identity_payload(headers["committer"][0]),
        }

    @staticmethod
    def _git_identity_payload(value: str) -> dict[str, str]:
        match = _GIT_IDENTITY_RE.fullmatch(value)
        if match is None:
            raise RepositoryPublishError("repository commit identity is not supported")
        name, email, raw_timestamp, raw_offset = match.groups()
        sign = 1 if raw_offset[0] == "+" else -1
        offset = timedelta(
            hours=sign * int(raw_offset[1:3]),
            minutes=sign * int(raw_offset[3:5]),
        )
        try:
            date = datetime.fromtimestamp(
                int(raw_timestamp),
                tz=timezone(offset),
            ).isoformat(timespec="seconds")
        except (OverflowError, ValueError) as exc:
            raise RepositoryPublishError("repository commit date is not supported") from exc
        return {"name": name, "email": email, "date": date}

    @classmethod
    async def _git_binary_output(
        cls,
        backend: SandboxBackendProtocol,
        command: str,
    ) -> bytes:
        response = await backend.aexecute(
            f"{command} | base64 | tr -d '\\n'",
            timeout=120,
        )
        if response.exit_code != 0 or response.truncated:
            raise RepositoryPublishError("repository command output was incomplete")
        try:
            return base64.b64decode(response.output, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RepositoryPublishError("repository returned invalid binary data") from exc

    @staticmethod
    def _git_blob_sha(content: bytes) -> str:
        prefix = f"blob {len(content)}\0".encode()
        return hashlib.sha1(prefix + content, usedforsecurity=False).hexdigest()

    @staticmethod
    def _response_sha(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        value = str(payload.get("sha") or "").strip().casefold()
        return value if _SHA_RE.fullmatch(value) else ""

    def _open_pull_request(
        self,
        workspace: RepositoryWorkspace,
        *,
        token: str,
        title: str,
        body: str,
        draft: bool,
    ) -> str:
        _, owner, repository = self._normalize_repository_url(workspace.repository_url)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        with self._http_client_factory() as client:
            response = client.post(
                f"https://api.github.com/repos/{owner}/{repository}/pulls",
                headers=headers,
                json={
                    "title": title,
                    "head": workspace.branch,
                    "base": workspace.base_ref,
                    "body": body,
                    "draft": draft,
                },
            )
            if response.status_code == 422:
                existing = client.get(
                    f"https://api.github.com/repos/{owner}/{repository}/pulls",
                    headers=headers,
                    params={
                        "state": "open",
                        "head": f"{owner}:{workspace.branch}",
                        "base": workspace.base_ref,
                    },
                )
                if existing.is_success:
                    items = existing.json()
                    if isinstance(items, list) and items:
                        url = str(items[0].get("html_url") or "")
                        if url:
                            return url
            if not response.is_success:
                raise RepositoryPublishError("GitHub could not create the pull request")
            payload = response.json()
        url = str(payload.get("html_url") or "") if isinstance(payload, dict) else ""
        if not url.startswith("https://github.com/"):
            raise RepositoryPublishError("GitHub returned an invalid pull request response")
        return url

    @staticmethod
    async def _run(
        backend: SandboxBackendProtocol,
        command: str,
        *,
        timeout: int,
    ) -> str:
        response = await backend.aexecute(command, timeout=timeout)
        if response.exit_code != 0:
            raise RepositoryWorkspaceError("repository command failed")
        return response.output

    @classmethod
    async def _git_sha(cls, backend: SandboxBackendProtocol, ref: str) -> str:
        value = (
            await cls._run(backend, f"git rev-parse {cls._shell_quote(ref)}", timeout=60)
        ).strip()
        if not _SHA_RE.fullmatch(value):
            raise RepositoryWorkspaceError("repository returned an invalid commit")
        return value

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

    @staticmethod
    def _public_error(error: BaseException) -> str:
        if isinstance(error, RepositoryWorkspaceError | RepositorySandboxError):
            return str(error)[:500]
        return "repository workspace operation failed"


__all__ = [
    "RepositoryPublishError",
    "RepositoryWorkspaceConflictError",
    "RepositoryWorkspaceError",
    "RepositoryWorkspaceNotFoundError",
    "RepositoryWorkspaceService",
]
