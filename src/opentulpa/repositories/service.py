"""Tenant-owned repository workspace lifecycle and publishing."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from contextlib import suppress
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
_MAX_STATUS_LINES = 100


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
        http_client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self._store = store
        self._providers = providers
        self._github_token_resolver = github_token_resolver
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
            remote_url = (
                await self._run(backend, "git remote get-url origin", timeout=60)
            ).strip()
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
            if not token:
                raise RepositoryPublishError(
                    "GitHub publishing is not configured; paste GITHUB_TOKEN=<token>"
                )
            await asyncio.to_thread(provider.push, workspace, github_token=token)
            pull_request_url = await asyncio.to_thread(
                self._open_pull_request,
                workspace,
                token=token,
                title=safe_title,
                body=safe_body,
                draft=bool(draft),
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
            }

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
        value = (await cls._run(backend, f"git rev-parse {cls._shell_quote(ref)}", timeout=60)).strip()
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
