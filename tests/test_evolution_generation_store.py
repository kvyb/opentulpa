from __future__ import annotations

import hashlib
import os
import platform
import stat
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest

from opentulpa.evolution.generation import (
    GenerationDescriptor,
    GenerationIdentity,
    GenerationManifest,
    StateContract,
    canonical_json_bytes,
    generation_manifest_sha256,
)
from opentulpa.evolution.generation_store import (
    GenerationStore,
    GenerationStoreError,
    runtime_tree_sha256,
)


def _contract() -> StateContract:
    return StateContract(
        runtime_protocol=1,
        controller_min=1,
        controller_max=2,
        product_state_schema=1,
        workspace_api=1,
    )


def _installed_generation(
    generations_root: Path,
    *,
    identity_changes: dict[str, object] | None = None,
    complete_payload: bytes = b"",
) -> tuple[Path, GenerationManifest]:
    generations_root.mkdir(mode=0o711, exist_ok=True)
    generations_root.chmod(0o711)
    wheel = b"deterministic wheel contents"
    lock = b"version = 1\n"
    contract = _contract()
    identity_values: dict[str, object] = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "wheel_sha256": hashlib.sha256(wheel).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(lock).hexdigest(),
        "evaluator_fingerprint": f"sha256:{'c' * 64}",
        "evaluation_input_sha256": "d" * 64,
        "python_runtime_sha256": "e" * 64,
        "cpython_version": platform.python_version(),
        "cpython_cache_tag": sys.implementation.cache_tag,
        "cpython_abi_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "os_name": os.name,
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
        "build_recipe_version": "fixed-wheel-v2",
        "runtime_protocol": 1,
        "controller_min": 1,
        "controller_max": 2,
        "state_contract_sha256": contract.sha256(),
        "install_profile": "runtime",
        "extras": (),
        "entrypoint": ("venv/bin/python", "-I", "-m", "opentulpa"),
    }
    identity_values.update(identity_changes or {})
    identity = GenerationIdentity.model_validate(identity_values)
    generation = generations_root / identity.generation_id
    artifacts = generation / "artifacts"
    bin_path = generation / "venv" / "bin"
    artifacts.mkdir(parents=True)
    bin_path.mkdir(parents=True)
    wheel_path = artifacts / "opentulpa-0.1-py3-none-any.whl"
    lock_path = artifacts / "uv.lock"
    wheel_path.write_bytes(wheel)
    lock_path.write_bytes(lock)
    interpreter = bin_path / "python"
    interpreter.write_bytes(b"trusted interpreter placeholder\n")
    interpreter.chmod(0o555)
    empty_directory = generation / "venv" / "empty"
    empty_directory.mkdir()
    wheel_path.chmod(0o444)
    lock_path.chmod(0o444)
    for directory in (artifacts, bin_path, empty_directory, generation / "venv"):
        directory.chmod(0o555)
    runtime_digest = runtime_tree_sha256(generation)
    manifest = GenerationManifest(
        identity=identity,
        state_contract=contract,
        descriptor=GenerationDescriptor(
            wheel_path="artifacts/opentulpa-0.1-py3-none-any.whl",
            wheel_size_bytes=len(wheel),
            uv_lock_path="artifacts/uv.lock",
            uv_lock_size_bytes=len(lock),
            venv_path="venv",
        ),
        runtime_tree_sha256=runtime_digest,
    )
    manifest_path = generation / "manifest.json"
    complete_path = generation / "COMPLETE"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    complete_path.write_bytes(complete_payload)
    manifest_path.chmod(0o444)
    complete_path.chmod(0o444)
    generation.chmod(0o555)
    return generation, manifest


def _staged_generation(
    generations_root: Path,
) -> tuple[Path, GenerationManifest]:
    generation, published_manifest = _installed_generation(generations_root)
    generation.chmod(0o700)
    (generation / "COMPLETE").unlink()
    (generation / "manifest.json").unlink()
    for path in generation.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o755)
        elif stat.S_ISREG(metadata.st_mode):
            path.chmod(0o755 if metadata.st_mode & 0o111 else 0o644)
    (generation / "BUILDING").write_bytes(
        canonical_json_bytes(
            {
                "nonce": "a" * 32,
                "pid": os.getpid(),
                "started_at": time.time(),
            }
        )
    )
    (generation / "BUILDING").chmod(0o600)
    return generation, published_manifest.model_copy(update={"runtime_tree_sha256": "0" * 64})


class _ExpectedProvenance(TypedDict):
    expected_manifest_digest: str
    expected_state_contract_digest: str
    expected_evaluator_fingerprint: str
    expected_install_profile: str
    controller_protocol: int


def _expected(manifest: GenerationManifest) -> _ExpectedProvenance:
    return {
        "expected_manifest_digest": generation_manifest_sha256(manifest),
        "expected_state_contract_digest": manifest.state_contract.sha256(),
        "expected_evaluator_fingerprint": manifest.identity.evaluator_fingerprint,
        "expected_install_profile": manifest.identity.install_profile,
        "controller_protocol": 1,
    }


def test_store_owns_staged_generation_publication(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    generation, staged_manifest = _staged_generation(root)
    store = GenerationStore(root)

    manifest = store.publish_staged(staged_manifest.identity.generation_id, staged_manifest)
    installed = store.open(manifest.identity.generation_id, **_expected(manifest))

    assert installed.manifest == manifest
    assert manifest.runtime_tree_sha256 != "0" * 64
    assert not (generation / "BUILDING").exists()
    assert (generation / "COMPLETE").read_bytes() == b""
    assert stat.S_IMODE(generation.stat().st_mode) == 0o555
    assert stat.S_IMODE((generation / "manifest.json").stat().st_mode) == 0o444


def test_staged_publication_fails_before_complete_on_artifact_tampering(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    generation, staged_manifest = _staged_generation(root)
    wheel = generation / staged_manifest.descriptor.wheel_path
    wheel.write_bytes(b"tampered wheel")

    with pytest.raises(GenerationStoreError, match="hash or size"):
        GenerationStore(root).publish_staged(
            staged_manifest.identity.generation_id,
            staged_manifest,
        )

    assert (generation / "BUILDING").exists()
    assert not (generation / "COMPLETE").exists()


def test_store_requires_external_provenance_and_exposes_exact_paths(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    generation, manifest = _installed_generation(root)
    store = GenerationStore(root)

    installed = store.open(manifest.identity.generation_id, **_expected(manifest))

    assert installed.path == generation
    assert installed.interpreter_path == generation / "venv/bin/python"
    assert installed.entrypoint_path == generation / "venv/bin/python"
    assert installed.entrypoint_argv == (
        str(installed.entrypoint_path),
        "-I",
        "-m",
        "opentulpa",
    )
    with pytest.raises(TypeError):
        store.open(manifest.identity.generation_id)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("expected_manifest_digest", f"sha256:{'0' * 64}", "manifest provenance"),
        ("expected_state_contract_digest", "0" * 64, "state contract provenance"),
        ("expected_evaluator_fingerprint", f"sha256:{'0' * 64}", "evaluator provenance"),
        ("expected_install_profile", "other", "install profile provenance"),
        ("controller_protocol", 3, "controller protocol"),
    ],
)
def test_store_rejects_expected_provenance_mismatch(
    tmp_path: Path,
    field: str,
    replacement: str | int,
    message: str,
) -> None:
    root = tmp_path / "generations"
    _, manifest = _installed_generation(root)
    expected = _expected(manifest)
    cast(dict[str, Any], expected)[field] = replacement

    with pytest.raises(GenerationStoreError, match=message):
        GenerationStore(root).open(manifest.identity.generation_id, **expected)  # type: ignore[arg-type]


def test_store_detects_runtime_file_tampering(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    generation, manifest = _installed_generation(root)
    entrypoint = generation / "venv/bin/python"
    generation.chmod(0o700)
    (generation / "venv/bin").chmod(0o755)
    entrypoint.chmod(0o755)
    entrypoint.write_text("tampered interpreter\n", encoding="utf-8")
    entrypoint.chmod(0o555)
    (generation / "venv/bin").chmod(0o555)
    generation.chmod(0o555)

    with pytest.raises(GenerationStoreError, match="runtime tree"):
        GenerationStore(root).open(manifest.identity.generation_id, **_expected(manifest))


def test_store_detects_empty_directory_removal_and_directory_mode_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    generation, manifest = _installed_generation(root)
    empty_directory = generation / "venv/empty"
    empty_directory.chmod(0o500)

    assert runtime_tree_sha256(generation) != manifest.runtime_tree_sha256
    with pytest.raises(GenerationStoreError, match="directory mode"):
        GenerationStore(root).open(manifest.identity.generation_id, **_expected(manifest))

    empty_directory.chmod(0o555)
    (generation / "venv").chmod(0o755)
    empty_directory.rmdir()
    (generation / "venv").chmod(0o555)
    with pytest.raises(GenerationStoreError, match="runtime tree"):
        GenerationStore(root).open(manifest.identity.generation_id, **_expected(manifest))


def test_store_rejects_platform_incompatible_generation(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    _, manifest = _installed_generation(root, identity_changes={"machine": "other-machine"})

    with pytest.raises(GenerationStoreError, match="incompatible"):
        GenerationStore(root).open(manifest.identity.generation_id, **_expected(manifest))


def test_cleanup_validates_malformed_complete_and_preserves_valid_complete(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    valid, _ = _installed_generation(root)
    malformed, _ = _installed_generation(
        root,
        identity_changes={"evaluation_input_sha256": "e" * 64},
        complete_payload=b"not-empty",
    )

    quarantined = GenerationStore(root).cleanup_incomplete()

    assert valid.is_dir()
    assert not malformed.exists()
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith(f"{malformed.name}.")


def test_cleanup_distinguishes_live_and_stale_building_owners(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    store = GenerationStore(root)
    live = root / ("a" * 64)
    stale = root / ("b" * 64)
    live.mkdir()
    stale.mkdir()
    (live / "BUILDING").write_text(
        f'{{"pid":{os.getpid()},"started_at":{time.time()}}}',
        encoding="ascii",
    )
    (stale / "BUILDING").write_text(
        '{"pid":99999999,"started_at":0}',
        encoding="ascii",
    )

    quarantined = store.cleanup_incomplete(stale_after_seconds=60)

    assert live.is_dir()
    assert not stale.exists()
    assert len(quarantined) == 1


def test_store_requires_dedicated_traversable_root_and_rejects_symlink_ancestors(
    tmp_path: Path,
) -> None:
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="mode 0711"):
        GenerationStore(public)

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    link = tmp_path / "linked"
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link ancestor"):
        GenerationStore(link / "generations")


def test_published_layout_has_unprivileged_traversal_without_control_read_access(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime-generations"
    generation, manifest = _installed_generation(root)
    store = GenerationStore(root)
    store.open(manifest.identity.generation_id, **_expected(manifest))

    assert stat.S_IMODE(root.stat().st_mode) == 0o711
    assert stat.S_IMODE(generation.stat().st_mode) == 0o555
    assert stat.S_IMODE((generation / "venv/bin").stat().st_mode) == 0o555
    assert stat.S_IMODE((generation / "venv/bin/python").stat().st_mode) == 0o555
    assert not (root / ".generation-store.lock").exists()
    assert stat.S_IMODE((store.control_root / "generation-store.lock").stat().st_mode) == 0o600
    assert stat.S_IMODE(store.control_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(root.stat().st_mode) & 0o007 == 0o001
    assert stat.S_IMODE(generation.stat().st_mode) & 0o007 == 0o005
