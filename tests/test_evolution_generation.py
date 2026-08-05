from __future__ import annotations

from importlib import resources

import pytest
from pydantic import ValidationError

import opentulpa
from opentulpa.evolution.generation import (
    UPSTREAM_LINEAGE_METADATA_KEY,
    GenerationDescriptor,
    GenerationIdentity,
    GenerationManifest,
    StateContract,
    UpstreamLineage,
    canonical_json_bytes,
)
from opentulpa.evolution.models import Candidate


def _state_contract() -> StateContract:
    return StateContract(
        runtime_protocol=1,
        controller_min=1,
        controller_max=1,
        product_state_schema=1,
        workspace_api=1,
    )


def _identity(**changes: object) -> GenerationIdentity:
    values: dict[str, object] = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "wheel_sha256": "c" * 64,
        "uv_lock_sha256": "d" * 64,
        "evaluator_fingerprint": "sha256:" + "e" * 64,
        "evaluation_input_sha256": "f" * 64,
        "python_runtime_sha256": "1" * 64,
        "cpython_version": "3.12.7",
        "cpython_cache_tag": "cpython-312",
        "cpython_abi_tag": "cp312",
        "os_name": "posix",
        "platform": "macosx-15.0-arm64",
        "machine": "arm64",
        "build_recipe_version": "wheel-venv-v1",
        "runtime_protocol": 1,
        "controller_min": 1,
        "controller_max": 1,
        "state_contract_sha256": _state_contract().sha256(),
        "install_profile": "runtime",
        "extras": ("browser", "documents"),
        "entrypoint": ("venv/bin/python", "-I", "-m", "opentulpa"),
    }
    values.update(changes)
    return GenerationIdentity.model_validate(values)


def _descriptor(**changes: object) -> GenerationDescriptor:
    values: dict[str, object] = {
        "wheel_path": "artifacts/opentulpa-0.1.0-py3-none-any.whl",
        "wheel_size_bytes": 42,
        "uv_lock_path": "artifacts/uv.lock",
        "uv_lock_size_bytes": 84,
        "venv_path": "venv",
    }
    values.update(changes)
    return GenerationDescriptor.model_validate(values)


def test_generation_id_is_canonical_across_mapping_order() -> None:
    identity = _identity()
    payload = identity.model_dump(exclude={"generation_id"})
    reordered = dict(reversed(tuple(payload.items())))

    rebuilt = GenerationIdentity.model_validate(reordered)

    assert rebuilt.generation_id == identity.generation_id
    assert len(identity.generation_id) == 64
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}'


def test_generation_identity_has_golden_canonical_json_and_id() -> None:
    identity = _identity()
    canonical = canonical_json_bytes(identity.model_dump(mode="json", exclude={"generation_id"}))

    assert canonical == (
        b'{"build_recipe_version":"wheel-venv-v1","controller_max":1,"controller_min":1,'
        b'"cpython_abi_tag":"cp312","cpython_cache_tag":"cpython-312",'
        b'"cpython_version":"3.12.7","entrypoint":["venv/bin/python","-I","-m","opentulpa"],'
        b'"evaluation_input_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",'
        b'"evaluator_fingerprint":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
        b'"extras":["browser","documents"],"install_profile":"runtime","machine":"arm64",'
        b'"os_name":"posix","platform":"macosx-15.0-arm64",'
        b'"python_runtime_sha256":"1111111111111111111111111111111111111111111111111111111111111111",'
        b'"runtime_protocol":1,'
        b'"source_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"source_tree_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        b'"state_contract_sha256":"b2ca0677c7b450ae917e64a59743dc50bf907b1bd85dc2c9e5c8e4c3f057994a",'
        b'"uv_lock_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
        b'"wheel_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}'
    )
    assert identity.generation_id == "ff30f7dd2821188d0e36fbcd4daca4e8cc27b01428c1c06b58549a2a985eb915"


def test_generation_identity_detects_serialized_tampering() -> None:
    payload = _identity().model_dump(mode="json")
    payload["wheel_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="generation_id does not match"):
        GenerationIdentity.model_validate_json(canonical_json_bytes(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit", "a" * 41),
        ("source_tree_sha256", "sha256:" + "b" * 64),
        ("wheel_sha256", "C" * 64),
        ("uv_lock_sha256", "d" * 63),
        ("evaluator_fingerprint", "e" * 64),
        ("evaluation_input_sha256", "sha256:" + "f" * 64),
        ("python_runtime_sha256", "1" * 63),
        ("state_contract_sha256", "0" * 65),
    ],
)
def test_generation_identity_rejects_invalid_commits_and_digests(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        _identity(**{field: value})


@pytest.mark.parametrize(
    "entrypoint",
    [
        (),
        ("/venv/bin/opentulpa",),
        ("venv/../bin/opentulpa",),
        ("venv//opentulpa",),
        ("C:/app",),
    ],
)
def test_generation_identity_rejects_invalid_entrypoint(entrypoint: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        _identity(entrypoint=entrypoint)


def test_generation_identity_rejects_nul_path_and_incoherent_runtime() -> None:
    with pytest.raises(ValidationError, match="NUL"):
        _identity(entrypoint=("venv/bin/open\x00tulpa",))
    with pytest.raises(ValidationError, match="cache tag"):
        _identity(cpython_cache_tag="cpython-313")
    with pytest.raises(ValidationError, match="ABI tag"):
        _identity(cpython_abi_tag="cp313")
    with pytest.raises(ValidationError, match="protocol range"):
        _identity(runtime_protocol=2)
    with pytest.raises(ValidationError):
        _identity(controller_min=0)
    with pytest.raises(ValidationError):
        _identity(runtime_protocol="1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpython_version", "03.12.7"),
        ("cpython_version", "3.012.7"),
        ("cpython_version", "3.12.07"),
        ("cpython_version", "3.12"),
        ("cpython_cache_tag", "cpython-312x"),
        ("cpython_abi_tag", "cp312m"),
        ("cpython_abi_tag", "cp3120"),
    ],
)
def test_generation_identity_requires_canonical_cpython_metadata(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        _identity(**{field: value})


@pytest.mark.parametrize(
    "extras",
    [
        ("Browser",),
        ("browser_tools",),
        ("browser.tools",),
        ("browser--tools",),
        ("browser-",),
        ("-browser",),
        ("documents", "browser"),
        ("browser", "browser"),
    ],
)
def test_generation_identity_rejects_noncanonical_extras(extras: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        _identity(extras=extras)


def test_generation_identity_keeps_python_tuple_input_strict_but_parses_json_arrays() -> None:
    identity = _identity()

    with pytest.raises(ValidationError):
        _identity(extras=["browser", "documents"])
    with pytest.raises(ValidationError):
        _identity(entrypoint=["venv/bin/opentulpa"])
    assert GenerationIdentity.model_validate_json(canonical_json_bytes(identity)) == identity


def test_manifest_binds_format_identity_and_artifact_metadata() -> None:
    identity = _identity()
    descriptor = _descriptor()
    manifest = GenerationManifest(
        identity=identity,
        state_contract=_state_contract(),
        descriptor=descriptor,
        runtime_tree_sha256="0" * 64,
    )

    assert manifest.manifest_format == 2
    assert manifest.identity.generation_id == identity.generation_id
    assert manifest.state_contract.sha256() == identity.state_contract_sha256
    assert manifest.descriptor.wheel_size_bytes == 42
    changed_runtime = manifest.model_copy(update={"runtime_tree_sha256": "1" * 64})
    assert changed_runtime.identity.generation_id == manifest.identity.generation_id
    assert canonical_json_bytes(changed_runtime) != canonical_json_bytes(manifest)
    with pytest.raises(ValidationError):
        GenerationManifest.model_validate(
            {
                "identity": identity,
                "state_contract": _state_contract(),
                "descriptor": descriptor,
                "runtime_tree_sha256": "0" * 64,
                "manifest_format": 1,
            }
        )


@pytest.mark.parametrize(
    "identity_changes",
    [
        {"state_contract_sha256": "0" * 64},
        {"controller_max": 2},
        {"runtime_protocol": 2, "controller_max": 2},
    ],
)
def test_manifest_rejects_identity_state_contract_contradictions(
    identity_changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="contradicts state contract"):
        GenerationManifest(
            identity=_identity(**identity_changes),
            state_contract=_state_contract(),
            descriptor=_descriptor(),
            runtime_tree_sha256="0" * 64,
        )

    contract = StateContract(
        runtime_protocol=2,
        controller_min=1,
        controller_max=2,
        product_state_schema=1,
        workspace_api=1,
    )
    with pytest.raises(ValidationError, match="controller_min"):
        GenerationManifest(
            identity=_identity(
                runtime_protocol=2,
                controller_min=2,
                controller_max=2,
                state_contract_sha256=contract.sha256(),
            ),
            state_contract=contract,
            descriptor=_descriptor(),
            runtime_tree_sha256="0" * 64,
        )


def test_descriptor_rejects_file_ancestors_and_non_venv_entrypoints() -> None:
    with pytest.raises(ValidationError, match="ancestors"):
        _descriptor(wheel_path="artifacts")

    for entrypoint in (("bin/opentulpa",), ("venv/lib/opentulpa",)):
        with pytest.raises(ValidationError, match="declared virtualenv"):
            GenerationManifest(
                identity=_identity(entrypoint=entrypoint),
                state_contract=_state_contract(),
                descriptor=_descriptor(),
                runtime_tree_sha256="0" * 64,
            )

    with pytest.raises(ValidationError, match="cannot contain one another"):
        GenerationManifest(
            identity=_identity(),
            state_contract=_state_contract(),
            descriptor=_descriptor(wheel_path="venv/bin"),
            runtime_tree_sha256="0" * 64,
        )

    with pytest.raises(ValidationError, match="cannot contain one another"):
        GenerationManifest(
            identity=_identity(),
            state_contract=_state_contract(),
            descriptor=_descriptor(wheel_path="venv/bin/python/package.whl"),
            runtime_tree_sha256="0" * 64,
        )


def test_entrypoint_argument_whitespace_is_identity_significant() -> None:
    spaced = _identity(entrypoint=("venv/bin/opentulpa", " value "))
    plain = _identity(entrypoint=("venv/bin/opentulpa", "value"))

    assert spaced.entrypoint[1] == " value "
    assert spaced.generation_id != plain.generation_id


def test_packaged_release_contract_parses_and_has_stable_digest() -> None:
    resource = resources.files(opentulpa).joinpath("resources", "release_contract.json")
    assert resource.is_file()
    payload = resource.read_bytes()
    payload.decode("ascii")
    contract = StateContract.model_validate_json(payload)

    assert contract == _state_contract()
    assert contract.sha256() == "b2ca0677c7b450ae917e64a59743dc50bf907b1bd85dc2c9e5c8e4c3f057994a"


def test_generation_contracts_are_frozen_and_forbid_extra_fields() -> None:
    identity = _identity()

    with pytest.raises(ValidationError, match="frozen"):
        identity.machine = "x86_64"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StateContract.model_validate(
            {
                **_state_contract().model_dump(),
                "unknown_version": 1,
            }
        )


def test_candidate_without_lineage_retains_baseline_keys_and_canonical_bytes() -> None:
    old_payload = {
        "id": "candidate-old",
        "base_commit": "legacy-release-reference",
        "requested_improvement": "Keep old payloads readable",
        "created_at": "2026-08-03T10:00:00Z",
        "updated_at": "2026-08-03T10:00:00Z",
    }
    expected = {
        "id": "candidate-old",
        "base_commit": "legacy-release-reference",
        "requested_improvement": "Keep old payloads readable",
        "status": "building",
        "revision": 1,
        "parent_candidate_id": None,
        "source_commit": None,
        "worktree_path": None,
        "dependency_lock_hash": None,
        "artifact_digest": None,
        "evaluator_fingerprint": None,
        "evaluation_report": None,
        "contribution": None,
        "metadata": {},
        "created_at": "2026-08-03T10:00:00Z",
        "updated_at": "2026-08-03T10:00:00Z",
    }

    candidate = Candidate.model_validate(old_payload)

    assert candidate.model_dump(mode="json") == expected
    assert canonical_json_bytes(candidate) == canonical_json_bytes(expected)
    with pytest.raises(ValidationError):
        Candidate.model_validate({**old_payload, "upstream_commit": "a" * 40})


def test_upstream_lineage_uses_namespaced_candidate_metadata_and_coupled_commits() -> None:
    lineage = UpstreamLineage(
        upstream_commit="a" * 40,
        merge_base_commit="b" * 40,
    )
    candidate = Candidate(
        id="candidate-lineage",
        base_commit="base-release",
        requested_improvement="Track upstream safely",
        metadata={UPSTREAM_LINEAGE_METADATA_KEY: lineage.model_dump(mode="json")},
    )

    stored = candidate.metadata[UPSTREAM_LINEAGE_METADATA_KEY]
    assert UpstreamLineage.model_validate(stored) == lineage
    for one_sided in ({"upstream_commit": "a" * 40}, {"merge_base_commit": "b" * 64}):
        with pytest.raises(ValidationError, match="recorded together"):
            UpstreamLineage.model_validate(one_sided)
    with pytest.raises(ValidationError):
        UpstreamLineage(upstream_commit="a" * 41, merge_base_commit="b" * 40)
    with pytest.raises(ValidationError, match="same object format"):
        UpstreamLineage(upstream_commit="a" * 40, merge_base_commit="b" * 64)
