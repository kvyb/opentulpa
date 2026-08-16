from __future__ import annotations

import json
import os
import platform
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from opentulpa.client.config import Connection
from opentulpa.host import cli
from opentulpa.host.paths import HostPathError, HostPaths


def _client_factory(
    monkeypatch: pytest.MonkeyPatch,
    handler: httpx.MockTransport,
) -> None:
    original = httpx.Client

    def client(**kwargs: object) -> httpx.Client:
        return original(transport=handler, **kwargs)

    monkeypatch.setattr(cli.httpx, "Client", client)


def test_connect_accepts_owner_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer owner-token"
        return httpx.Response(200, json={"claimed": True, "authenticated": True})

    _client_factory(monkeypatch, httpx.MockTransport(handler))

    assert cli._connect_credential("https://tulpa.example", "owner-token") == "owner-token"  # noqa: SLF001


def test_connect_exchanges_pairing_code_for_owner_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"claimed": False, "authenticated": False})
        return httpx.Response(200, json={"claimed": True, "owner_token": "issued-owner-token"})

    _client_factory(monkeypatch, httpx.MockTransport(handler))

    assert (
        cli._connect_credential("https://tulpa.example", "one-time-pairing-code")  # noqa: SLF001
        == "issued-owner-token"
    )
    assert requests[1].method == "POST"
    assert requests[1].url.path == "/_host/api/claim"
    assert b"one-time-pairing-code" in requests[1].content


def test_bare_opentulpa_opens_tui_not_server(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[bool] = []
    monkeypatch.setattr(cli, "_open_tui", lambda: opened.append(True))
    monkeypatch.setattr(cli, "serve", lambda: pytest.fail("bare command started the server"))
    monkeypatch.setattr("sys.argv", ["opentulpa"])

    cli.main()

    assert opened == [True]


def test_first_launch_defaults_to_private_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Connection(
        url="http://127.0.0.1:8123",
        token="",
        thread_id="thread-1",
        credential_storage="none",
    )
    configured: list[Connection] = []
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(cli, "ensure_local_server", lambda: connection.url)
    monkeypatch.setattr(cli, "_connect_credential", lambda url, token: "")
    monkeypatch.setattr(cli, "save_connection", lambda url, token: connection)
    monkeypatch.setattr(cli, "_configure_runtime", configured.append)

    assert cli._first_connection() == connection  # noqa: SLF001
    assert configured == [connection]


def test_first_launch_configures_model_without_printing_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"configured": False})
        return httpx.Response(200, json={"runtime": "ready"})

    _client_factory(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    answers = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "model-secret")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    connection = Connection(
        url="http://127.0.0.1:8000",
        token="",
        thread_id="thread-1",
        credential_storage="none",
    )

    cli._configure_runtime(connection)  # noqa: SLF001

    assert requests[-1].method == "PUT"
    assert b"model-secret" in requests[-1].content
    assert b"moonshotai/kimi-k3" in requests[-1].content
    assert "model-secret" not in capsys.readouterr().out


def test_server_origin_and_pairing_code_are_pasteable(tmp_path: Path) -> None:
    assert (
        cli._server_origin(  # noqa: SLF001
            host="0.0.0.0", port=8000, public_url="https://tulpa.example/"
        )
        == "https://tulpa.example"
    )
    code = cli._private_pairing_code(tmp_path / "pairing-code")  # noqa: SLF001
    assert len(code) == 17
    assert code.count("-") == 2


def test_telegram_bootstrap_from_environment_uses_first_numeric_allowed_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:telegram-secret")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "7,8")

    assert cli._telegram_bootstrap_from_environment() == ("123:telegram-secret", 7)  # noqa: SLF001


def test_runtime_probation_environment_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENTULPA_RUNTIME_PROBATION_SECONDS", raising=False)
    monkeypatch.delenv("OPENTULPA_RUNTIME_PROBATION_PROBE_INTERVAL_SECONDS", raising=False)
    assert cli._runtime_probation_settings() == (30, 1)  # noqa: SLF001
    monkeypatch.setenv("OPENTULPA_RUNTIME_PROBATION_SECONDS", "4.5")
    monkeypatch.setenv("OPENTULPA_RUNTIME_PROBATION_PROBE_INTERVAL_SECONDS", "0.25")
    assert cli._runtime_probation_settings() == (4.5, 0.25)  # noqa: SLF001


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENTULPA_RUNTIME_PROBATION_SECONDS", "-1"),
        ("OPENTULPA_RUNTIME_PROBATION_SECONDS", "nan"),
        ("OPENTULPA_RUNTIME_PROBATION_PROBE_INTERVAL_SECONDS", "0"),
    ],
)
def test_runtime_probation_environment_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        cli._runtime_probation_settings()  # noqa: SLF001


def test_native_tui_receives_credentials_only_through_inherited_pipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "opentulpa-tui"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o700)
    captured: dict[str, object] = {}

    class Process:
        returncode = 0

        def __init__(
            self,
            args: list[str],
            *,
            env: dict[str, str],
            pass_fds: tuple[int, ...],
        ) -> None:
            captured["args"] = args
            captured["env"] = env
            captured["pass_fds"] = pass_fds
            self.connection_fd = os.dup(int(env["OPENTULPA_CONNECTION_FD"]))
            self.state_fd = os.dup(int(env["OPENTULPA_STATE_FD"]))

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            with os.fdopen(self.connection_fd, "rb") as stream:
                captured["connection"] = json.loads(stream.read())
            with os.fdopen(self.state_fd, "wb") as stream:
                stream.write(b'{"thread_id":"thread-2"}')
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

    connection = Connection(
        url="https://tulpa.example",
        token="owner-secret",
        thread_id="thread-1",
        credential_storage="keyring",
    )
    updated: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "_ensure_tui_binary", lambda: binary)
    monkeypatch.setattr(cli, "_migrate_legacy_session_names", lambda _: None)
    monkeypatch.setattr(cli.subprocess, "Popen", Process)
    monkeypatch.setattr(
        cli,
        "update_connection",
        lambda value, **changes: updated.append((value.thread_id, changes["thread_id"])),
    )

    cli._launch_tui(connection)  # noqa: SLF001

    assert captured["args"] == [str(binary)]
    assert "owner-secret" not in " ".join(captured["args"])
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "owner-secret" not in environment.values()
    assert captured["connection"] == {
        "url": "https://tulpa.example",
        "token": "owner-secret",
        "thread_id": "thread-1",
        "credential_storage": "keyring",
    }
    assert updated == [("thread-1", "thread-2")]


def test_source_checkout_prefers_its_built_tui_over_an_installed_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    system = {"Darwin": "darwin", "Linux": "linux"}[platform.system()]
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64"}[
        platform.machine()
    ]
    project = tmp_path / "project"
    module = project / "src" / "opentulpa" / "host" / "cli.py"
    module.parent.mkdir(parents=True)
    local_binary = project / "clients" / "tui" / "dist" / f"opentulpa-tui-{system}-{machine}"
    local_binary.parent.mkdir(parents=True)
    local_binary.write_text("local", encoding="utf-8")
    local_binary.chmod(0o700)
    (local_binary.parent / "manifest.json").write_text(
        '{"protocol_version":2,"source_digest":"source-digest"}',
        encoding="utf-8",
    )
    installed_binary = tmp_path / "bin" / "opentulpa-tui"
    installed_binary.parent.mkdir()
    installed_binary.write_text("installed", encoding="utf-8")
    installed_binary.chmod(0o700)
    monkeypatch.setattr(cli, "__file__", str(module))
    monkeypatch.setattr(cli.shutil, "which", lambda _: str(installed_binary))
    monkeypatch.setattr(cli, "_tui_source_digest", lambda _: "source-digest")
    monkeypatch.setattr(cli, "_tui_protocol", lambda _: "2")

    assert cli._ensure_tui_binary() == local_binary  # noqa: SLF001


def test_host_paths_provision_separate_control_and_product_roots(
    tmp_path: Path,
) -> None:
    paths = HostPaths.from_environment({"OPENTULPA_DATA_ROOT": str(tmp_path / "data")})

    paths.provision()
    paths.provision()

    assert paths.control_root.stat().st_mode & 0o777 == 0o700
    assert paths.product_root.stat().st_mode & 0o777 == 0o700
    assert paths.runtime_control_path.parent == paths.control_root
    assert paths.notification_store_path == paths.product_root / ".opentulpa/notifications.db"


def test_host_paths_migrate_known_legacy_product_entries(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    legacy = data_root / ".opentulpa" / "deepagents"
    legacy.mkdir(parents=True)
    (legacy / "store.db").write_bytes(b"state")
    (data_root / "customer_profiles.db").write_bytes(b"profiles")
    (data_root / "file_vault").mkdir()
    (data_root / "file_vault" / "doc.txt").write_text("doc", encoding="utf-8")
    (data_root / "file_vault.db").write_bytes(b"files")
    (data_root / "knowledge").mkdir()
    (data_root / "knowledge" / "knowledge.db").write_bytes(b"knowledge")
    (data_root / "telegram_business.db").write_bytes(b"telegram")
    (data_root / "intake_workflows.db").write_bytes(b"intake")
    (data_root / "intake_sinks").mkdir()
    (data_root / "intake_sinks" / "sink.json").write_text("{}", encoding="utf-8")
    paths = HostPaths.from_environment({"OPENTULPA_DATA_ROOT": str(data_root)})

    paths.provision()

    assert not (data_root / ".opentulpa").exists()
    assert (paths.product_root / ".opentulpa/deepagents/store.db").read_bytes() == b"state"
    assert (paths.product_root / "customer_profiles.db").read_bytes() == b"profiles"
    assert (paths.product_root / "file_vault" / "doc.txt").read_text(encoding="utf-8") == "doc"
    assert (paths.product_root / "file_vault.db").read_bytes() == b"files"
    assert (paths.product_root / "knowledge" / "knowledge.db").read_bytes() == b"knowledge"
    assert (paths.product_root / "telegram_business.db").read_bytes() == b"telegram"
    assert (paths.product_root / "intake_workflows.db").read_bytes() == b"intake"
    assert (paths.product_root / "intake_sinks" / "sink.json").read_text(
        encoding="utf-8"
    ) == "{}"


def test_host_paths_archive_conflicting_top_level_legacy_notifications(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    state = data_root / ".opentulpa"
    state.mkdir(parents=True)
    (state / "notifications.db").write_bytes(b"state notifications")
    (data_root / "notifications.db").write_bytes(b"top notifications")
    paths = HostPaths.from_environment({"OPENTULPA_DATA_ROOT": str(data_root)})

    paths.provision()

    assert (paths.product_root / ".opentulpa/notifications.db").read_bytes() == b"state notifications"
    assert (
        paths.product_root / ".opentulpa/notifications.legacy-from-data-root.db"
    ).read_bytes() == b"top notifications"


def test_host_paths_allow_known_controller_entries_during_product_migration(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    for name in (
        "bootstrap",
        "bun",
        "install",
        "lost+found",
        ".runtime-generations-control",
        "runtime-generations",
        "runtime-source-envs",
        "sandbox-host",
        "sandbox_worker",
        "source",
    ):
        (data_root / name).mkdir(parents=True)
    paths = HostPaths.from_environment({"OPENTULPA_DATA_ROOT": str(data_root)})

    paths.provision()

    for name in (
        "bootstrap",
        "bun",
        "install",
        "lost+found",
        ".runtime-generations-control",
        "runtime-generations",
        "runtime-source-envs",
        "sandbox-host",
        "sandbox_worker",
        "source",
    ):
        assert (data_root / name).is_dir()


def test_host_paths_honor_safe_control_and_product_overrides(tmp_path: Path) -> None:
    paths = HostPaths.from_environment(
        {
            "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
            "OPENTULPA_CONTROL_ROOT": str(tmp_path / "controller"),
            "OPENTULPA_PRODUCT_ROOT": str(tmp_path / "mutable-product"),
        }
    )

    paths.provision()

    assert paths.control_root == tmp_path / "controller"
    assert paths.product_root == tmp_path / "mutable-product"
    assert paths.runtime_control_path.parent == tmp_path / "controller"


@pytest.mark.parametrize("unsafe", ["unknown", "symlink"])
def test_host_paths_reject_ambiguous_or_linked_legacy_entries(
    tmp_path: Path,
    unsafe: str,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    if unsafe == "unknown":
        (data_root / "mystery.db").write_bytes(b"unknown")
    else:
        (data_root / ".opentulpa").symlink_to(tmp_path)
    paths = HostPaths.from_environment({"OPENTULPA_DATA_ROOT": str(data_root)})

    with pytest.raises(HostPathError, match="symbolic-link|unknown ambiguous"):
        paths.provision()


def test_host_application_root_uses_package_resources_without_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_module = tmp_path / "site-packages" / "opentulpa" / "host" / "cli.py"
    installed_module.parent.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(installed_module))
    monkeypatch.delenv("OPENTULPA_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("OPENTULPA_INSTALL_ASSETS_ROOT", raising=False)

    root = cli._host_application_root()  # noqa: SLF001

    assert (root / "resources" / "release_contract.json").is_file()
    assert not cli._is_source_checkout(root)  # noqa: SLF001


def test_host_application_root_clones_configured_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "data" / "source"
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        assert kwargs["check"] is False
        clone_target = Path(command[-1])
        (clone_target / "src" / "opentulpa").mkdir(parents=True)
        (clone_target / "src" / "opentulpa" / "__init__.py").write_text("", encoding="utf-8")
        (clone_target / ".git").mkdir()
        (clone_target / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (clone_target / "uv.lock").write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("OPENTULPA_SOURCE_ROOT", str(source))
    monkeypatch.setenv("EVOLUTION_SOURCE_REPOSITORY", "https://example.test/opentulpa.git")
    monkeypatch.setenv("OPENTULPA_INSTALL_REF", "release/ref")
    monkeypatch.setattr(cli.subprocess, "run", run)

    root = cli._host_application_root()  # noqa: SLF001

    assert root == source.resolve()
    assert (source / ".git").is_dir()
    assert commands == [
        [
            "git",
            "clone",
            "--branch",
            "release/ref",
            "--single-branch",
            "https://example.test/opentulpa.git",
            str(commands[0][-1]),
        ]
    ]
    assert str(commands[0][-1]).startswith(str(source.parent / f".{source.name}.clone-"))


def test_host_application_root_rejects_empty_configured_source_without_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "data" / "source"
    source.mkdir(parents=True)
    monkeypatch.setenv("OPENTULPA_SOURCE_ROOT", str(source))
    monkeypatch.delenv("EVOLUTION_SOURCE_REPOSITORY", raising=False)

    with pytest.raises(RuntimeError, match="EVOLUTION_SOURCE_REPOSITORY"):
        cli._host_application_root()  # noqa: SLF001


def test_server_reports_live_source_required_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*, public_url: str | None) -> None:
        del public_url
        raise cli.HostLiveSourceRequiredError("source checkout is missing")

    monkeypatch.setattr(cli, "serve", fail)

    with pytest.raises(SystemExit, match="live_source_required: source checkout is missing"):
        cli._server_command(  # noqa: SLF001
            Namespace(host="127.0.0.1", port=8000, public_url=None)
        )


def _installed_controller_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    managed_source: bool,
) -> tuple[Path, Path]:
    install_root = tmp_path / "install root"
    controller = install_root / "controller"
    controller.mkdir(parents=True)
    source = tmp_path / "source checkout"
    source.mkdir()
    metadata = {
        "format_version": 1,
        "generation_id": "a" * 64,
        "managed_source": managed_source,
        "ref": "release/ref",
        "repository": "https://example.test/opentulpa.git",
        "source_root": str(source),
    }
    metadata_path = controller / "install.json"
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    metadata_path.chmod(0o600)
    installer = controller / "installer.sh"
    installer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    installer.chmod(0o700)
    generation = controller / "generations" / ("b" * 64)
    generation.mkdir(parents=True)
    executable = generation / "bin" / "opentulpa-host"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o500)
    (controller / "current").symlink_to(Path("generations") / generation.name)
    monkeypatch.setenv("OPENTULPA_INSTALL_ROOT", str(install_root))
    return controller, source


def test_update_invokes_recorded_managed_installer_with_fetch_and_no_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller, _ = _installed_controller_metadata(tmp_path, monkeypatch, managed_source=True)
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)

    result = cli._update_command(  # noqa: SLF001
        Namespace(source=None, fetch=True, restart_local_host=False)
    )

    assert result == 0
    assert captured["command"] == [str(controller / "installer.sh"), "--fetch"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert "shell" not in kwargs
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["OPENTULPA_INSTALL_ROOT"] == str(controller.parent)
    assert environment["OPENTULPA_INSTALL_REF"] == "release/ref"
    assert f"Activated OpenTulpa controller generation {'b' * 64}." in capsys.readouterr().out


def test_update_imports_explicit_clean_source_as_argv_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _ = _installed_controller_metadata(tmp_path, monkeypatch, managed_source=False)
    manual = tmp_path / "manual source with spaces"
    manual.mkdir()
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)

    result = cli._update_command(  # noqa: SLF001
        Namespace(source=str(manual), fetch=False, restart_local_host=False)
    )

    assert result == 0
    assert commands == [[str(controller / "installer.sh"), "--source", str(manual.resolve())]]


def test_failed_update_returns_installer_status_and_keeps_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller, source = _installed_controller_metadata(tmp_path, monkeypatch, managed_source=False)
    original = (controller / "current").resolve()
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=17),
    )

    result = cli._update_command(  # noqa: SLF001
        Namespace(source=None, fetch=False, restart_local_host=False)
    )

    assert result == 17
    assert (controller / "current").resolve() == original
    output = capsys.readouterr()
    assert str(source) not in output.err
    assert "active controller was unchanged" in output.err


def test_update_rejects_fetch_for_explicit_source_without_starting_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_controller_metadata(tmp_path, monkeypatch, managed_source=True)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("installer must not run"),
    )

    result = cli._update_command(  # noqa: SLF001
        Namespace(source=str(tmp_path / "manual"), fetch=True, restart_local_host=False)
    )

    assert result == 2


def test_update_restart_uses_new_controller_generation_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _ = _installed_controller_metadata(tmp_path, monkeypatch, managed_source=True)
    generation_id = "b" * 64
    restarted: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        cli,
        "restart_remembered_local_server",
        lambda *, controller_executable, controller_generation_id: restarted.append(
            (controller_executable, controller_generation_id)
        )
        or "http://127.0.0.1:8000",
    )

    result = cli._update_command(  # noqa: SLF001
        Namespace(source=None, fetch=False, restart_local_host=True)
    )

    assert result == 0
    assert restarted == [
        (controller / "generations" / generation_id / "bin" / "opentulpa-host", generation_id)
    ]
