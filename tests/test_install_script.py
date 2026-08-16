from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    "opentulpa",
    "opentulpa-host",
    "opentulpa-sandbox-worker",
    "opentulpa-migrate-deepagents",
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _source(root: Path, *, commit: str = "1" * 40) -> Path:
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".commit").write_text(commit, encoding="ascii")
    (root / ".verified").write_text(commit, encoding="ascii")
    (root / f".verified-{hashlib.sha256(b'main').hexdigest()}").write_text(
        commit, encoding="ascii"
    )
    (root / ".remote").write_text(
        "https://github.com/kvyb/opentulpa.git\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='opentulpa'\nversion='0.1.0'\n", encoding="utf-8"
    )
    (root / "uv.lock").write_text(f"version = 1\ncommit = '{commit}'\n", encoding="utf-8")
    (root / "install.sh").write_text(
        (REPO_ROOT / "install.sh").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "controller_generation.py").write_text(
        (REPO_ROOT / "controller_generation.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    bridge = root / "railway_sandbox_bridge"
    bridge.mkdir()
    (bridge / "bridge.mjs").write_text("export {}\n", encoding="utf-8")
    (bridge / "package.json").write_text("{}\n", encoding="utf-8")
    (bridge / "package-lock.json").write_text("{}\n", encoding="utf-8")
    (root / "opentulpa.config.yaml").write_text("host: 127.0.0.1\n", encoding="utf-8")
    return root


def _fake_tools(root: Path) -> tuple[Path, Path]:
    tools = root / "fake tools"
    tools.mkdir()
    calls = root / "tool-calls.jsonl"
    _write_executable(
        tools / "git",
        f"""#!{sys.executable}
import os
import pathlib
import sys
import tarfile

args = sys.argv[1:]
with pathlib.Path({str(calls)!r}).open("a", encoding="utf-8") as stream:
    stream.write("git " + " ".join(args) + "\\n")
if args[:2] == ["clone", "--branch"]:
    destination = pathlib.Path(args[-1])
    destination.mkdir(parents=True)
    (destination / ".git").mkdir()
    (destination / ".remote").write_text(args[-2], encoding="utf-8")
    (destination / ".commit").write_text("1" * 40, encoding="ascii")
    (destination / ".verified").write_text("1" * 40, encoding="ascii")
    raise SystemExit(0)
source = pathlib.Path(args[1]) if args and args[0] == "-C" else pathlib.Path.cwd()
command = args[2] if len(args) > 2 and args[0] == "-C" else args[0]
def verified_path(reference: str) -> pathlib.Path:
    return source / (".verified-" + reference.removesuffix("^{{commit}}").rsplit("/", 1)[-1])

if command == "status":
    marker = source / ".dirty"
    if marker.exists():
        print(" M tracked.py")
elif command == "rev-parse":
    selected = verified_path(args[-1]) if "refs/opentulpa/install/verified" in args[-1] else source / ".commit"
    if not selected.exists():
        raise SystemExit(1)
    print(selected.read_text(encoding="ascii").strip())
elif command == "remote":
    print((source / ".remote").read_text(encoding="utf-8").strip())
elif command == "update-ref":
    value = (source / ".commit").read_text(encoding="ascii").strip()
    (source / ".verified").write_text(value, encoding="ascii")
    verified_path(args[3]).write_text(value, encoding="ascii")
elif command == "fetch":
    value_path = source / ".fetched-commit"
    value = value_path.read_text(encoding="ascii").strip() if value_path.exists() else (source / ".commit").read_text(encoding="ascii").strip()
    (source / ".verified").write_text(value, encoding="ascii")
    verified_path(args[-1].split(":", 1)[1]).write_text(value, encoding="ascii")
elif command == "merge-base":
    if (source / ".non-fast-forward").exists():
        raise SystemExit(1)
elif command == "merge":
    value = (source / ".verified").read_text(encoding="ascii").strip()
    (source / ".commit").write_text(value, encoding="ascii")
elif command == "archive":
    excluded = {{".git", ".commit", ".verified", ".remote", ".dirty", ".fetched-commit", ".non-fast-forward"}}
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part in excluded or part.startswith(".verified-") for part in relative.parts):
                continue
            archive.add(path, arcname=relative, recursive=False)
""",
    )
    _write_executable(
        tools / "pip",
        f"""#!{sys.executable}
import pathlib
import sys

args = sys.argv[1:]
with pathlib.Path({str(calls)!r}).open("a", encoding="utf-8") as stream:
    stream.write("pip " + " ".join(args) + "\\n")
destination = pathlib.Path(args[args.index("--dest") + 1])
destination.mkdir(parents=True, exist_ok=True)
(destination / "dependency-1.0-py3-none-any.whl").write_bytes(b"locked dependency wheel")
""",
    )
    _write_executable(
        tools / "uv",
        f"""#!{sys.executable}
import os
import pathlib
import platform
import shlex
import sys
import time

COMMANDS = {COMMANDS!r}
args = sys.argv[1:]
with pathlib.Path({str(calls)!r}).open("a", encoding="utf-8") as stream:
    stream.write("uv " + " ".join(args) + "\\n")
if args[:2] == ["python", "find"]:
    print({sys.executable!r})
elif args and args[0] == "build":
    if os.environ.get("FAKE_UV_DELAY"):
        time.sleep(float(os.environ["FAKE_UV_DELAY"]))
    output = pathlib.Path(args[args.index("--out-dir") + 1])
    source = pathlib.Path(args[-1])
    output.mkdir(parents=True, exist_ok=True)
    lock = (source / "uv.lock").read_bytes()
    (output / "opentulpa-0.1.0-py3-none-any.whl").write_bytes(lock)
elif args and args[0] == "export":
    output = pathlib.Path(args[args.index("--output-file") + 1])
    output.write_text(
        "dependency==1.0 --hash=sha256:" + "a" * 64 + "\\n", encoding="utf-8"
    )
elif args and args[0] == "venv":
    destination = pathlib.Path(args[-1])
    (destination / "bin").mkdir(parents=True, exist_ok=True)
    (destination / "pyvenv.cfg").write_text(
        "home = " + str(pathlib.Path({sys.executable!r}).resolve().parent) + "\\n"
        "include-system-site-packages = false\\n"
        "version = " + platform.python_version() + "\\n",
        encoding="utf-8",
    )
    python = destination / "bin" / "python"
    if not python.exists():
        python.symlink_to({sys.executable!r})
elif args[:2] == ["pip", "install"]:
    if os.environ.get("FAKE_INSTALL_FAILURE") == "1" and any(
        value.endswith(".whl") for value in args
    ):
        raise SystemExit(19)
    if not any(value.endswith(".whl") and "opentulpa" in value for value in args):
        raise SystemExit(0)
    python = pathlib.Path(args[args.index("--python") + 1])
    generation = python.parent.parent
    version = f"python{{sys.version_info.major}}.{{sys.version_info.minor}}"
    site = generation / "lib" / version / "site-packages"
    package = site / "opentulpa"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "resources").mkdir()
    (package / "resources" / "release_contract.json").write_text("{{}}\\n", encoding="utf-8")
    (package / "fake_entrypoints.py").write_text(
        "def main():\\n    return None\\n", encoding="utf-8"
    )
    dist = site / "opentulpa-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\\nName: opentulpa\\nVersion: 0.1.0\\n", encoding="utf-8"
    )
    entries = "[console_scripts]\\n" + "".join(
        f"{{name}} = opentulpa.fake_entrypoints:main\\n" for name in COMMANDS
    )
    (dist / "entry_points.txt").write_text(entries, encoding="utf-8")
    for name in COMMANDS:
        script = generation / "bin" / name
        interpreter = generation / "bin" / "python"
        launcher = f"#!{{interpreter}}\\n"
        if os.environ.get("FAKE_UV_LONG_SHEBANG") == "1":
            launcher = (
                "#!/bin/sh\\n'''exec' "
                + shlex.quote(str(interpreter))
                + ' "$0" "$@"\\n'
                + "' '''\\n"
            )
        script.write_text(
            launcher +
            "import json, os, pathlib, sys\\n"
            "if len(sys.argv) == 3 and sys.argv[1] == '--probe':\\n"
            "    pathlib.Path(sys.argv[2]).write_text(json.dumps({{k: os.environ.get(k) for k in "
            "['OPENTULPA_INSTALL_ROOT', 'OPENTULPA_SOURCE_SEED_ROOT', "
            "'OPENTULPA_TRUSTED_WHEELHOUSE', 'OPENTULPA_INSTALL_ASSETS_ROOT', "
            "'OPENTULPA_UV_BIN', 'OPENTULPA_SOURCE_SEED_OID', "
            "'OPENTULPA_SOURCE_SEED_SHA256', 'OPENTULPA_SOURCE_SEED_TREE_OID']}}), "
            "encoding='utf-8')\\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
""",
    )
    tui_script = '#!/bin/sh\n[ "$1" = --protocol-version ] && printf \'2\\n\'\n'
    bun_template = tools / "bun-template"
    _write_executable(
        bun_template,
        f"""#!{sys.executable}
import pathlib
import platform
import subprocess
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("1.3.14")
elif args == ["run", "build"]:
    subprocess.run(["bun", "run", "build.ts"], check=True)
elif args == ["run", "build.ts"]:
    target = {{
        ("Darwin", "arm64"): "darwin-arm64",
        ("Darwin", "x86_64"): "darwin-x64",
        ("Linux", "aarch64"): "linux-arm64",
        ("Linux", "arm64"): "linux-arm64",
        ("Linux", "x86_64"): "linux-x64",
    }}[(platform.system(), platform.machine())]
    output = pathlib.Path("dist") / f"opentulpa-tui-{{target}}"
    output.parent.mkdir()
    output.write_text({tui_script!r}, encoding="utf-8")
    output.chmod(0o755)
""",
    )
    install_bun = f'cp {shlex.quote(str(bun_template))} "$BUN_INSTALL/bin/bun"'
    _write_executable(
        tools / "curl",
        f"""#!{sys.executable}
print('#!/bin/sh')
print('mkdir -p "$BUN_INSTALL/bin"')
print({install_bun!r})
print('chmod 755 "$BUN_INSTALL/bin/bun"')
""",
    )
    return tools, calls


def _run_install(
    tmp_path: Path,
    source: Path,
    tools: Path,
    *,
    extra_env: dict[str, str] | None = None,
    explicit_source: bool = True,
    installer_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    install_root = tmp_path / "install root"
    bin_root = tmp_path / "command bin"
    bin_root.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{tools}:/usr/bin:/bin",
        "OPENTULPA_BIN_DIR": str(bin_root),
        "OPENTULPA_INSTALL_ROOT": str(install_root),
        "OPENTULPA_PIP_BIN": str(tools / "pip"),
    }
    if explicit_source:
        environment["OPENTULPA_INSTALL_SOURCE"] = str(source)
        installer = REPO_ROOT / "install.sh"
    else:
        environment.pop("OPENTULPA_INSTALL_SOURCE", None)
        installer = tmp_path / "managed-installer.sh"
        installer.write_text((REPO_ROOT / "install.sh").read_text(encoding="utf-8"), encoding="utf-8")
        installer.chmod(0o755)
    environment.update(extra_env or {})
    return subprocess.run(
        ["sh", str(installer), *installer_args],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _current_generation(tmp_path: Path) -> Path:
    return (tmp_path / "install root" / "controller" / "current").resolve()


def test_installer_builds_final_path_controller_and_exact_dispatcher(tmp_path: Path) -> None:
    source = _source(tmp_path / "source checkout")
    tools, calls = _fake_tools(tmp_path)

    result = _run_install(tmp_path, source, tools)

    assert result.returncode == 0, result.stderr
    generation = _current_generation(tmp_path)
    assert generation.parent.name == "generations"
    assert len(generation.name) == 64
    assert (generation / "COMPLETE").read_bytes() == b""
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_id"] == generation.name
    assert manifest["identity"]["source_commit"] == "1" * 40
    assert manifest["identity"]["install_profile"] == "controller-runtime-no-dev"
    assert len(manifest["runtime_tree_sha256"]) == 64
    assert manifest["source"]["kind"] == "local"
    assert manifest["source"]["oid"] == "1" * 40
    for command in COMMANDS:
        assert (generation / "bin" / command).read_text(encoding="utf-8").splitlines()[0] == (
            f"#!{generation / 'bin' / 'python'}"
        )
    assert not list(generation.rglob("*.pth"))
    assert list((generation / "wheelhouse").glob("*.whl"))
    assert (generation / "source-seed" / "uv.lock").is_file()
    assert not (generation / "source-seed").is_symlink()
    source_manifest = json.loads(
        (generation / "source-seed-manifest.json").read_text(encoding="utf-8")
    )
    assert source_manifest["source_commit"] == "1" * 40
    assert source_manifest["source_tree_oid"] == "1" * 40
    assert source_manifest["source_seed_sha256"] == manifest["identity"]["source_seed_sha256"]

    probe = tmp_path / "dispatcher environment.json"
    dispatcher = tmp_path / "command bin" / "opentulpa"
    dispatched = subprocess.run(
        [str(dispatcher), "--probe", str(probe)], capture_output=True, text=True, check=False
    )
    assert dispatched.returncode == 0, dispatched.stderr
    environment = json.loads(probe.read_text(encoding="utf-8"))
    assert environment == {
        "OPENTULPA_INSTALL_ROOT": str(tmp_path / "install root"),
        "OPENTULPA_SOURCE_SEED_ROOT": str(generation / "source-seed"),
        "OPENTULPA_TRUSTED_WHEELHOUSE": str(generation / "wheelhouse"),
        "OPENTULPA_INSTALL_ASSETS_ROOT": str(generation / "assets"),
        "OPENTULPA_UV_BIN": str(generation / "assets" / "toolchain" / "uv"),
        "OPENTULPA_SOURCE_SEED_OID": "1" * 40,
        "OPENTULPA_SOURCE_SEED_SHA256": source_manifest["source_seed_sha256"],
        "OPENTULPA_SOURCE_SEED_TREE_OID": "1" * 40,
    }
    logged = calls.read_text(encoding="utf-8")
    assert "uv build --wheel" in logged
    assert "uv export --frozen --no-dev --no-emit-project" in logged
    assert "--extra evaluation" not in logged
    assert "pip download --disable-pip-version-check --require-hashes --only-binary=:all:" in logged
    assert "uv sync" not in logged


def test_installer_accepts_uv_long_shebang_wrapper(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)

    result = _run_install(
        tmp_path,
        source,
        tools,
        extra_env={"FAKE_UV_LONG_SHEBANG": "1"},
    )

    assert result.returncode == 0, result.stderr
    generation = _current_generation(tmp_path)
    assert (generation / "bin" / "opentulpa").read_text().startswith(
        "#!/bin/sh\n'''exec' "
    )
    probe = tmp_path / "long shebang environment.json"
    dispatched = subprocess.run(
        [str(tmp_path / "command bin" / "opentulpa"), "--probe", str(probe)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dispatched.returncode == 0, dispatched.stderr

    reused = _run_install(
        tmp_path,
        source,
        tools,
        extra_env={"FAKE_UV_LONG_SHEBANG": "1"},
    )
    assert reused.returncode == 0, reused.stderr
    assert _current_generation(tmp_path) == generation


def test_installer_builds_tui_with_downloaded_bun(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tui = source / "clients" / "tui"
    tui.mkdir(parents=True)
    (tui / "package.json").write_text("{}\n", encoding="utf-8")
    tools, _ = _fake_tools(tmp_path)

    result = _run_install(tmp_path, source, tools)

    assert result.returncode == 0, result.stderr
    assert list((_current_generation(tmp_path) / "assets" / "tui").iterdir())


def test_installer_reuses_identity_and_atomically_tracks_previous(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    first = _run_install(tmp_path, source, tools)
    assert first.returncode == 0, first.stderr
    first_generation = _current_generation(tmp_path)

    reused = _run_install(tmp_path, source, tools)
    assert reused.returncode == 0, reused.stderr
    assert _current_generation(tmp_path) == first_generation
    assert len(list(first_generation.parent.iterdir())) == 1
    assert not (tmp_path / "install root" / "controller" / "previous").exists()

    second_commit = "2" * 40
    (source / ".commit").write_text(second_commit, encoding="ascii")
    (source / "uv.lock").write_text(f"commit = '{second_commit}'\n", encoding="utf-8")
    second = _run_install(tmp_path, source, tools)
    assert second.returncode == 0, second.stderr
    second_generation = _current_generation(tmp_path)
    assert second_generation != first_generation
    previous = (tmp_path / "install root" / "controller" / "previous").resolve()
    assert previous == first_generation

    third_commit = "3" * 40
    (source / ".commit").write_text(third_commit, encoding="ascii")
    (source / "uv.lock").write_text(f"commit = '{third_commit}'\n", encoding="utf-8")
    failed = _run_install(tmp_path, source, tools, extra_env={"FAKE_INSTALL_FAILURE": "1"})
    assert failed.returncode == 19
    assert _current_generation(tmp_path) == second_generation
    assert (tmp_path / "install root" / "controller" / "previous").resolve() == first_generation
    assert len(list(second_generation.parent.iterdir())) == 2


def test_installer_rejects_dirty_source_and_regular_command_without_activation(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    (source / ".dirty").write_text("dirty", encoding="utf-8")

    dirty = _run_install(tmp_path, source, tools)

    assert dirty.returncode == 1
    assert "source checkout is dirty" in dirty.stderr
    assert not (tmp_path / "install root" / "controller" / "current").exists()

    (source / ".dirty").unlink()
    command = tmp_path / "command bin" / "opentulpa"
    command.write_text("do not replace\n", encoding="utf-8")
    refused = _run_install(tmp_path, source, tools)
    assert refused.returncode == 1
    assert "refusing to replace existing regular file" in refused.stderr
    assert command.read_text(encoding="utf-8") == "do not replace\n"
    assert not (tmp_path / "install root" / "controller" / "current").exists()


def test_installer_imports_new_exact_clean_local_commit(tmp_path: Path) -> None:
    source = _source(tmp_path / "manual checkout")
    tools, _ = _fake_tools(tmp_path)
    first = _run_install(tmp_path, source, tools)
    assert first.returncode == 0, first.stderr

    new_commit = "a" * 40
    (source / ".commit").write_text(new_commit, encoding="ascii")
    (source / "uv.lock").write_text(f"commit = '{new_commit}'\n", encoding="utf-8")
    updated = _run_install(tmp_path, source, tools)

    assert updated.returncode == 0, updated.stderr
    manifest = json.loads(
        (_current_generation(tmp_path) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["identity"]["source_commit"] == new_commit


def test_installer_serializes_concurrent_identical_and_different_generations(
    tmp_path: Path,
) -> None:
    first_source = _source(tmp_path / "first source", commit="1" * 40)
    second_source = _source(tmp_path / "second source", commit="2" * 40)
    tools, _ = _fake_tools(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            _run_install,
            tmp_path,
            first_source,
            tools,
            extra_env={"FAKE_UV_DELAY": "1"},
        )
        second_future = executor.submit(_run_install, tmp_path, second_source, tools)
        first = first_future.result(timeout=30)
        second = second_future.result(timeout=30)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    controller = tmp_path / "install root" / "controller"
    current = (controller / "current").resolve()
    previous = (controller / "previous").resolve()
    assert current != previous
    for generation in (current, previous):
        manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["generation_id"] == generation.name
        assert (generation / "COMPLETE").read_bytes() == b""

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: _run_install(tmp_path, second_source, tools),
                range(2),
            )
        )
    assert all(result.returncode == 0 for result in results), [
        (result.returncode, result.stdout, result.stderr) for result in results
    ]
    assert len(list((controller / "generations").iterdir())) == 2


def test_installer_does_not_reclaim_existing_lock(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    lock = tmp_path / "install root" / "controller" / "install.lock"
    lock.parent.mkdir(parents=True, mode=0o700)
    lock.mkdir(mode=0o700)

    result = _run_install(
        tmp_path,
        source,
        tools,
        extra_env={"OPENTULPA_INSTALL_LOCK_WAIT_SECONDS": "0"},
    )

    assert result.returncode == 1
    assert "timed out waiting" in result.stderr
    assert lock.is_dir()


def test_installer_lock_acquisition_has_no_ownerless_directory_window(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    lock = tmp_path / "install root" / "controller" / "install.lock"
    lock.mkdir(parents=True, mode=0o700)

    result = _run_install(
        tmp_path,
        source,
        tools,
        extra_env={"OPENTULPA_INSTALL_LOCK_WAIT_SECONDS": "0"},
    )

    assert result.returncode == 1
    assert "timed out waiting" in result.stderr
    installer = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    assert 'mkdir "$INSTALL_LOCK"' in installer
    assert "os.link" not in installer
    assert "os.kill" not in installer
    assert "stale" not in installer


def test_installer_rejects_nonempty_lock_without_reclaiming_it(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    lock = tmp_path / "install root" / "controller" / "install.lock"
    lock.mkdir(parents=True, mode=0o700)
    (lock / "operator-note").write_text("keep", encoding="utf-8")

    result = _run_install(tmp_path, source, tools)

    assert result.returncode == 1
    assert "install lock is unsafe" in result.stderr
    assert (lock / "operator-note").read_text(encoding="utf-8") == "keep"


def test_installer_rejects_unsafe_lock_path_without_reclaiming_it(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    lock = tmp_path / "install root" / "controller" / "install.lock"
    lock.parent.mkdir(parents=True, mode=0o700)
    lock.symlink_to(tmp_path)

    result = _run_install(tmp_path, source, tools)

    assert result.returncode == 1
    assert "install lock is unsafe" in result.stderr
    assert lock.is_symlink()


def test_installer_rebuilds_runtime_tree_after_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    installed = _run_install(tmp_path, source, tools)
    assert installed.returncode == 0, installed.stderr
    generation = _current_generation(tmp_path)
    asset = generation / "assets" / "railway_sandbox_bridge" / "bridge.mjs"
    asset.chmod(0o600)
    asset.write_text("tampered\n", encoding="utf-8")

    repaired = _run_install(tmp_path, source, tools)

    assert repaired.returncode == 0, repaired.stderr
    assert _current_generation(tmp_path) == generation
    assert asset.read_text(encoding="utf-8") == "export {}\n"


def test_dispatcher_rejects_current_pointer_escape(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    installed = _run_install(tmp_path, source, tools)
    assert installed.returncode == 0, installed.stderr
    current = tmp_path / "install root" / "controller" / "current"
    current.unlink()
    current.symlink_to(source)

    dispatched = subprocess.run(
        [str(tmp_path / "command bin" / "opentulpa")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert dispatched.returncode != 0
    assert "escapes the generation store" in dispatched.stderr


def test_dispatcher_rejects_tampered_generation_python_before_execution(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    installed = _run_install(tmp_path, source, tools)
    assert installed.returncode == 0, installed.stderr
    generation = _current_generation(tmp_path)
    python = generation / "bin" / "python"
    marker = tmp_path / "tampered-python-ran"
    generation.chmod(0o700)
    python.parent.chmod(0o700)
    python.unlink()
    _write_executable(python, f"#!/bin/sh\ntouch '{marker}'\nexit 99\n")

    dispatched = subprocess.run(
        [str(tmp_path / "command bin" / "opentulpa")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert dispatched.returncode == 1
    assert "failed launch validation" in dispatched.stderr
    assert not marker.exists()


def test_managed_source_requires_matching_origin_and_verified_ref(tmp_path: Path) -> None:
    source = _source(tmp_path / "install root" / "source")
    tools, _ = _fake_tools(tmp_path)
    installed = _run_install(tmp_path, source, tools, explicit_source=False)
    assert installed.returncode == 0, installed.stderr
    manifest = json.loads(
        (_current_generation(tmp_path) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["kind"] == "managed"
    assert manifest["source"]["actual_remote"] == "https://github.com/kvyb/opentulpa.git"
    assert manifest["source"]["configured_ref"] == "main"

    (source / ".remote").write_text("https://evil.example/repository.git\n", encoding="utf-8")
    mismatch = _run_install(tmp_path, source, tools, explicit_source=False)
    assert mismatch.returncode == 1
    assert "managed source origin does not match" in mismatch.stderr

    (source / ".remote").write_text(
        "https://github.com/kvyb/opentulpa.git\n",
        encoding="utf-8",
    )
    private_ref = next(source.glob(".verified-*"))
    private_ref.write_text("f" * 40, encoding="ascii")
    ref_mismatch = _run_install(tmp_path, source, tools, explicit_source=False)
    assert ref_mismatch.returncode == 1
    assert "HEAD does not match" in ref_mismatch.stderr

    private_ref.write_text("1" * 40, encoding="ascii")
    (source / ".fetched-commit").write_text("f" * 40, encoding="ascii")
    (source / ".non-fast-forward").write_text("blocked\n", encoding="utf-8")
    non_fast_forward = _run_install(
        tmp_path,
        source,
        tools,
        explicit_source=False,
        installer_args=("--fetch",),
    )
    assert non_fast_forward.returncode == 1
    assert "not a fast-forward" in non_fast_forward.stderr


def test_managed_source_cannot_relabel_a_previous_verified_ref(tmp_path: Path) -> None:
    source = _source(tmp_path / "install root" / "source")
    tools, _ = _fake_tools(tmp_path)
    installed = _run_install(tmp_path, source, tools, explicit_source=False)
    assert installed.returncode == 0, installed.stderr

    changed = _run_install(
        tmp_path,
        source,
        tools,
        explicit_source=False,
        extra_env={"OPENTULPA_INSTALL_REF": "release"},
    )

    assert changed.returncode == 1
    assert "unverified" in changed.stderr


def test_tui_bytes_are_material_to_controller_generation_identity(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    tui = tmp_path / "opentulpa-tui"
    _write_executable(tui, "#!/bin/sh\n[ \"$1\" = --protocol-version ] && printf '2\\n'\n")
    environment = {"OPENTULPA_TUI_BINARY": str(tui)}
    first = _run_install(tmp_path, source, tools, extra_env=environment)
    assert first.returncode == 0, first.stderr
    first_generation = _current_generation(tmp_path)

    _write_executable(
        tui,
        "#!/bin/sh\n# changed bytes\n[ \"$1\" = --protocol-version ] && printf '2\\n'\n",
    )
    second = _run_install(tmp_path, source, tools, extra_env=environment)

    assert second.returncode == 0, second.stderr
    assert _current_generation(tmp_path) != first_generation


def test_installer_refuses_command_file_inserted_during_build(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    tools, _ = _fake_tools(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _run_install,
            tmp_path,
            source,
            tools,
            extra_env={"FAKE_UV_DELAY": "1"},
        )
        command = tmp_path / "command bin" / "opentulpa"
        for _ in range(100):
            if (tmp_path / "install root" / "controller" / "install.lock").exists():
                break
            time.sleep(0.01)
        command.write_text("do not replace\n", encoding="utf-8")
        result = future.result(timeout=30)

    assert result.returncode == 1
    assert "refusing to replace existing regular file" in result.stderr
    assert command.read_text(encoding="utf-8") == "do not replace\n"


def test_install_script_has_posix_shell_syntax() -> None:
    result = subprocess.run(
        ["sh", "-n", str(REPO_ROOT / "install.sh")], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
