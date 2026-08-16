#!/bin/sh
set -eu

REPOSITORY="${OPENTULPA_INSTALL_REPOSITORY:-https://github.com/kvyb/opentulpa.git}"
REF="${OPENTULPA_INSTALL_REF:-main}"
PYTHON_REQUEST="${OPENTULPA_INSTALL_PYTHON:-3.12}"
PIP_VERSION="${OPENTULPA_INSTALL_PIP_VERSION:-25.1.1}"
PROFILE="${OPENTULPA_INSTALL_PROFILE:-controller-runtime-no-dev}"
DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
INSTALL_ROOT="${OPENTULPA_INSTALL_ROOT:-${DATA_HOME}/opentulpa/install}"
CONTROLLER_ROOT="${INSTALL_ROOT}/controller"
GENERATIONS_ROOT="${CONTROLLER_ROOT}/generations"
BIN_ROOT="${INSTALL_ROOT}/bin"
BIN_DIR="${OPENTULPA_BIN_DIR:-${HOME}/.local/bin}"
COMMANDS="opentulpa opentulpa-host opentulpa-sandbox-worker opentulpa-migrate-deepagents"

SOURCE_ARGUMENT=""
FETCH=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      [ "$#" -ge 2 ] || {
        printf '%s\n' "--source requires a path" >&2
        exit 2
      }
      SOURCE_ARGUMENT=$2
      shift 2
      ;;
    --fetch)
      FETCH=1
      shift
      ;;
    --dev-allow-dirty)
      OPENTULPA_ALLOW_DIRTY_SOURCE=1
      export OPENTULPA_ALLOW_DIRTY_SOURCE
      shift
      ;;
    -h|--help)
      printf '%s\n' "Usage: install.sh [--source PATH] [--fetch]"
      exit 0
      ;;
    *)
      printf '%s\n' "Unknown installer option: $1" >&2
      exit 2
      ;;
  esac
done

say() {
  printf '%s\n' "[opentulpa] $*"
}

fail() {
  printf '%s\n' "[opentulpa] error: $*" >&2
  exit 1
}

SCRIPT_SOURCE=""
if [ -f "$0" ]; then
  candidate_source=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
  if [ -f "${candidate_source}/pyproject.toml" ] \
    && [ -f "${candidate_source}/uv.lock" ] \
    && [ -e "${candidate_source}/.git" ]; then
    SCRIPT_SOURCE=$candidate_source
  fi
fi

EXPLICIT_SOURCE=0
if [ -n "$SOURCE_ARGUMENT" ]; then
  SOURCE_ROOT=$SOURCE_ARGUMENT
  EXPLICIT_SOURCE=1
elif [ -n "${OPENTULPA_INSTALL_SOURCE:-}" ]; then
  SOURCE_ROOT=$OPENTULPA_INSTALL_SOURCE
  EXPLICIT_SOURCE=1
elif [ -n "$SCRIPT_SOURCE" ]; then
  SOURCE_ROOT=$SCRIPT_SOURCE
  EXPLICIT_SOURCE=1
else
  SOURCE_ROOT="${INSTALL_ROOT}/source"
fi

case "$SOURCE_ROOT" in
  /*) ;;
  *) SOURCE_ROOT="$(pwd -P)/${SOURCE_ROOT}" ;;
esac

umask 077
mkdir -p "$CONTROLLER_ROOT" "$GENERATIONS_ROOT" "$BIN_ROOT" "$BIN_DIR"
chmod 700 "$INSTALL_ROOT" "$CONTROLLER_ROOT" "$GENERATIONS_ROOT" "$BIN_ROOT"

command -v git >/dev/null 2>&1 || fail "git is required"

if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
else
  command -v curl >/dev/null 2>&1 \
    || fail "uv is required; install it from https://astral.sh/uv"
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="${HOME}/.local/bin/uv"
  [ -x "$UV_BIN" ] || fail "uv installation did not create ${UV_BIN}"
fi

PYTHON_BIN=$("$UV_BIN" python find "$PYTHON_REQUEST")
[ -x "$PYTHON_BIN" ] || fail "uv did not resolve Python ${PYTHON_REQUEST}"

BUILD_ROOT=""
RESERVED_GENERATION=""
INSTALL_LOCK="${CONTROLLER_ROOT}/install.lock"
LOCK_HELD=0
cleanup() {
  status=$?
  if [ -n "$RESERVED_GENERATION" ]; then
    chmod -R u+w "$RESERVED_GENERATION" 2>/dev/null || true
    rm -rf "$RESERVED_GENERATION"
  fi
  if [ -n "$BUILD_ROOT" ] && [ -d "$BUILD_ROOT" ]; then
    chmod -R u+w "$BUILD_ROOT" 2>/dev/null || true
    rm -rf "$BUILD_ROOT"
  fi
  if [ "$LOCK_HELD" -eq 1 ]; then
    rmdir "$INSTALL_LOCK" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT HUP INT TERM

LOCK_WAIT_SECONDS="${OPENTULPA_INSTALL_LOCK_WAIT_SECONDS:-120}"
case "$LOCK_WAIT_SECONDS" in
  ''|*[!0-9]*) fail "OPENTULPA_INSTALL_LOCK_WAIT_SECONDS must be an integer" ;;
esac
[ "$LOCK_WAIT_SECONDS" -le 3600 ] \
  || fail "OPENTULPA_INSTALL_LOCK_WAIT_SECONDS cannot exceed 3600"
waited=0
while :; do
  if mkdir "$INSTALL_LOCK" 2>/dev/null; then
    chmod 700 "$INSTALL_LOCK"
    LOCK_HELD=1
    break
  fi
  lock_state=$("$PYTHON_BIN" - "$INSTALL_LOCK" <<'PY'
import os
import pathlib
import stat
import sys

lock = pathlib.Path(sys.argv[1])
try:
    metadata = lock.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or any(lock.iterdir())
    ):
        print("unsafe")
    else:
        print("held")
except FileNotFoundError:
    print("missing")
except OSError:
    print("unsafe")
PY
  )
  if [ "$lock_state" = "missing" ]; then
    continue
  fi
  [ "$lock_state" = "held" ] || fail "controller install lock is unsafe"
  [ "$waited" -lt "$LOCK_WAIT_SECONDS" ] \
    || fail "timed out waiting for another OpenTulpa install"
  sleep 1
  waited=$((waited + 1))
done

for command_name in $COMMANDS; do
  target_command="${BIN_DIR}/${command_name}"
  if [ -e "$target_command" ] && [ ! -L "$target_command" ]; then
    fail "refusing to replace existing regular file: ${target_command}"
  fi
done

SOURCE_KIND="local"
ACTUAL_REMOTE=""
VERIFIED_REF=""
VERIFIED_OID=""
REF_KEY=$(REF="$REF" "$PYTHON_BIN" - <<'PY'
import hashlib
import os

print(hashlib.sha256(os.environ["REF"].encode()).hexdigest())
PY
)
PRIVATE_REF="refs/opentulpa/install/verified/${REF_KEY}"
if [ "$EXPLICIT_SOURCE" -eq 0 ]; then
  SOURCE_KIND="managed"
  if [ -e "${SOURCE_ROOT}/.git" ]; then
    say "using managed source seed at ${SOURCE_ROOT}"
  else
    [ ! -e "$SOURCE_ROOT" ] \
      || fail "managed source path exists but is not a Git checkout: ${SOURCE_ROOT}"
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    say "cloning the trusted source seed"
    git clone --branch "$REF" --single-branch "$REPOSITORY" "$SOURCE_ROOT"
    git -C "$SOURCE_ROOT" update-ref "$PRIVATE_REF" HEAD
  fi
else
  say "using explicit source checkout at ${SOURCE_ROOT}"
fi

[ -e "${SOURCE_ROOT}/.git" ] || fail "source is not a Git checkout: ${SOURCE_ROOT}"
[ -f "${SOURCE_ROOT}/pyproject.toml" ] \
  || fail "source has no pyproject.toml: ${SOURCE_ROOT}"
[ -f "${SOURCE_ROOT}/uv.lock" ] || fail "source has no uv.lock: ${SOURCE_ROOT}"
[ -f "${SOURCE_ROOT}/install.sh" ] || fail "source has no install.sh: ${SOURCE_ROOT}"

dirty=$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=normal)
if [ -n "$dirty" ] && [ "${OPENTULPA_ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  fail "source checkout is dirty; commit/stash changes or use --dev-allow-dirty for development only"
fi
if [ "$FETCH" -eq 1 ] && [ "$EXPLICIT_SOURCE" -eq 1 ]; then
  fail "--fetch is only valid for the installer-managed source seed"
fi
if [ "$EXPLICIT_SOURCE" -eq 0 ]; then
  ACTUAL_REMOTE=$(git -C "$SOURCE_ROOT" remote get-url origin)
  [ "$ACTUAL_REMOTE" = "$REPOSITORY" ] \
    || fail "managed source origin does not match OPENTULPA_INSTALL_REPOSITORY"
fi
if [ "$FETCH" -eq 1 ]; then
  [ -z "$dirty" ] \
    || fail "refusing to fetch into a dirty managed source checkout"
  say "fetching configured source ref ${REF}"
  SOURCE_BEFORE_FETCH=$(git -C "$SOURCE_ROOT" rev-parse --verify HEAD)
  git -C "$SOURCE_ROOT" fetch --no-tags origin "${REF}:${PRIVATE_REF}"
  VERIFIED_OID=$(git -C "$SOURCE_ROOT" rev-parse --verify "${PRIVATE_REF}^{commit}")
  git -C "$SOURCE_ROOT" merge-base --is-ancestor "$SOURCE_BEFORE_FETCH" "$VERIFIED_OID" \
    || fail "configured source ref is not a fast-forward of managed HEAD"
  git -C "$SOURCE_ROOT" merge --ff-only "$VERIFIED_OID"
elif [ "$EXPLICIT_SOURCE" -eq 0 ]; then
  VERIFIED_OID=$(git -C "$SOURCE_ROOT" rev-parse --verify "${PRIVATE_REF}^{commit}") \
    || fail "managed source ref is unverified; rerun with --fetch"
  MANAGED_HEAD=$(git -C "$SOURCE_ROOT" rev-parse --verify HEAD)
  [ "$MANAGED_HEAD" = "$VERIFIED_OID" ] \
    || fail "managed source HEAD does not match the last verified configured ref"
fi

SOURCE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse --verify HEAD)
case "$SOURCE_COMMIT" in
  *[!0-9a-fA-F]*|'') fail "Git returned an invalid source commit" ;;
esac
case "${#SOURCE_COMMIT}" in
  40|64) ;;
  *) fail "Git returned an invalid source commit length" ;;
esac
if [ "$EXPLICIT_SOURCE" -eq 0 ]; then
  [ "$SOURCE_COMMIT" = "$VERIFIED_OID" ] \
    || fail "managed source commit changed after configured ref verification"
  VERIFIED_REF=$REF
else
  VERIFIED_OID=$SOURCE_COMMIT
fi

BUILD_ROOT=$(mktemp -d "${CONTROLLER_ROOT}/.build.XXXXXX")
command -v tar >/dev/null 2>&1 || fail "tar is required to install exact source"
EXACT_SOURCE="${BUILD_ROOT}/exact-source"
mkdir "$EXACT_SOURCE"
git -C "$SOURCE_ROOT" archive "$SOURCE_COMMIT" | tar -x -C "$EXACT_SOURCE"
[ -f "${EXACT_SOURCE}/pyproject.toml" ] && [ -f "${EXACT_SOURCE}/uv.lock" ] \
  && [ -f "${EXACT_SOURCE}/install.sh" ] \
  || fail "verified source archive is incomplete"

hash_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
with pathlib.Path(sys.argv[1]).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
}

python_identity() {
  "$PYTHON_BIN" - <<'PY'
import json
import platform
import sys
import sysconfig

print(json.dumps({
    "implementation": platform.python_implementation(),
    "platform": sysconfig.get_platform(),
    "python": platform.python_version(),
    "soabi": sysconfig.get_config_var("SOABI") or "",
}, sort_keys=True, separators=(",", ":")))
PY
}

atomic_replace() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import os
import sys

os.replace(sys.argv[1], sys.argv[2])
PY
}

runtime_tree_digest() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
paths = []
for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
    directory_names.sort()
    file_names.sort()
    paths.extend(pathlib.Path(directory) / name for name in (*directory_names, *file_names))
for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    if relative in {"manifest.json", "COMPLETE"}:
        continue
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISDIR(metadata.st_mode):
        kind = b"D"
        payload = b""
    elif stat.S_ISREG(metadata.st_mode):
        if metadata.st_nlink != 1:
            raise SystemExit(f"runtime tree contains a hard-linked file: {relative}")
        kind = b"F"
        payload = path.read_bytes()
    elif stat.S_ISLNK(metadata.st_mode):
        kind = b"L"
        payload = os.readlink(path).encode("utf-8")
    else:
        raise SystemExit(f"runtime tree contains a special file: {relative}")
    digest.update(kind)
    digest.update(b"\0")
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{mode:o}".encode("ascii"))
    digest.update(b"\0")
    digest.update(str(len(payload)).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

source_tree_digest() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
paths = []
for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
    directory_names.sort()
    file_names.sort()
    paths.extend(pathlib.Path(directory) / name for name in (*directory_names, *file_names))
for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        kind, payload, mode = b"D", b"", 0o755
    elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
        kind = b"F"
        payload = path.read_bytes()
        mode = 0o755 if metadata.st_mode & 0o111 else 0o644
    else:
        raise SystemExit(f"source seed contains a link, hard link, or special file: {relative}")
    digest.update(kind + b"\0")
    digest.update(relative.encode("utf-8") + b"\0")
    digest.update(f"{mode:o}".encode("ascii") + b"\0")
    digest.update(str(len(payload)).encode("ascii") + b"\0")
    digest.update(payload + b"\0")
print(digest.hexdigest())
PY
}

SOURCE_TREE_OID=$(git -C "$SOURCE_ROOT" rev-parse --verify "${SOURCE_COMMIT}^{tree}")
case "$SOURCE_TREE_OID" in
  *[!0-9a-fA-F]*|'') fail "Git returned an invalid source tree identity" ;;
esac
case "${#SOURCE_TREE_OID}" in
  40|64) ;;
  *) fail "Git returned an invalid source tree identity length" ;;
esac
SOURCE_SEED_SHA256=$(source_tree_digest "$EXACT_SOURCE")
UV_SHA256=$(hash_file "$UV_BIN")
BOOTSTRAP_PYTHON=$(
  "$PYTHON_BIN" - "$PYTHON_BIN" <<'PY'
import pathlib
import sys

print(pathlib.Path(sys.argv[1]).resolve(strict=True))
PY
)
case "$BOOTSTRAP_PYTHON" in
  *'
'*) fail "bootstrap Python path contains a newline" ;;
esac
BOOTSTRAP_PYTHON_SHA256=$(hash_file "$BOOTSTRAP_PYTHON")
if [ "$(id -u)" -eq 0 ]; then
  "$PYTHON_BIN" - "$BOOTSTRAP_PYTHON" "$UV_BIN" <<'PY'
import pathlib
import stat
import sys

for raw in sys.argv[1:]:
    path = pathlib.Path(raw).resolve(strict=True)
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise SystemExit("root installation requires root-owned immutable bootstrap tools")
PY
fi

WHEEL_DIR="${BUILD_ROOT}/project-wheel"
REQUIREMENTS="${BUILD_ROOT}/requirements.txt"
DOWNLOADED_WHEELS="${BUILD_ROOT}/wheelhouse"
mkdir -p "$WHEEL_DIR" "$DOWNLOADED_WHEELS"

say "building the controller wheel from commit ${SOURCE_COMMIT}"
"$UV_BIN" build --wheel --out-dir "$WHEEL_DIR" "$EXACT_SOURCE"
set -- "$WHEEL_DIR"/*.whl
[ "$#" -eq 1 ] && [ -f "$1" ] \
  || fail "controller build did not produce exactly one wheel"
PROJECT_WHEEL=$1

"$UV_BIN" export \
  --frozen \
  --no-dev \
  --no-emit-project \
  --no-header \
  --project "$EXACT_SOURCE" \
  --output-file "$REQUIREMENTS"
[ -s "$REQUIREMENTS" ] || fail "uv export produced no locked runtime requirements"

if [ -n "${OPENTULPA_PIP_BIN:-}" ]; then
  PIP_BIN=$OPENTULPA_PIP_BIN
  [ -x "$PIP_BIN" ] || fail "OPENTULPA_PIP_BIN is not executable: ${PIP_BIN}"
else
  PIP_ENV="${BUILD_ROOT}/pip"
  "$UV_BIN" venv --python "$PYTHON_BIN" "$PIP_ENV"
  "$UV_BIN" pip install --python "${PIP_ENV}/bin/python" "pip==${PIP_VERSION}"
  PIP_BIN="${PIP_ENV}/bin/pip"
fi

say "building the binary-only trusted wheelhouse"
if ! "$PIP_BIN" download \
  --disable-pip-version-check \
  --require-hashes \
  --only-binary=:all: \
  --dest "$DOWNLOADED_WHEELS" \
  --requirement "$REQUIREMENTS"; then
  fail "a locked dependency has no binary wheel; a trusted runtime base rebuild is required"
fi

SOURCE_AFTER_BUILD=$(git -C "$SOURCE_ROOT" rev-parse --verify HEAD)
dirty_after_build=$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=normal)
[ "$SOURCE_AFTER_BUILD" = "$SOURCE_COMMIT" ] \
  || fail "source commit changed while the controller was being built"
if [ -n "$dirty_after_build" ] && [ "${OPENTULPA_ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  fail "source checkout changed while the controller was being built"
fi

LOCK_SHA256=$(hash_file "${EXACT_SOURCE}/uv.lock")
WHEEL_SHA256=$(hash_file "$PROJECT_WHEEL")
REQUIREMENTS_SHA256=$(hash_file "$REQUIREMENTS")
PYTHON_IDENTITY=$(python_identity)
WHEEL_NAME=$(basename "$PROJECT_WHEEL")
WHEELHOUSE_JSON=$(
  "$PYTHON_BIN" - "$DOWNLOADED_WHEELS" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = []
for path in sorted(root.glob("*.whl"), key=lambda item: item.name):
    files.append({
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    })
if not files:
    raise SystemExit("locked wheelhouse is empty")
print(json.dumps(files, sort_keys=True, separators=(",", ":")))
PY
)
WHEELHOUSE_SHA256=$(
  WHEELHOUSE_JSON="$WHEELHOUSE_JSON" "$PYTHON_BIN" - <<'PY'
import hashlib
import os

print(hashlib.sha256(os.environ["WHEELHOUSE_JSON"].encode()).hexdigest())
PY
)

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) TUI_TARGET=darwin-arm64 ;;
  Darwin-x86_64) TUI_TARGET=darwin-x64 ;;
  Linux-aarch64|Linux-arm64) TUI_TARGET=linux-arm64 ;;
  Linux-x86_64) TUI_TARGET=linux-x64 ;;
  *) TUI_TARGET="" ;;
esac
TUI_BINARY="${OPENTULPA_TUI_BINARY:-}"
if [ -z "$TUI_BINARY" ] && [ -n "$TUI_TARGET" ] \
  && [ -f "${EXACT_SOURCE}/clients/tui/package.json" ]; then
  BUN_VERSION=1.3.14
  if command -v bun >/dev/null 2>&1 \
    && [ "$(bun --version 2>/dev/null || true)" = "$BUN_VERSION" ]; then
    BUN_BIN=$(command -v bun)
  else
    command -v curl >/dev/null 2>&1 \
      || fail "curl is required to install the pinned TUI build tool"
    BUN_HOME="${BUILD_ROOT}/bun"
    say "installing the pinned TUI build tool"
    BUN_INSTALL="$BUN_HOME" curl -fsSL https://bun.com/install \
      | BUN_INSTALL="$BUN_HOME" bash -s "bun-v${BUN_VERSION}"
    BUN_BIN="${BUN_HOME}/bin/bun"
  fi
  [ -x "$BUN_BIN" ] || fail "pinned Bun installation failed"
  TUI_SNAPSHOT="${BUILD_ROOT}/tui-source"
  mkdir -p "${TUI_SNAPSHOT}/clients"
  cp -R "${EXACT_SOURCE}/clients/tui" "${TUI_SNAPSHOT}/clients/tui"
  TUI_BUILD_ROOT="${TUI_SNAPSHOT}/clients/tui"
  [ -f "${TUI_BUILD_ROOT}/package.json" ] \
    || fail "verified terminal client source is unavailable"
  say "building the native terminal client"
  (
    cd "$TUI_BUILD_ROOT"
    "$BUN_BIN" install --frozen-lockfile
    "$BUN_BIN" run build
  )
  TUI_BINARY="${TUI_BUILD_ROOT}/dist/opentulpa-tui-${TUI_TARGET}"
fi
TUI_NAME=""
TUI_SHA256=""
if [ -n "$TUI_BINARY" ]; then
  [ -x "$TUI_BINARY" ] || fail "terminal client is not executable: ${TUI_BINARY}"
  [ "$("$TUI_BINARY" --protocol-version)" = "2" ] \
    || fail "terminal client protocol is incompatible"
  TUI_NAME=$(basename "$TUI_BINARY")
  TUI_SHA256=$(hash_file "$TUI_BINARY")
fi

GENERATION_ID=$(
  SOURCE_COMMIT="$SOURCE_COMMIT" \
  SOURCE_TREE_OID="$SOURCE_TREE_OID" \
  SOURCE_SEED_SHA256="$SOURCE_SEED_SHA256" \
  LOCK_SHA256="$LOCK_SHA256" \
  WHEEL_SHA256="$WHEEL_SHA256" \
  WHEEL_NAME="$WHEEL_NAME" \
  REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
  WHEELHOUSE_SHA256="$WHEELHOUSE_SHA256" \
  UV_SHA256="$UV_SHA256" \
  TUI_NAME="$TUI_NAME" \
  TUI_SHA256="$TUI_SHA256" \
  BOOTSTRAP_PYTHON="$BOOTSTRAP_PYTHON" \
  BOOTSTRAP_PYTHON_SHA256="$BOOTSTRAP_PYTHON_SHA256" \
  PYTHON_IDENTITY="$PYTHON_IDENTITY" \
  PROFILE="$PROFILE" \
  "$PYTHON_BIN" "${EXACT_SOURCE}/controller_generation.py" generation-id
)
GENERATION="${GENERATIONS_ROOT}/${GENERATION_ID}"

generation_valid() {
  "$PYTHON_BIN" "${EXACT_SOURCE}/controller_generation.py" verify-manifest \
    "$1/manifest.json" "$GENERATION_ID" || return 1
  "$PYTHON_BIN" - "$1" "$GENERATION_ID" $COMMANDS <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
generation_id = sys.argv[2]
commands = sys.argv[3:]
complete = root / "COMPLETE"
manifest_path = root / "manifest.json"
if not complete.is_file() or complete.stat().st_size != 0 or not manifest_path.is_file():
    raise SystemExit(1)
raw = manifest_path.read_bytes()
manifest = json.loads(raw)
def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
if sha256(root / "requirements.txt") != manifest["identity"]["requirements_sha256"]:
    raise SystemExit(1)
wheel = root / "wheels" / manifest["identity"]["wheel_name"]
if sha256(wheel) != manifest["identity"]["wheel_sha256"]:
    raise SystemExit(1)
expected_wheelhouse = manifest.get("wheelhouse")
actual_wheelhouse = []
for path in sorted((root / "wheelhouse").glob("*.whl"), key=lambda item: item.name):
    actual_wheelhouse.append({
        "name": path.name,
        "sha256": sha256(path),
        "size": path.stat().st_size,
    })
if actual_wheelhouse != expected_wheelhouse:
    raise SystemExit(1)
encoded_wheelhouse = json.dumps(
    actual_wheelhouse, sort_keys=True, separators=(",", ":")
).encode()
if hashlib.sha256(encoded_wheelhouse).hexdigest() != manifest["identity"]["wheelhouse_sha256"]:
    raise SystemExit(1)
if (root / "source-seed").is_symlink() or not (root / "source-seed").is_dir():
    raise SystemExit(1)
for asset in ("bridge.mjs", "package.json", "package-lock.json"):
    if not (root / "assets" / "railway_sandbox_bridge" / asset).is_file():
        raise SystemExit(1)
uv = root / "assets" / "toolchain" / "uv"
if not uv.is_file() or not uv.stat().st_mode & stat.S_IXUSR:
    raise SystemExit(1)
runtime_digest = hashlib.sha256()
runtime_paths = []
for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
    directory_names.sort()
    file_names.sort()
    runtime_paths.extend(pathlib.Path(directory) / name for name in (*directory_names, *file_names))
for path in sorted(runtime_paths, key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    if relative in {"manifest.json", "COMPLETE"}:
        continue
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISDIR(metadata.st_mode):
        kind, payload = b"D", b""
    elif stat.S_ISREG(metadata.st_mode):
        if metadata.st_nlink != 1:
            raise SystemExit(1)
        kind, payload = b"F", path.read_bytes()
    elif stat.S_ISLNK(metadata.st_mode):
        kind, payload = b"L", os.readlink(path).encode("utf-8")
    else:
        raise SystemExit(1)
    runtime_digest.update(kind)
    runtime_digest.update(b"\0")
    runtime_digest.update(relative.encode("utf-8"))
    runtime_digest.update(b"\0")
    runtime_digest.update(f"{mode:o}".encode("ascii"))
    runtime_digest.update(b"\0")
    runtime_digest.update(str(len(payload)).encode("ascii"))
    runtime_digest.update(b"\0")
    runtime_digest.update(payload)
    runtime_digest.update(b"\0")
if runtime_digest.hexdigest() != manifest.get("runtime_tree_sha256"):
    raise SystemExit(1)
expected = {f"#!{root / 'bin' / name}" for name in ("python", "python3")}
for command in commands:
    entrypoint = root / "bin" / command
    if not entrypoint.is_file() or entrypoint.read_text(errors="replace").splitlines()[0] not in expected:
        raise SystemExit(1)
PY
}

REUSED=0
if [ -e "$GENERATION" ]; then
  if generation_valid "$GENERATION"; then
    REUSED=1
    say "reusing verified controller generation ${GENERATION_ID}"
  else
    say "removing incomplete controller generation ${GENERATION_ID}"
    chmod -R u+w "$GENERATION" 2>/dev/null || true
    rm -rf "$GENERATION"
  fi
fi

if [ "$REUSED" -eq 0 ]; then
  mkdir "$GENERATION"
  RESERVED_GENERATION=$GENERATION
  say "creating final-path controller environment ${GENERATION}"
  "$UV_BIN" venv --python "$PYTHON_BIN" "$GENERATION"
  mkdir -p "${GENERATION}/wheelhouse" "${GENERATION}/wheels" \
    "${GENERATION}/assets/railway_sandbox_bridge" "${GENERATION}/assets/toolchain"
  cp "$DOWNLOADED_WHEELS"/*.whl "${GENERATION}/wheelhouse/"
  cp "$PROJECT_WHEEL" "${GENERATION}/wheels/${WHEEL_NAME}"
  cp "$REQUIREMENTS" "${GENERATION}/requirements.txt"
  cp "$UV_BIN" "${GENERATION}/assets/toolchain/uv"
  chmod 500 "${GENERATION}/assets/toolchain/uv"

  "$UV_BIN" pip install \
    --python "${GENERATION}/bin/python" \
    --link-mode=copy \
    --no-index \
    --find-links "${GENERATION}/wheelhouse" \
    --require-hashes \
    --requirement "${GENERATION}/requirements.txt"
  "$UV_BIN" pip install \
    --python "${GENERATION}/bin/python" \
    --link-mode=copy \
    --no-index \
    --no-deps \
    "${GENERATION}/wheels/${WHEEL_NAME}"

  for asset_name in bridge.mjs package.json package-lock.json; do
    [ -f "${EXACT_SOURCE}/railway_sandbox_bridge/${asset_name}" ] \
      || fail "required controller asset is missing: railway_sandbox_bridge/${asset_name}"
    cp "${EXACT_SOURCE}/railway_sandbox_bridge/${asset_name}" \
      "${GENERATION}/assets/railway_sandbox_bridge/${asset_name}"
  done
  if [ -f "${EXACT_SOURCE}/opentulpa.config.yaml" ]; then
    cp "${EXACT_SOURCE}/opentulpa.config.yaml" "${GENERATION}/assets/opentulpa.config.yaml"
  fi

  if [ -n "$TUI_BINARY" ]; then
    mkdir -p "${GENERATION}/assets/tui"
    cp "$TUI_BINARY" "${GENERATION}/assets/tui/${TUI_NAME}"
    chmod 500 "${GENERATION}/assets/tui/${TUI_NAME}"
  fi
  mkdir "${GENERATION}/source-seed"
  tar -c -C "$EXACT_SOURCE" . | tar -x -C "${GENERATION}/source-seed"
  "$PYTHON_BIN" - "${GENERATION}/source-seed" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
for directory, directory_names, file_names in os.walk(root, followlinks=False):
    for name in (*directory_names, *file_names):
        path = pathlib.Path(directory) / name
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not (
            stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
        ):
            raise SystemExit("trusted source seed contains a link or special file")
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
            raise SystemExit("trusted source seed contains a hard-linked file")
PY
  COPIED_SOURCE_SHA256=$(source_tree_digest "${GENERATION}/source-seed")
  [ "$COPIED_SOURCE_SHA256" = "$SOURCE_SEED_SHA256" ] \
    || fail "installed source seed differs from the verified Git archive"
  SOURCE_COMMIT="$SOURCE_COMMIT" \
  SOURCE_TREE_OID="$SOURCE_TREE_OID" \
  SOURCE_SEED_SHA256="$SOURCE_SEED_SHA256" \
  "$PYTHON_BIN" - "${GENERATION}/source-seed-manifest.json" <<'PY'
import json
import os
import pathlib
import sys

payload = {
    "format_version": 1,
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_seed_sha256": os.environ["SOURCE_SEED_SHA256"],
    "source_tree_oid": os.environ["SOURCE_TREE_OID"],
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

  "${GENERATION}/bin/python" - "$GENERATION" "$SOURCE_ROOT" $COMMANDS <<'PY'
import importlib.metadata
import importlib.resources
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).resolve()
commands = sys.argv[3:]
expected = {f"#!{root / 'bin' / name}" for name in ("python", "python3")}
distribution = importlib.metadata.distribution("opentulpa")
entrypoints = {item.name: item for item in distribution.entry_points if item.group == "console_scripts"}
for command in commands:
    script = root / "bin" / command
    if command not in entrypoints or not script.is_file():
        raise SystemExit(f"missing console entrypoint: {command}")
    if script.read_text(errors="replace").splitlines()[0] not in expected:
        raise SystemExit(f"entrypoint does not use its final interpreter: {command}")
    if not callable(entrypoints[command].load()):
        raise SystemExit(f"console entrypoint is not callable: {command}")
package = importlib.resources.files("opentulpa")
if not package.joinpath("resources", "release_contract.json").is_file():
    raise SystemExit("packaged release contract is missing")
for site in (root / "lib").glob("python*/site-packages"):
    for pth in site.glob("*.pth"):
        text = pth.read_text(errors="replace")
        if "editable" in text.casefold() or str(source) in text:
            raise SystemExit(f"editable source path found in {pth}")
PY

  chmod -R a-w "$GENERATION"
  chmod u+w "$GENERATION"
  RUNTIME_TREE_SHA256=$(runtime_tree_digest "$GENERATION")
  IDENTITY_JSON=$(
    SOURCE_COMMIT="$SOURCE_COMMIT" \
    SOURCE_TREE_OID="$SOURCE_TREE_OID" \
    SOURCE_SEED_SHA256="$SOURCE_SEED_SHA256" \
    LOCK_SHA256="$LOCK_SHA256" \
    WHEEL_SHA256="$WHEEL_SHA256" \
    WHEEL_NAME="$WHEEL_NAME" \
    REQUIREMENTS_SHA256="$REQUIREMENTS_SHA256" \
    WHEELHOUSE_SHA256="$WHEELHOUSE_SHA256" \
    UV_SHA256="$UV_SHA256" \
    TUI_NAME="$TUI_NAME" \
    TUI_SHA256="$TUI_SHA256" \
    BOOTSTRAP_PYTHON="$BOOTSTRAP_PYTHON" \
    BOOTSTRAP_PYTHON_SHA256="$BOOTSTRAP_PYTHON_SHA256" \
    PYTHON_IDENTITY="$PYTHON_IDENTITY" \
    PROFILE="$PROFILE" \
  "$PYTHON_BIN" "${EXACT_SOURCE}/controller_generation.py" identity
  )
  GENERATION_ID="$GENERATION_ID" \
  IDENTITY_JSON="$IDENTITY_JSON" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  REPOSITORY="$REPOSITORY" \
  REF="$REF" \
  EXPLICIT_SOURCE="$EXPLICIT_SOURCE" \
  SOURCE_KIND="$SOURCE_KIND" \
  ACTUAL_REMOTE="$ACTUAL_REMOTE" \
  VERIFIED_REF="$VERIFIED_REF" \
  VERIFIED_OID="$VERIFIED_OID" \
  RUNTIME_TREE_SHA256="$RUNTIME_TREE_SHA256" \
  WHEELHOUSE_JSON="$WHEELHOUSE_JSON" \
  "$PYTHON_BIN" "${EXACT_SOURCE}/controller_generation.py" write-manifest \
    "${GENERATION}/manifest.json"
  chmod 400 "${GENERATION}/manifest.json"
  : > "${GENERATION}/COMPLETE"
  chmod 400 "${GENERATION}/COMPLETE"
  chmod a-w "$GENERATION"
  generation_valid "$GENERATION" || fail "completed generation failed validation"
  RESERVED_GENERATION=""
fi

write_dispatcher() {
  command_name=$1
  destination="${BIN_ROOT}/${command_name}"
  temporary="${BIN_ROOT}/.${command_name}.$$"
  bootstrap_python_shell=$(
    "$PYTHON_BIN" - "$BOOTSTRAP_PYTHON" <<'PY'
import shlex
import sys

print(shlex.quote(sys.argv[1]))
PY
  )
  {
    printf '%s\n' '#!/bin/sh'
    printf '%s\n' "bootstrap_python=${bootstrap_python_shell}"
    printf '%s\n' "bootstrap_python_sha256=${BOOTSTRAP_PYTHON_SHA256}"
    cat <<'DISPATCHER'
set -eu

# Root-owned deployments enforce this boundary against candidate code. Non-root
# installations provide tamper detection only because a same-UID process can race it.
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH

command_name=$(basename "$0")
self=$0
while [ -L "$self" ]; do
  link=$(readlink "$self")
  case "$link" in
    /*) self=$link ;;
    *) self=$(dirname "$self")/$link ;;
  esac
done
bin_root=$(CDPATH= cd "$(dirname "$self")" && pwd -P)
install_root=$(CDPATH= cd "${bin_root}/.." && pwd -P)
current="${install_root}/controller/current"
[ -L "$current" ] || {
  printf '%s\n' "OpenTulpa has no active controller generation." >&2
  exit 1
}
target=$(readlink "$current")
case "$target" in
  generations/*) ;;
  *)
    printf '%s\n' "OpenTulpa controller pointer escapes the generation store." >&2
    exit 1
    ;;
esac
generation_id=${target#generations/}
case "$generation_id" in
  *[!0-9a-f]*|'')
    printf '%s\n' "OpenTulpa controller generation identity is invalid." >&2
    exit 1
    ;;
esac
[ "${#generation_id}" -eq 64 ] || {
  printf '%s\n' "OpenTulpa controller generation identity is invalid." >&2
  exit 1
}
generation="${install_root}/controller/${target}"
[ -d "${install_root}/controller" ] && [ ! -L "${install_root}/controller" ] \
  && [ -d "${install_root}/controller/generations" ] \
  && [ ! -L "${install_root}/controller/generations" ] \
  && [ -d "$generation" ] && [ ! -L "$generation" ] || {
  printf '%s\n' "OpenTulpa controller generation directory is invalid." >&2
  exit 1
}
entrypoint="${generation}/bin/${command_name}"
python="${generation}/bin/python"
[ -f "$bootstrap_python" ] && [ ! -L "$bootstrap_python" ] || {
  printf '%s\n' "OpenTulpa bootstrap verifier is invalid." >&2
  exit 1
}
if command -v sha256sum >/dev/null 2>&1; then
  actual_bootstrap_sha256=$(sha256sum "$bootstrap_python")
  actual_bootstrap_sha256=${actual_bootstrap_sha256%% *}
elif command -v shasum >/dev/null 2>&1; then
  actual_bootstrap_sha256=$(shasum -a 256 "$bootstrap_python")
  actual_bootstrap_sha256=${actual_bootstrap_sha256%% *}
else
  printf '%s\n' "OpenTulpa cannot verify its bootstrap interpreter." >&2
  exit 1
fi
[ "$actual_bootstrap_sha256" = "$bootstrap_python_sha256" ] || {
  printf '%s\n' "OpenTulpa bootstrap verifier failed integrity validation." >&2
  exit 1
}
for required_file in "$entrypoint" "${generation}/manifest.json" \
  "${generation}/COMPLETE" "${generation}/source-seed-manifest.json"; do
  [ -f "$required_file" ] && [ ! -L "$required_file" ] || {
    printf '%s\n' "OpenTulpa controller generation layout is invalid." >&2
    exit 1
  }
done
[ -x "$python" ] || {
  printf '%s\n' "OpenTulpa controller interpreter is invalid." >&2
  exit 1
}
[ -d "${generation}/source-seed" ] && [ ! -L "${generation}/source-seed" ] || {
  printf '%s\n' "OpenTulpa source seed is invalid." >&2
  exit 1
}
if ! source_identity=$(env -i HOME=/nonexistent PATH="$PATH" PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 "$bootstrap_python" -I -S - \
  "$generation" "$generation_id" "$entrypoint" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
generation_id = sys.argv[2]
entrypoint = pathlib.Path(sys.argv[3])
manifest_path = root / "manifest.json"
complete = root / "COMPLETE"
source_manifest_path = root / "source-seed-manifest.json"
for path, empty in (
    (manifest_path, False),
    (complete, True),
    (entrypoint, False),
    (source_manifest_path, False),
):
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or (empty and metadata.st_size != 0)
    ):
        raise SystemExit(1)
    if path in {manifest_path, complete} and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise SystemExit(1)
raw = manifest_path.read_bytes()
manifest = json.loads(raw)
canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
digest = str(manifest.get("runtime_tree_sha256") or "")
identity = manifest.get("identity")
if (
    raw != canonical
    or manifest.get("format_version") != 1
    or manifest.get("generation_id") != generation_id
    or not isinstance(identity, dict)
    or hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() != generation_id
    or len(digest) != 64
    or any(character not in "0123456789abcdef" for character in digest)
):
    raise SystemExit(1)
runtime_digest = hashlib.sha256()
runtime_paths = []
for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
    directory_names.sort()
    file_names.sort()
    runtime_paths.extend(pathlib.Path(directory) / name for name in (*directory_names, *file_names))
for path in sorted(runtime_paths, key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    if relative in {"manifest.json", "COMPLETE"}:
        continue
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISDIR(metadata.st_mode):
        kind, payload = b"D", b""
    elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
        kind, payload = b"F", path.read_bytes()
    elif stat.S_ISLNK(metadata.st_mode):
        kind, payload = b"L", os.readlink(path).encode("utf-8")
    else:
        raise SystemExit(1)
    runtime_digest.update(kind + b"\0")
    runtime_digest.update(relative.encode("utf-8") + b"\0")
    runtime_digest.update(f"{mode:o}".encode("ascii") + b"\0")
    runtime_digest.update(str(len(payload)).encode("ascii") + b"\0")
    runtime_digest.update(payload + b"\0")
if runtime_digest.hexdigest() != digest:
    raise SystemExit(1)
expected = {f"#!{root / 'bin' / name}" for name in ("python", "python3")}
if entrypoint.read_text(errors="replace").splitlines()[0] not in expected:
    raise SystemExit(1)
source = manifest.get("source")
oid = str(source.get("oid") if isinstance(source, dict) else "")
if len(oid) not in {40, 64} or any(character not in "0123456789abcdef" for character in oid):
    raise SystemExit(1)
source_raw = source_manifest_path.read_bytes()
source_manifest = json.loads(source_raw)
source_canonical = (
    json.dumps(source_manifest, sort_keys=True, separators=(",", ":")) + "\n"
).encode()
source_sha256 = str(source_manifest.get("source_seed_sha256") or "")
tree_oid = str(source_manifest.get("source_tree_oid") or "")
if (
    source_raw != source_canonical
    or source_manifest.get("format_version") != 1
    or source_manifest.get("source_commit") != oid
    or identity.get("source_commit") != oid
    or identity.get("source_seed_sha256") != source_sha256
    or identity.get("source_tree_oid") != tree_oid
    or len(source_sha256) != 64
    or any(character not in "0123456789abcdef" for character in source_sha256)
    or len(tree_oid) not in {40, 64}
    or any(character not in "0123456789abcdef" for character in tree_oid)
):
    raise SystemExit(1)
source_digest = hashlib.sha256()
source_root = root / "source-seed"
source_paths = []
for directory, directory_names, file_names in os.walk(
    source_root, topdown=True, followlinks=False
):
    directory_names.sort()
    file_names.sort()
    source_paths.extend(pathlib.Path(directory) / name for name in (*directory_names, *file_names))
for path in sorted(source_paths, key=lambda item: item.relative_to(source_root).as_posix()):
    relative = path.relative_to(source_root).as_posix()
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        kind, payload, mode = b"D", b"", 0o755
    elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
        kind = b"F"
        payload = path.read_bytes()
        mode = 0o755 if metadata.st_mode & 0o111 else 0o644
    else:
        raise SystemExit(1)
    source_digest.update(kind + b"\0")
    source_digest.update(relative.encode("utf-8") + b"\0")
    source_digest.update(f"{mode:o}".encode("ascii") + b"\0")
    source_digest.update(str(len(payload)).encode("ascii") + b"\0")
    source_digest.update(payload + b"\0")
if source_digest.hexdigest() != source_sha256:
    raise SystemExit(1)
print(f"{oid}:{source_sha256}:{tree_oid}")
PY
); then
  printf '%s\n' "OpenTulpa controller generation failed launch validation." >&2
  exit 1
fi
source_oid=${source_identity%%:*}
source_remainder=${source_identity#*:}
source_seed_sha256=${source_remainder%%:*}
source_tree_oid=${source_remainder#*:}
source_seed=$(CDPATH= cd "${generation}/source-seed" && pwd -P) || {
  printf '%s\n' "OpenTulpa source seed is unavailable." >&2
  exit 1
}
export OPENTULPA_INSTALL_ROOT="$install_root"
export OPENTULPA_SOURCE_SEED_ROOT="$source_seed"
export OPENTULPA_SOURCE_SEED_OID="$source_oid"
export OPENTULPA_SOURCE_SEED_SHA256="$source_seed_sha256"
export OPENTULPA_SOURCE_SEED_TREE_OID="$source_tree_oid"
export OPENTULPA_TRUSTED_WHEELHOUSE="${generation}/wheelhouse"
export OPENTULPA_INSTALL_ASSETS_ROOT="${generation}/assets"
export OPENTULPA_UV_BIN="${generation}/assets/toolchain/uv"
export OPENTULPA_CONTROLLER_GENERATION_ID="$generation_id"
export OPENTULPA_CONTROLLER_HOST_EXECUTABLE="${generation}/bin/opentulpa-host"
IFS= read -r first_line < "$entrypoint"
if [ "$first_line" = "#!${generation}/bin/python" ]; then
  exec "${generation}/bin/python" "$entrypoint" "$@"
fi
exec "$entrypoint" "$@"
DISPATCHER
  } > "$temporary"
  chmod 500 "$temporary"
  atomic_replace "$temporary" "$destination"
}

for command_name in $COMMANDS; do
  write_dispatcher "$command_name"
done

for command_name in $COMMANDS; do
  target_command="${BIN_DIR}/${command_name}"
  public_temp="${BIN_DIR}/.${command_name}.$$"
  if [ -e "$target_command" ] && [ ! -L "$target_command" ]; then
    fail "refusing to replace existing regular file: ${target_command}"
  fi
  [ ! -e "$public_temp" ] && [ ! -L "$public_temp" ] \
    || fail "temporary public command path already exists: ${public_temp}"
  ln -s "${BIN_ROOT}/${command_name}" "$public_temp"
  if [ -e "$target_command" ] && [ ! -L "$target_command" ]; then
    rm -f "$public_temp"
    fail "refusing to replace existing regular file: ${target_command}"
  fi
  atomic_replace "$public_temp" "$target_command"
done

INSTALLER_TEMP="${CONTROLLER_ROOT}/.installer.$$"
cp "${EXACT_SOURCE}/install.sh" "$INSTALLER_TEMP"
chmod 500 "$INSTALLER_TEMP"
atomic_replace "$INSTALLER_TEMP" "${CONTROLLER_ROOT}/installer.sh"

INSTALL_METADATA_TEMP="${CONTROLLER_ROOT}/.install.json.$$"
SOURCE_ROOT="$SOURCE_ROOT" \
REPOSITORY="$REPOSITORY" \
REF="$REF" \
EXPLICIT_SOURCE="$EXPLICIT_SOURCE" \
SOURCE_KIND="$SOURCE_KIND" \
ACTUAL_REMOTE="$ACTUAL_REMOTE" \
VERIFIED_REF="$VERIFIED_REF" \
VERIFIED_OID="$VERIFIED_OID" \
GENERATION_ID="$GENERATION_ID" \
"$PYTHON_BIN" - "$INSTALL_METADATA_TEMP" <<'PY'
import json
import os
import pathlib
import sys

metadata = {
    "format_version": 1,
    "generation_id": os.environ["GENERATION_ID"],
    "managed_source": os.environ["EXPLICIT_SOURCE"] != "1",
    "source_kind": os.environ["SOURCE_KIND"],
    "source_oid": os.environ["VERIFIED_OID"],
    "actual_remote": os.environ["ACTUAL_REMOTE"] or None,
    "verified_ref": os.environ["VERIFIED_REF"] or None,
    "ref": os.environ["REF"],
    "repository": os.environ["REPOSITORY"],
    "source_root": os.environ["SOURCE_ROOT"],
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
chmod 600 "$INSTALL_METADATA_TEMP"
atomic_replace "$INSTALL_METADATA_TEMP" "${CONTROLLER_ROOT}/install.json"

CURRENT="${CONTROLLER_ROOT}/current"
PREVIOUS="${CONTROLLER_ROOT}/previous"
NEW_TARGET="generations/${GENERATION_ID}"
OLD_TARGET=""
if [ -L "$CURRENT" ]; then
  OLD_TARGET=$(readlink "$CURRENT")
elif [ -e "$CURRENT" ]; then
  fail "controller/current exists and is not a symbolic link"
fi
if [ -n "$OLD_TARGET" ]; then
  case "$OLD_TARGET" in
    generations/*) ;;
    *) fail "controller/current has an invalid generation target" ;;
  esac
  old_generation_id=${OLD_TARGET#generations/}
  case "$old_generation_id" in
    *[!0-9a-f]*|'') fail "controller/current has an invalid generation identity" ;;
  esac
  [ "${#old_generation_id}" -eq 64 ] \
    || fail "controller/current has an invalid generation identity"
fi

if [ "$OLD_TARGET" != "$NEW_TARGET" ]; then
  if [ -n "$OLD_TARGET" ]; then
    previous_temp="${CONTROLLER_ROOT}/.previous.$$"
    ln -s "$OLD_TARGET" "$previous_temp"
    atomic_replace "$previous_temp" "$PREVIOUS"
  fi
  current_temp="${CONTROLLER_ROOT}/.current.$$"
  ln -s "$NEW_TARGET" "$current_temp"
  atomic_replace "$current_temp" "$CURRENT"
  say "activated controller generation ${GENERATION_ID}"
else
  say "controller generation ${GENERATION_ID} is already active"
fi

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *) say "add ${BIN_DIR} to PATH" ;;
esac

printf '\nOpenTulpa installed. Start it with:\n\n  opentulpa\n\n'
trap - EXIT HUP INT TERM
cleanup
