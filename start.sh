#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

cd "${REPO_ROOT}"

MODE="up"
RUNTIME_MODE="${START_MODE:-auto}"
INSTALL_BROWSER_USE="${INSTALL_BROWSER_USE:-1}"
INSTALL_CLOUDFLARED="${INSTALL_CLOUDFLARED:-auto}"
DRY_RUN=0
PASSTHRU=()

load_dotenv() {
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
      local line key value
      line="${raw_line#"${raw_line%%[![:space:]]*}"}"
      [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
      key="${line%%=*}"
      value="${line#*=}"
      key="${key%"${key##*[![:space:]]}"}"
      if [[ -n "${!key+x}" ]]; then
        continue
      fi
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      export "${key}=${value}"
    done < "${REPO_ROOT}/.env"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./start.sh [up|install|run] [--app|--manager] [options] [-- extra-args]

Defaults:
  - mode: up      (install, then run)
  - runtime: auto (app when PUBLIC_BASE_URL/RAILWAY_PUBLIC_DOMAIN is set, otherwise manager)

Options:
  --app                 Force direct app mode
  --manager             Force quick-tunnel manager mode
  --browser-use         Install Browser Use Chromium
  --no-browser-use      Skip Browser Use Chromium install
  --cloudflared         Install cloudflared when manager mode needs it
  --no-cloudflared      Never install cloudflared automatically
  --install-only        Install/setup only
  --run-only            Run only
  --dry-run             Print commands without running them
  -h, --help            Show this help

.env knobs:
  START_MODE=auto|app|manager
  INSTALL_BROWSER_USE=1|0
  INSTALL_CLOUDFLARED=auto|1|0
EOF
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_falsey() {
  case "${1:-}" in
    0|false|FALSE|no|NO|off|OFF) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  echo "[start] $*"
}

die() {
  echo "[start] error: $*" >&2
  exit 1
}

run_cmd() {
  log "$*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      up|install|run)
        MODE="$1"
        shift
        ;;
      --install-only)
        MODE="install"
        shift
        ;;
      --run-only)
        MODE="run"
        shift
        ;;
      --app)
        RUNTIME_MODE="app"
        shift
        ;;
      --manager)
        RUNTIME_MODE="manager"
        shift
        ;;
      --browser-use)
        INSTALL_BROWSER_USE="1"
        shift
        ;;
      --no-browser-use)
        INSTALL_BROWSER_USE="0"
        shift
        ;;
      --cloudflared)
        INSTALL_CLOUDFLARED="1"
        shift
        ;;
      --no-cloudflared)
        INSTALL_CLOUDFLARED="0"
        shift
        ;;
      --dry-run)
        DRY_RUN="1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        PASSTHRU=("$@")
        break
        ;;
      *)
        PASSTHRU+=("$1")
        shift
        ;;
    esac
  done
}

resolve_runtime_mode() {
  case "${RUNTIME_MODE}" in
    app|manager)
      printf '%s\n' "${RUNTIME_MODE}"
      ;;
    auto|"")
      if [[ -n "${PUBLIC_BASE_URL:-}" || -n "${RAILWAY_PUBLIC_DOMAIN:-}" ]]; then
        printf '%s\n' "app"
      else
        printf '%s\n' "manager"
      fi
      ;;
    *)
      die "invalid START_MODE/runtime mode: ${RUNTIME_MODE}"
      ;;
  esac
}

ensure_uv() {
  command -v uv >/dev/null 2>&1 || die "uv is required but was not found in PATH"
}

install_python_deps() {
  ensure_uv
  run_cmd uv sync
}

install_browser_use_deps() {
  if is_falsey "${INSTALL_BROWSER_USE}"; then
    log "skipping Browser Use Chromium install."
    return 0
  fi
  run_cmd uv run playwright install chromium
}

install_cloudflared_linux() {
  local arch deb_arch url tmp installer=()
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|amd64) deb_arch="amd64" ;;
    arm64|aarch64) deb_arch="arm64" ;;
    *) die "unsupported Linux architecture for automatic cloudflared install: ${arch}" ;;
  esac
  command -v curl >/dev/null 2>&1 || die "curl is required for automatic cloudflared install"
  command -v dpkg >/dev/null 2>&1 || die "dpkg is required for automatic cloudflared install"
  url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${deb_arch}.deb"
  tmp="/tmp/cloudflared-linux-${deb_arch}.deb"
  run_cmd curl -L "${url}" -o "${tmp}"
  if [[ "$(id -u)" != "0" ]] && command -v sudo >/dev/null 2>&1; then
    installer=(sudo dpkg -i "${tmp}")
  else
    installer=(dpkg -i "${tmp}")
  fi
  run_cmd "${installer[@]}"
}

install_cloudflared_macos() {
  command -v brew >/dev/null 2>&1 || die "Homebrew is required for automatic cloudflared install on macOS"
  run_cmd brew install cloudflared
}

ensure_cloudflared() {
  if command -v cloudflared >/dev/null 2>&1; then
    return 0
  fi
  if is_falsey "${INSTALL_CLOUDFLARED}"; then
    die "cloudflared is required for manager mode but is not installed"
  fi
  case "$(uname -s)" in
    Darwin)
      install_cloudflared_macos
      ;;
    Linux)
      install_cloudflared_linux
      ;;
    *)
      die "automatic cloudflared install is not supported on this OS"
      ;;
  esac
}

run_app() {
  if ((${#PASSTHRU[@]})); then
    run_cmd uv run python -m opentulpa "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run python -m opentulpa
}

run_manager() {
  if ((${#PASSTHRU[@]})); then
    run_cmd uv run python scripts/manager.py "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run python scripts/manager.py
}

main() {
  load_dotenv
  parse_args "$@"

  local runtime
  runtime="$(resolve_runtime_mode)"

  if [[ "${MODE}" != "run" ]]; then
    install_python_deps
    install_browser_use_deps
    if [[ "${runtime}" == "manager" ]]; then
      ensure_cloudflared
    fi
  fi

  if [[ "${MODE}" == "install" ]]; then
    return 0
  fi

  if [[ "${runtime}" == "app" ]]; then
    log "running direct app mode."
    run_app
    return 0
  fi

  log "running quick-tunnel manager mode."
  run_manager
}

main "$@"
