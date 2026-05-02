#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

cd "${REPO_ROOT}"

MODE="up"
RUNTIME_MODE="${START_MODE:-local}"
INSTALL_BROWSER_USE="${INSTALL_BROWSER_USE:-1}"
INSTALL_CLOUDFLARED="${INSTALL_CLOUDFLARED:-auto}"
INSTALL_UV="${INSTALL_UV:-1}"
UV_PYTHON="${UV_PYTHON:-3.12}"
export UV_PYTHON
ASSUME_YES=0
NO_INSTALL_UV=0
DRY_RUN=0
UV_BOOTSTRAPPED=0
PASSTHRU=()

usage() {
  cat <<'EOF_USAGE'
Usage:
  ./start.sh [local|server|install|run|doctor] [options] [-- extra-args]

Commands:
  local                 Install, then run local Telegram mode: app + Cloudflare tunnel + webhook sync (default)
  server                Install, then run the plain app server
  install               Install/setup only
  run [local|server]    Run only, without installing
  doctor [local|server] Check startup readiness

Compatibility aliases:
  up                    Same as local
  --manager             Deprecated alias for local mode
  --app                 Deprecated alias for server mode
  --install-only        Same as install
  --run-only            Same as run

Options:
  --local               Force local Telegram mode
  --server              Force plain app server mode
  --browser-use         Install Browser Use Chromium
  --no-browser-use      Skip Browser Use Chromium install
  --cloudflared         Install cloudflared when local mode needs it
  --no-cloudflared      Never install cloudflared automatically
  --yes, -y             Answer yes to installer prompts
  --no-install-uv       Never install uv automatically
  --dry-run             Print commands without running them
  -h, --help            Show this help

.env knobs:
  START_MODE=local|server|auto  (app and manager are deprecated aliases)
  INSTALL_BROWSER_USE=1|0
  INSTALL_CLOUDFLARED=auto|1|0
  INSTALL_UV=1|auto|0       (default: 1, bootstrap uv when missing)
  UV_PYTHON=3.12            (default: 3.12)
EOF_USAGE
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

is_interactive() {
  [[ -t 0 && -t 1 ]]
}

log() {
  echo "[start] $*"
}

warn() {
  echo "[start] warning: $*" >&2
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

append_env_value() {
  local key="$1"
  local value="$2"
  printf '\n%s=%s\n' "${key}" "${value}" >> "${REPO_ROOT}/.env"
}

env_is_set() {
  local key="$1"
  local value="${!key:-}"
  [[ -n "${value}" && "${value}" != "..." ]]
}

telegram_allowlist_is_set() {
  env_is_set "TELEGRAM_ALLOWED_USERNAMES" || env_is_set "TELEGRAM_ALLOWED_USER_IDS"
}

yaml_value_is_set() {
  local key="$1"
  local line value
  line="$(grep -E "^[[:space:]]*${key}:[[:space:]]*" "${REPO_ROOT}/opentulpa.config.yaml" 2>/dev/null | head -n 1 || true)"
  [[ -n "${line}" ]] || return 1
  value="${line#*:}"
  value="${value%%#*}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  [[ -n "${value}" && "${value}" != "null" && "${value}" != "\"\"" && "${value}" != "''" ]]
}

multimodal_model_is_set() {
  env_is_set "MULTIMODAL_LLM" || yaml_value_is_set "multimodal_llm"
}

openrouter_base_url_is_set() {
  local base="${OPENAI_COMPATIBLE_BASE_URL:-${OPENROUTER_BASE_URL:-}}"
  base="$(printf '%s' "${base}" | tr '[:upper:]' '[:lower:]')"
  [[ "${base}" == *"openrouter.ai"* ]]
}

emit_model_config_notice() {
  if ! multimodal_model_is_set; then
    log "warning: MULTIMODAL_LLM is not set and opentulpa.config.yaml has no multimodal_llm; image/file/browser functionality may not work."
  fi
  if ! openrouter_base_url_is_set; then
    log "warning: OPENAI_COMPATIBLE_BASE_URL is not OpenRouter. Check opentulpa.config.yaml model settings, especially multimodal_llm, memory_llm_model, and business_knowledge_oracle_model."
  fi
}

prompt_env_value() {
  local key="$1"
  local prompt="$2"
  local secret="${3:-0}"
  local default_value="${4:-}"
  local value

  if [[ "${secret}" == "1" ]]; then
    read -r -s -p "${prompt}: " value
    printf '\n'
  elif [[ -n "${default_value}" ]]; then
    read -r -p "${prompt} [${default_value}]: " value
    value="${value:-${default_value}}"
  else
    read -r -p "${prompt}: " value
  fi

  value="${value//[$'\r\n']/}"
  [[ -n "${value}" ]] || die "${key} cannot be blank"
  append_env_value "${key}" "${value}"
  export "${key}=${value}"
  log "saved ${key} to .env"
}

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

normalize_runtime_mode() {
  case "${1:-}" in
    local|server|auto|"")
      printf '%s\n' "${1:-local}"
      ;;
    app)
      warn "START_MODE=app is deprecated; use START_MODE=server."
      printf '%s\n' "server"
      ;;
    manager)
      warn "START_MODE=manager is deprecated; use START_MODE=local."
      printf '%s\n' "local"
      ;;
    *)
      die "invalid START_MODE/runtime mode: ${1}"
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      local|server)
        RUNTIME_MODE="$1"
        MODE="up"
        shift
        ;;
      up)
        MODE="up"
        RUNTIME_MODE="${RUNTIME_MODE:-local}"
        shift
        ;;
      install)
        MODE="install"
        shift
        ;;
      run)
        MODE="run"
        shift
        if [[ $# -gt 0 && ( "$1" == "local" || "$1" == "server" ) ]]; then
          RUNTIME_MODE="$1"
          shift
        fi
        ;;
      doctor)
        MODE="doctor"
        shift
        if [[ $# -gt 0 && ( "$1" == "local" || "$1" == "server" ) ]]; then
          RUNTIME_MODE="$1"
          shift
        fi
        ;;
      --install-only)
        MODE="install"
        shift
        ;;
      --run-only)
        MODE="run"
        shift
        ;;
      --local)
        RUNTIME_MODE="local"
        shift
        ;;
      --server)
        RUNTIME_MODE="server"
        shift
        ;;
      --app)
        warn "--app is deprecated; use server or --server."
        RUNTIME_MODE="server"
        shift
        ;;
      --manager)
        warn "--manager is deprecated; use local or --local."
        RUNTIME_MODE="local"
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
      --yes|-y)
        ASSUME_YES="1"
        shift
        ;;
      --no-install-uv)
        NO_INSTALL_UV="1"
        INSTALL_UV="0"
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
  local normalized
  normalized="$(normalize_runtime_mode "${RUNTIME_MODE}")"
  case "${normalized}" in
    local|server)
      printf '%s\n' "${normalized}"
      ;;
    auto)
      if [[ -n "${PUBLIC_BASE_URL:-}" || -n "${RAILWAY_PUBLIC_DOMAIN:-}" ]]; then
        printf '%s\n' "server"
      else
        printf '%s\n' "local"
      fi
      ;;
    *)
      die "invalid runtime mode: ${normalized}"
      ;;
  esac
}

install_uv() {
  if [[ "${DRY_RUN}" != "1" ]]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to install uv. Install uv manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
  run_cmd sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  UV_BOOTSTRAPPED=1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${UV_BOOTSTRAPPED}" == "1" ]]; then
    return 0
  fi

  if [[ "${NO_INSTALL_UV}" == "1" ]] || is_falsey "${INSTALL_UV}"; then
    die "uv is required but was not found in PATH. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi

  if is_truthy "${INSTALL_UV}" || [[ "${ASSUME_YES}" == "1" ]]; then
    log "uv was not found in PATH; bootstrapping uv."
    install_uv
  elif is_interactive; then
    read -r -p "uv is required and was not found. Install it now? [Y/n] " reply
    case "${reply:-Y}" in
      y|Y|yes|YES) install_uv ;;
      *) die "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
    esac
  else
    die "uv is required but was not found in PATH. Re-run with --yes to install it, or install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  command -v uv >/dev/null 2>&1 || die "uv install completed but uv is still not in PATH. Try opening a new shell or add ~/.local/bin to PATH."
}

ensure_env_file() {
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    return 0
  fi
  [[ -f "${REPO_ROOT}/.env.example" ]] || die ".env is missing and .env.example was not found"
  run_cmd cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
}

ensure_required_env() {
  local runtime="$1"
  local missing=()

  if [[ "${MODE}" == "install" ]]; then
    return 0
  fi
  ensure_env_file
  load_dotenv

  env_is_set "OPENAI_COMPATIBLE_API_KEY" || missing+=("OPENAI_COMPATIBLE_API_KEY")

  if [[ "${runtime}" == "local" ]]; then
    env_is_set "TELEGRAM_BOT_TOKEN" || missing+=("TELEGRAM_BOT_TOKEN")
    if ! telegram_allowlist_is_set; then
      missing+=("TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS")
    fi
  fi

  if [[ "${runtime}" == "server" ]]; then
    env_is_set "TELEGRAM_BOT_TOKEN" || missing+=("TELEGRAM_BOT_TOKEN")
    env_is_set "TELEGRAM_WEBHOOK_SECRET" || missing+=("TELEGRAM_WEBHOOK_SECRET")
    env_is_set "PUBLIC_BASE_URL" || missing+=("PUBLIC_BASE_URL")
    env_is_set "OPENTULPA_DATA_ROOT" || missing+=("OPENTULPA_DATA_ROOT")
    if ! telegram_allowlist_is_set; then
      missing+=("TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS")
    fi
  fi

  if ! env_is_set "COMPOSIO_API_KEY"; then
    log "warning: COMPOSIO_API_KEY is not set; connector integrations such as Google Sheets and Instagram will be unavailable."
  fi
  emit_model_config_notice

  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "required .env value(s) missing for ${runtime}: ${missing[*]}"
    return 0
  fi

  if ! is_interactive; then
    die "required .env value(s) missing for ${runtime}: ${missing[*]}. Set them in .env or run interactively to enter them."
  fi

  env_is_set "OPENAI_COMPATIBLE_API_KEY" || prompt_env_value "OPENAI_COMPATIBLE_API_KEY" "OPENAI_COMPATIBLE_API_KEY" 1
  if [[ "${runtime}" == "local" || "${runtime}" == "server" ]]; then
    env_is_set "TELEGRAM_BOT_TOKEN" || prompt_env_value "TELEGRAM_BOT_TOKEN" "TELEGRAM_BOT_TOKEN" 1
    if ! telegram_allowlist_is_set; then
      prompt_env_value "TELEGRAM_ALLOWED_USERNAMES" "TELEGRAM_ALLOWED_USERNAMES (comma-separated, no @)"
    fi
  fi
  if [[ "${runtime}" == "server" ]]; then
    env_is_set "TELEGRAM_WEBHOOK_SECRET" || prompt_env_value "TELEGRAM_WEBHOOK_SECRET" "TELEGRAM_WEBHOOK_SECRET" 1
    env_is_set "PUBLIC_BASE_URL" || prompt_env_value "PUBLIC_BASE_URL" "PUBLIC_BASE_URL"
    env_is_set "OPENTULPA_DATA_ROOT" || prompt_env_value "OPENTULPA_DATA_ROOT" "OPENTULPA_DATA_ROOT" 0 "/app/opentulpa_data"
  fi
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
    die "cloudflared is required for local mode but is not installed"
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
  ensure_uv
  if ((${#PASSTHRU[@]})); then
    run_cmd uv run python -m opentulpa "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run python -m opentulpa
}

run_manager() {
  ensure_uv
  if ((${#PASSTHRU[@]})); then
    run_cmd uv run python scripts/manager.py "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run python scripts/manager.py
}

doctor_check() {
  local label="$1"
  local ok="$2"
  local fix="${3:-}"
  if [[ "${ok}" == "1" ]]; then
    echo "[doctor] ok: ${label}"
  else
    echo "[doctor] fail: ${label}"
    if [[ -n "${fix}" ]]; then
      echo "[doctor] fix: ${fix}"
    fi
    return 1
  fi
}

run_doctor() {
  local runtime="$1"
  local failures=0

  doctor_check "uv is available" "$(command -v uv >/dev/null 2>&1 && echo 1 || echo 0)" "curl -LsSf https://astral.sh/uv/install.sh | sh" || failures=$((failures + 1))
  doctor_check ".env exists" "$([[ -f "${REPO_ROOT}/.env" ]] && echo 1 || echo 0)" "cp .env.example .env and set required values" || failures=$((failures + 1))
  load_dotenv
  doctor_check "OPENAI_COMPATIBLE_API_KEY is set" "$(env_is_set "OPENAI_COMPATIBLE_API_KEY" && echo 1 || echo 0)" "set OPENAI_COMPATIBLE_API_KEY in .env" || failures=$((failures + 1))
  doctor_check "TELEGRAM_BOT_TOKEN is set" "$(env_is_set "TELEGRAM_BOT_TOKEN" && echo 1 || echo 0)" "set TELEGRAM_BOT_TOKEN in .env" || failures=$((failures + 1))
  doctor_check "Telegram allowlist is set" "$(telegram_allowlist_is_set && echo 1 || echo 0)" "set TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS in .env" || failures=$((failures + 1))
  if env_is_set "COMPOSIO_API_KEY"; then
    echo "[doctor] ok: COMPOSIO_API_KEY is set"
  else
    echo "[doctor] warn: COMPOSIO_API_KEY is not set; connector integrations such as Google Sheets and Instagram will be unavailable"
  fi
  if [[ "${runtime}" == "server" ]]; then
    doctor_check "TELEGRAM_WEBHOOK_SECRET is set" "$(env_is_set "TELEGRAM_WEBHOOK_SECRET" && echo 1 || echo 0)" "set a stable TELEGRAM_WEBHOOK_SECRET in .env" || failures=$((failures + 1))
    doctor_check "PUBLIC_BASE_URL is set" "$(env_is_set "PUBLIC_BASE_URL" && echo 1 || echo 0)" "set PUBLIC_BASE_URL to the public HTTPS URL" || failures=$((failures + 1))
    doctor_check "OPENTULPA_DATA_ROOT is set" "$(env_is_set "OPENTULPA_DATA_ROOT" && echo 1 || echo 0)" "set OPENTULPA_DATA_ROOT=/app/opentulpa_data and mount persistent storage there" || failures=$((failures + 1))
    if env_is_set "OPENTULPA_DATA_ROOT"; then
      doctor_check "OPENTULPA_DATA_ROOT is writable" "$(mkdir -p "${OPENTULPA_DATA_ROOT}" 2>/dev/null && [[ -w "${OPENTULPA_DATA_ROOT}" ]] && echo 1 || echo 0)" "mount a writable persistent volume at OPENTULPA_DATA_ROOT" || failures=$((failures + 1))
    fi
  fi
  doctor_check ".opentulpa is writable" "$(mkdir -p "${REPO_ROOT}/.opentulpa" 2>/dev/null && [[ -w "${REPO_ROOT}/.opentulpa" ]] && echo 1 || echo 0)" "make .opentulpa writable" || failures=$((failures + 1))
  doctor_check "tulpa_stuff is writable" "$(mkdir -p "${REPO_ROOT}/tulpa_stuff" 2>/dev/null && [[ -w "${REPO_ROOT}/tulpa_stuff" ]] && echo 1 || echo 0)" "make tulpa_stuff writable" || failures=$((failures + 1))

  if command -v lsof >/dev/null 2>&1; then
    local listeners
    listeners="$(lsof -t -iTCP:"${PORT:-8000}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${listeners}" ]]; then
      echo "[doctor] info: port ${PORT:-8000} is already in use by PID(s): ${listeners//$'\n'/,}"
    else
      echo "[doctor] ok: port ${PORT:-8000} is free"
    fi
  else
    echo "[doctor] info: lsof not available; skipping port check"
  fi

  if [[ "${runtime}" == "local" ]]; then
    doctor_check "cloudflared is available for local mode" "$(command -v cloudflared >/dev/null 2>&1 && echo 1 || echo 0)" "install cloudflared or run ./start.sh server" || failures=$((failures + 1))
  fi

  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "http://127.0.0.1:${PORT:-8000}/healthz" >/dev/null 2>&1; then
      echo "[doctor] ok: /healthz is responding"
    else
      echo "[doctor] info: /healthz is not responding; app may not be running"
    fi
    if curl -fsS "http://127.0.0.1:${PORT:-8000}/agent/healthz" >/dev/null 2>&1; then
      echo "[doctor] ok: /agent/healthz is responding"
    else
      echo "[doctor] info: /agent/healthz is not responding; app may not be running"
    fi
  else
    echo "[doctor] info: curl not available; skipping health endpoint checks"
  fi

  if [[ "${failures}" -gt 0 ]]; then
    die "doctor found ${failures} problem(s)"
  fi
  log "doctor checks passed."
}

main() {
  parse_args "$@"
  RUNTIME_MODE="$(normalize_runtime_mode "${RUNTIME_MODE}")"
  load_dotenv

  local runtime
  runtime="$(resolve_runtime_mode)"

  if [[ "${MODE}" == "doctor" ]]; then
    run_doctor "${runtime}"
    return 0
  fi

  if [[ "${MODE}" != "run" ]]; then
    install_python_deps
    install_browser_use_deps
    if [[ "${runtime}" == "local" ]]; then
      ensure_cloudflared
    fi
  fi

  if [[ "${MODE}" == "install" ]]; then
    return 0
  fi

  ensure_required_env "${runtime}"

  if [[ "${runtime}" == "server" ]]; then
    log "running server mode."
    run_app
    return 0
  fi

  log "running local Telegram mode."
  run_manager
}

main "$@"
