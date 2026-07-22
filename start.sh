#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

cd "${REPO_ROOT}"

MODE="up"
RUNTIME_MODE="${START_MODE:-server}"
INSTALL_BROWSER_USE="${INSTALL_BROWSER_USE:-0}"
INSTALL_CLOUDFLARED="${INSTALL_CLOUDFLARED:-auto}"
INSTALL_UV="${INSTALL_UV:-1}"
UV_PYTHON="${UV_PYTHON:-3.12}"
export UV_PYTHON
ASSUME_YES=0
NO_INSTALL_UV=0
DRY_RUN=0
UV_BOOTSTRAPPED=0
DIRECT_ENGINE_AVAILABLE=0
PASSTHRU=()
SELECTED_EXTRAS=()
SERVE_MODE=0
SERVER_PORT=""
SERVER_HOST=""
SERVER_DATA_ROOT=""
SERVER_PUBLIC_URL=""
CLI_API_KEY=""
CLI_TELEGRAM_BOT_TOKEN=""
CLI_TELEGRAM_USER_ID=""
CLI_OWNER_TOKEN=""

usage() {
  cat <<'EOF_USAGE'
Usage:
  ./start.sh [serve|local|server|managed|install|run|doctor] [options] [-- extra-args]

Commands:
  serve                 Start the headless host, Agent API, and configured interfaces
  local                 Install, then run local Telegram mode: app + Cloudflare tunnel + webhook sync
  server                Install, then run the headless Agent API directly
  managed               Install trusted OCI images, then run the self-replacing bootstrap
  install               Install/setup only
  run [local|server|managed] Run only, without installing
  doctor [local|server|managed] Check startup readiness

Compatibility aliases:
  up                    Same as server
  --manager             Deprecated alias for local mode
  --app                 Deprecated alias for server mode
  --install-only        Same as install
  --run-only            Same as run

Options:
  --local               Force local Telegram mode
  --server              Force plain app server mode
  --managed             Force immutable-bootstrap managed mode
  --browser-use         Install Browser Use Cloud adapter dependencies
  --no-browser-use      Skip Browser Use Cloud adapter dependencies
  --cloudflared         Install cloudflared when local mode needs it
  --no-cloudflared      Never install cloudflared automatically
  --yes, -y             Answer yes to installer prompts
  --no-install-uv       Never install uv automatically
  --dry-run             Print commands without running them
  --api-key KEY         OpenAI-compatible model API key
  --telegram-bot-token TOKEN
                        Telegram bot token; requires --telegram-user-id
  --telegram-user-id ID Telegram numeric owner ID; requires --telegram-bot-token
  --owner-token TOKEN   Optional remote Agent API owner token
  --host HOST           Bind host (local default: 127.0.0.1)
  --port PORT           Server port (default: PORT or 8000)
  --data-root PATH      Persistent data directory
  --public-url URL      Public base URL advertised by this deployment
  -h, --help            Show this help

.env knobs:
  START_MODE=server|managed|local|auto  (default: server; app and manager are deprecated aliases)
  INSTALL_BROWSER_USE=1|0        (default: 0; browser is an optional capability)
  OPENTULPA_EXTRAS=integrations,documents  (optional comma/space-separated extras)
  INSTALL_CLOUDFLARED=auto|1|0
  INSTALL_UV=1|auto|0       (default: 1, bootstrap uv when missing)
  UV_PYTHON=3.12            (default: 3.12)
  OPENTULPA_OPEN_BROWSER=auto|1|0  (default: auto; open only for an interactive local start)
  OPENTULPA_RESTART_GRACE_SECONDS=15  (graceful replacement wait, capped at 300)
  SANDBOX_IMAGE=opentulpa-tenant-sandbox:0.1.0
EOF_USAGE
}

configure_python_extras() {
  local raw item seen=""
  raw="${OPENTULPA_EXTRAS:-}"
  raw="${raw//,/ }"
  if is_truthy "${INSTALL_BROWSER_USE}"; then
    raw="${raw} browser"
  fi
  SELECTED_EXTRAS=()
  for item in ${raw}; do
    case "${item}" in
      browser|integrations|documents|research|bundled) ;;
      *) die "unsupported OPENTULPA_EXTRAS value: ${item}" ;;
    esac
    case " ${seen} " in
      *" ${item} "*) continue ;;
    esac
    SELECTED_EXTRAS+=("${item}")
    seen="${seen} ${item}"
  done
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

upsert_env_value() {
  local key="$1"
  local value="$2"
  local env_file="${REPO_ROOT}/.env" temporary found=0 raw line_key

  [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "invalid .env key: ${key}"
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || die "${key} cannot contain newlines"
  ensure_env_file || die ".env.example was not found"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "save ${key} in .env"
    return 0
  fi

  umask 077
  temporary="$(mktemp "${REPO_ROOT}/.env.tmp.XXXXXX")"
  while IFS= read -r raw || [[ -n "${raw}" ]]; do
    line_key="${raw%%=*}"
    line_key="${line_key#"${line_key%%[![:space:]]*}"}"
    line_key="${line_key%"${line_key##*[![:space:]]}"}"
    if [[ "${raw}" == *=* && "${line_key}" == "${key}" ]]; then
      if [[ "${found}" == "0" ]]; then
        printf '%s=%s\n' "${key}" "${value}" >> "${temporary}"
        found=1
      fi
      continue
    fi
    printf '%s\n' "${raw}" >> "${temporary}"
  done < "${env_file}"
  if [[ "${found}" == "0" ]]; then
    printf '\n%s=%s\n' "${key}" "${value}" >> "${temporary}"
  fi
  chmod 600 "${temporary}"
  mv -f "${temporary}" "${env_file}"
}

env_is_set() {
  local key="$1"
  local value="${!key:-}"
  [[ -n "${value}" && "${value}" != "..." ]]
}

telegram_allowlist_is_set() {
  env_is_set "TELEGRAM_ALLOWED_USERNAMES" || env_is_set "TELEGRAM_ALLOWED_USER_IDS"
}

public_base_url_is_set() {
  env_is_set "PUBLIC_BASE_URL" || env_is_set "RAILWAY_PUBLIC_DOMAIN"
}

server_telegram_enabled() {
  env_is_set "TELEGRAM_BOT_TOKEN" || telegram_allowlist_is_set
}

yaml_value() {
  local key="$1"
  local line value
  line="$(grep -E "^[[:space:]]*${key}:[[:space:]]*" "${REPO_ROOT}/opentulpa.config.yaml" 2>/dev/null | head -n 1 || true)"
  [[ -n "${line}" ]] || return 0
  value="${line#*:}"
  value="${value%%#*}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  [[ "${value}" == "null" ]] && return 0
  printf '%s\n' "${value}"
}

config_value() {
  local env_key="$1"
  local yaml_key="$2"
  local value="${!env_key:-}"
  if [[ -n "${value}" && "${value}" != "..." ]]; then
    printf '%s\n' "${value}"
    return 0
  fi
  yaml_value "${yaml_key}"
}

openrouter_base_url_is_set() {
  local base="${OPENAI_COMPATIBLE_BASE_URL:-${OPENROUTER_BASE_URL:-}}"
  base="$(printf '%s' "${base}" | tr '[:upper:]' '[:lower:]')"
  [[ "${base}" == *"openrouter.ai"* ]]
}

emit_model_config_notice() {
  if ! openrouter_base_url_is_set; then
    log "warning: OPENAI_COMPATIBLE_BASE_URL is not OpenRouter. Check opentulpa.config.yaml model settings for this provider: llm_model and business_knowledge_oracle_model."
  fi
}

check_model_catalog() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if ! env_is_set "OPENAI_COMPATIBLE_API_KEY"; then
    return 0
  fi
  command -v curl >/dev/null 2>&1 || {
    log "info: curl is not available; skipping OpenAI-compatible /models check."
    return 0
  }
  command -v python3 >/dev/null 2>&1 || {
    log "info: python3 is not available; skipping OpenAI-compatible /models check."
    return 0
  }

  local base_url="${OPENAI_COMPATIBLE_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}"
  base_url="${base_url%/}"

  local -a role_specs=(
    "llm_model|LLM_MODEL|llm_model"
    "llm_provider_rejection_fallback_model|LLM_PROVIDER_REJECTION_FALLBACK_MODEL|llm_provider_rejection_fallback_model"
    "business_knowledge_oracle_model|BUSINESS_KNOWLEDGE_ORACLE_MODEL|business_knowledge_oracle_model"
  )
  local expected_lines="" spec role env_key yaml_key model
  for spec in "${role_specs[@]}"; do
    IFS='|' read -r role env_key yaml_key <<<"${spec}"
    model="$(config_value "${env_key}" "${yaml_key}")"
    [[ -n "${model}" && "${model}" != "null" ]] || continue
    expected_lines+="${role}|${model}"$'\n'
  done
  [[ -n "${expected_lines}" ]] || return 0

  local catalog
  if ! catalog="$(curl -fsS -H "Authorization: Bearer ${OPENAI_COMPATIBLE_API_KEY}" "${base_url}/models" 2>/dev/null)"; then
    log "warning: could not fetch ${base_url}/models; verify opentulpa.config.yaml model IDs against your provider."
    return 0
  fi

  local missing
  missing="$(
    EXPECTED_MODELS="${expected_lines}" python3 -c '
import json
import os
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(2)

items = payload.get("data") if isinstance(payload, dict) else payload
ids = set()
if isinstance(items, list):
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            ids.add(str(item["id"]))
        elif isinstance(item, str):
            ids.add(item)

missing = []
for line in os.environ.get("EXPECTED_MODELS", "").splitlines():
    if not line.strip() or "|" not in line:
        continue
    role, model = line.split("|", 1)
    if model not in ids:
        missing.append(f"{role}={model}")

if missing:
    print(", ".join(missing))
' <<<"${catalog}" || printf '%s' "__parse_error__"
  )"
  if [[ "${missing}" == "__parse_error__" ]]; then
    log "warning: ${base_url}/models returned an unexpected response; verify opentulpa.config.yaml model IDs manually."
  elif [[ -n "${missing}" ]]; then
    log "warning: ${base_url}/models did not list configured model(s): ${missing}. Update opentulpa.config.yaml or provider env overrides."
  else
    log "OpenAI-compatible /models check passed for configured model IDs."
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
  upsert_env_value "${key}" "${value}"
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
    local|server|managed|auto|"")
      printf '%s\n' "${1:-server}"
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
      local|server|managed)
        SERVE_MODE=0
        RUNTIME_MODE="$1"
        MODE="up"
        shift
        ;;
      serve)
        SERVE_MODE=1
        MODE="up"
        RUNTIME_MODE="server"
        shift
        ;;
      up)
        MODE="up"
        RUNTIME_MODE="${RUNTIME_MODE:-server}"
        shift
        ;;
      install)
        MODE="install"
        shift
        if [[ $# -gt 0 && ( "$1" == "local" || "$1" == "server" || "$1" == "managed" ) ]]; then
          SERVE_MODE=0
          RUNTIME_MODE="$1"
          shift
        fi
        ;;
      run)
        MODE="run"
        shift
        if [[ $# -gt 0 && ( "$1" == "local" || "$1" == "server" || "$1" == "managed" ) ]]; then
          SERVE_MODE=0
          RUNTIME_MODE="$1"
          shift
        fi
        ;;
      doctor)
        MODE="doctor"
        shift
        if [[ $# -gt 0 && ( "$1" == "local" || "$1" == "server" || "$1" == "managed" ) ]]; then
          SERVE_MODE=0
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
        SERVE_MODE=0
        RUNTIME_MODE="local"
        shift
        ;;
      --server)
        SERVE_MODE=0
        RUNTIME_MODE="server"
        shift
        ;;
      --managed)
        SERVE_MODE=0
        RUNTIME_MODE="managed"
        shift
        ;;
      --app)
        warn "--app is deprecated; use server or --server."
        SERVE_MODE=0
        RUNTIME_MODE="server"
        shift
        ;;
      --manager)
        warn "--manager is deprecated; use local or --local."
        SERVE_MODE=0
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
      --api-key|--openai-compatible-api-key|--openapi-compatible-api-key)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        CLI_API_KEY="$2"
        shift 2
        ;;
      --telegram-bot-token)
        [[ $# -ge 2 ]] || die "--telegram-bot-token requires a value"
        CLI_TELEGRAM_BOT_TOKEN="$2"
        shift 2
        ;;
      --telegram-user-id)
        [[ $# -ge 2 ]] || die "--telegram-user-id requires a value"
        CLI_TELEGRAM_USER_ID="$2"
        shift 2
        ;;
      --owner-token)
        [[ $# -ge 2 ]] || die "--owner-token requires a value"
        CLI_OWNER_TOKEN="$2"
        shift 2
        ;;
      --host)
        [[ $# -ge 2 ]] || die "--host requires a value"
        SERVER_HOST="$2"
        shift 2
        ;;
      --port)
        [[ $# -ge 2 ]] || die "--port requires a value"
        SERVER_PORT="$2"
        shift 2
        ;;
      --data-root)
        [[ $# -ge 2 ]] || die "--data-root requires a value"
        SERVER_DATA_ROOT="$2"
        shift 2
        ;;
      --public-url)
        [[ $# -ge 2 ]] || die "--public-url requires a value"
        SERVER_PUBLIC_URL="$2"
        shift 2
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

default_to_serve() {
  local argument
  [[ -z "${START_MODE:-}" ]] || return 0
  for argument in "$@"; do
    case "${argument}" in
      serve|local|server|managed|install|run|doctor|--local|--server|--managed|--app|--manager)
        return 0
        ;;
    esac
  done
  SERVE_MODE=1
}

resolve_runtime_mode() {
  local normalized
  normalized="$(normalize_runtime_mode "${RUNTIME_MODE}")"
  case "${normalized}" in
    local|server|managed)
      printf '%s\n' "${normalized}"
      ;;
    auto)
      printf '%s\n' "server"
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
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "chmod 600 ${REPO_ROOT}/.env"
    else
      chmod 600 "${REPO_ROOT}/.env"
    fi
    return 0
  fi
  [[ -f "${REPO_ROOT}/.env.example" ]] || return 1
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "cp ${REPO_ROOT}/.env.example ${REPO_ROOT}/.env"
    log "chmod 600 ${REPO_ROOT}/.env"
    return 0
  fi
  umask 077
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  chmod 600 "${REPO_ROOT}/.env"
}

host_is_loopback() {
  case "${1:-}" in
    127.0.0.1|localhost|::1|\[::1\]) return 0 ;;
    *) return 1 ;;
  esac
}

local_server_bootstrap_enabled() {
  ! public_base_url_is_set && { [[ -z "${HOST:-}" ]] || host_is_loopback "${HOST}"; }
}

generate_owner_token() {
  local token=""
  if command -v openssl >/dev/null 2>&1; then
    token="$(openssl rand -hex 32)"
  elif [[ -r /dev/urandom ]] && command -v od >/dev/null 2>&1; then
    token="$(od -An -N32 -tx1 /dev/urandom | tr -d '[:space:]')"
  fi
  [[ "${token}" =~ ^[0-9a-f]{64}$ ]] || die "could not generate a secure owner credential"
  printf '%s\n' "${token}"
}

apply_serve_value() {
  local key="$1" value="$2"
  [[ -n "${value}" ]] || return 0
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || die "${key} cannot contain newlines"
  export "${key}=${value}"
  upsert_env_value "${key}" "${value}"
}

configure_serve() {
  local bot_configured=0 owner_configured=0
  [[ "${SERVE_MODE}" == "1" ]] || return 0

  [[ -z "${CLI_API_KEY}" ]] || export OPENAI_COMPATIBLE_API_KEY="${CLI_API_KEY}"
  [[ -z "${CLI_TELEGRAM_BOT_TOKEN}" ]] || export TELEGRAM_BOT_TOKEN="${CLI_TELEGRAM_BOT_TOKEN}"
  if [[ -n "${CLI_TELEGRAM_USER_ID}" && ! "${CLI_TELEGRAM_USER_ID}" =~ ^[1-9][0-9]*$ ]]; then
    die "--telegram-user-id must be a positive numeric Telegram user ID"
  fi
  [[ -z "${CLI_TELEGRAM_USER_ID}" ]] || export TELEGRAM_ALLOWED_USER_IDS="${CLI_TELEGRAM_USER_ID}"
  [[ -z "${CLI_OWNER_TOKEN}" ]] || export OPENTULPA_OWNER_TOKEN="${CLI_OWNER_TOKEN}"
  if [[ -n "${CLI_API_KEY}${CLI_TELEGRAM_BOT_TOKEN}${CLI_TELEGRAM_USER_ID}${CLI_OWNER_TOKEN}" ]]; then
    warn "secrets passed as arguments can remain in shell history; use the setup UI when possible"
  fi

  if [[ -n "${SERVER_HOST}" ]]; then
    [[ "${SERVER_HOST}" != *[[:space:]]* ]] || die "--host cannot contain whitespace"
    apply_serve_value "HOST" "${SERVER_HOST}"
  fi
  if [[ -n "${SERVER_PORT}" ]]; then
    [[ "${SERVER_PORT}" =~ ^[0-9]+$ ]] && ((SERVER_PORT >= 1 && SERVER_PORT <= 65535)) \
      || die "--port must be an integer between 1 and 65535"
    apply_serve_value "PORT" "${SERVER_PORT}"
  fi
  if [[ -n "${SERVER_DATA_ROOT}" ]]; then
    [[ "${SERVER_DATA_ROOT}" == /* ]] || die "--data-root must be an absolute path"
    apply_serve_value "OPENTULPA_DATA_ROOT" "${SERVER_DATA_ROOT}"
  fi
  if [[ -n "${SERVER_PUBLIC_URL}" ]]; then
    [[ "${SERVER_PUBLIC_URL}" =~ ^https?://[^[:space:]]+$ ]] \
      || die "--public-url must be an absolute HTTP(S) URL"
    apply_serve_value "PUBLIC_BASE_URL" "${SERVER_PUBLIC_URL%/}"
    export HOST="${HOST:-0.0.0.0}"
  fi

  env_is_set "TELEGRAM_BOT_TOKEN" && bot_configured=1
  telegram_allowlist_is_set && owner_configured=1
  [[ "${bot_configured}" == "${owner_configured}" ]] \
    || die "Telegram requires both --telegram-bot-token and --telegram-user-id"

  RUNTIME_MODE="server"
}

configure_local_server_defaults() {
  local data_root token_dir token_path token
  [[ ( "$1" == "server" || "${SERVE_MODE}" == "1" ) \
    && "${MODE}" != "install" && "${MODE}" != "doctor" ]] || return 0
  [[ "${SERVE_MODE}" == "1" ]] || local_server_bootstrap_enabled || return 0

  if [[ -z "${HOST:-}" ]]; then
    if public_base_url_is_set; then
      export HOST="0.0.0.0"
    else
      export HOST="127.0.0.1"
      log "binding the local Agent API to 127.0.0.1"
    fi
  fi
  if ! env_is_set "OPENTULPA_DATA_ROOT"; then
    [[ -n "${HOME:-}" ]] || die "HOME is required to select the local OpenTulpa data directory"
    data_root="${XDG_DATA_HOME:-${HOME}/.local/share}/opentulpa"
    export OPENTULPA_DATA_ROOT="${data_root}"
    log "using local data at ${OPENTULPA_DATA_ROOT}"
  fi
  env_is_set "OPENTULPA_OWNER_TOKEN" && return 0
  if [[ "${SERVE_MODE}" == "1" ]] && ! host_is_loopback "${HOST:-}"; then
    log "remote host is unclaimed; use the one-time pairing code printed at startup."
    return 0
  fi

  token_dir="${OPENTULPA_DATA_ROOT}/bootstrap"
  token_path="${token_dir}/owner.token"
  if [[ "${DRY_RUN}" == "1" ]]; then
    export OPENTULPA_OWNER_TOKEN="dry-run-local-owner-token"
    log "use the private generated owner credential in ${token_path}"
    return 0
  fi

  umask 077
  mkdir -p "${token_dir}"
  [[ ! -L "${token_dir}" ]] || die "owner credential directory cannot be a symbolic link"
  chmod 700 "${token_dir}"
  if [[ ! -e "${token_path}" ]]; then
    token="$(generate_owner_token)"
    (umask 077; printf '%s\n' "${token}" > "${token_path}")
  fi
  [[ -f "${token_path}" && ! -L "${token_path}" ]] || die "owner credential must be a regular file"
  chmod 600 "${token_path}"
  token="$(tr -d '\r\n' < "${token_path}")"
  [[ ${#token} -ge 32 && ${#token} -le 500 && "${token}" =~ ^[A-Za-z0-9_-]+$ ]] \
    || die "owner credential file is invalid: ${token_path}"
  export OPENTULPA_OWNER_TOKEN="${token}"
  log "using the private generated owner credential"
}

ensure_required_env() {
  local runtime="$1"
  local missing=()

  if [[ "${MODE}" == "install" ]]; then
    return 0
  fi
  ensure_env_file || true
  load_dotenv

  if [[ "${SERVE_MODE}" == "1" ]]; then
    log "stable host can start before model and interface credentials are configured."
    return 0
  fi

  env_is_set "OPENAI_COMPATIBLE_API_KEY" || missing+=("OPENAI_COMPATIBLE_API_KEY")

  if [[ "${runtime}" == "local" ]]; then
    env_is_set "TELEGRAM_BOT_TOKEN" || missing+=("TELEGRAM_BOT_TOKEN")
    if ! telegram_allowlist_is_set; then
      missing+=("TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS")
    fi
  fi

  if [[ "${runtime}" == "server" ]]; then
    env_is_set "OPENTULPA_DATA_ROOT" || missing+=("OPENTULPA_DATA_ROOT")
    env_is_set "OPENTULPA_OWNER_TOKEN" || missing+=("OPENTULPA_OWNER_TOKEN")
    if server_telegram_enabled; then
      env_is_set "TELEGRAM_BOT_TOKEN" || missing+=("TELEGRAM_BOT_TOKEN")
      env_is_set "TELEGRAM_WEBHOOK_SECRET" || missing+=("TELEGRAM_WEBHOOK_SECRET")
      public_base_url_is_set || missing+=("PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN")
      if ! telegram_allowlist_is_set; then
        missing+=("TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS")
      fi
    else
      log "server Telegram disabled; Agent API startup does not require Telegram env."
    fi
  fi

  if [[ "${runtime}" == "managed" ]]; then
    env_is_set "OPENTULPA_OWNER_TOKEN" || missing+=("OPENTULPA_OWNER_TOKEN")
    env_is_set "OPENTULPA_RECOVERY_TOKEN" || missing+=("OPENTULPA_RECOVERY_TOKEN")
    env_is_set "OPENTULPA_INGRESS_TOKEN" || missing+=("OPENTULPA_INGRESS_TOKEN")
    env_is_set "OPENTULPA_RELEASE_EGRESS_NETWORK" || missing+=("OPENTULPA_RELEASE_EGRESS_NETWORK")
    env_is_set "OPENTULPA_RELEASE_BASE_IMAGE" || missing+=("OPENTULPA_RELEASE_BASE_IMAGE")
    if server_telegram_enabled; then
      env_is_set "TELEGRAM_BOT_TOKEN" || missing+=("TELEGRAM_BOT_TOKEN")
      env_is_set "TELEGRAM_WEBHOOK_SECRET" || missing+=("TELEGRAM_WEBHOOK_SECRET")
      public_base_url_is_set || missing+=("PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN")
      if ! telegram_allowlist_is_set; then
        missing+=("TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS")
      fi
    fi
  fi

  if ! env_is_set "COMPOSIO_API_KEY"; then
    log "warning: COMPOSIO_API_KEY is not set; connector integrations such as Google Sheets and Instagram will be unavailable."
  fi
  emit_model_config_notice
  check_model_catalog

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
  if [[ "${runtime}" == "local" ]]; then
    env_is_set "TELEGRAM_BOT_TOKEN" || prompt_env_value "TELEGRAM_BOT_TOKEN" "TELEGRAM_BOT_TOKEN" 1
    if ! telegram_allowlist_is_set; then
      prompt_env_value "TELEGRAM_ALLOWED_USERNAMES" "TELEGRAM_ALLOWED_USERNAMES (comma-separated, no @)"
    fi
  fi
  if [[ "${runtime}" == "server" ]]; then
    env_is_set "OPENTULPA_OWNER_TOKEN" || prompt_env_value "OPENTULPA_OWNER_TOKEN" "OPENTULPA_OWNER_TOKEN" 1
    if server_telegram_enabled; then
      env_is_set "TELEGRAM_BOT_TOKEN" || prompt_env_value "TELEGRAM_BOT_TOKEN" "TELEGRAM_BOT_TOKEN" 1
      if ! telegram_allowlist_is_set; then
        prompt_env_value "TELEGRAM_ALLOWED_USERNAMES" "TELEGRAM_ALLOWED_USERNAMES (comma-separated, no @)"
      fi
      env_is_set "TELEGRAM_WEBHOOK_SECRET" || prompt_env_value "TELEGRAM_WEBHOOK_SECRET" "TELEGRAM_WEBHOOK_SECRET" 1
      public_base_url_is_set || prompt_env_value "PUBLIC_BASE_URL" "PUBLIC_BASE_URL"
    fi
    env_is_set "OPENTULPA_DATA_ROOT" || prompt_env_value "OPENTULPA_DATA_ROOT" "OPENTULPA_DATA_ROOT" 0 "/app/opentulpa_data"
  fi
  if [[ "${runtime}" == "managed" ]]; then
    env_is_set "OPENTULPA_OWNER_TOKEN" || prompt_env_value "OPENTULPA_OWNER_TOKEN" "OPENTULPA_OWNER_TOKEN" 1
    env_is_set "OPENTULPA_RECOVERY_TOKEN" || prompt_env_value "OPENTULPA_RECOVERY_TOKEN" "OPENTULPA_RECOVERY_TOKEN (32+ random characters)" 1
    env_is_set "OPENTULPA_INGRESS_TOKEN" || prompt_env_value "OPENTULPA_INGRESS_TOKEN" "OPENTULPA_INGRESS_TOKEN (32+ random characters)" 1
    env_is_set "OPENTULPA_RELEASE_EGRESS_NETWORK" || prompt_env_value "OPENTULPA_RELEASE_EGRESS_NETWORK" "OPENTULPA_RELEASE_EGRESS_NETWORK"
    env_is_set "OPENTULPA_RELEASE_BASE_IMAGE" || prompt_env_value "OPENTULPA_RELEASE_BASE_IMAGE" "OPENTULPA_RELEASE_BASE_IMAGE" 0 "opentulpa-runtime-base:0.1.0"
    if server_telegram_enabled; then
      env_is_set "TELEGRAM_BOT_TOKEN" || prompt_env_value "TELEGRAM_BOT_TOKEN" "TELEGRAM_BOT_TOKEN" 1
      env_is_set "TELEGRAM_WEBHOOK_SECRET" || prompt_env_value "TELEGRAM_WEBHOOK_SECRET" "TELEGRAM_WEBHOOK_SECRET" 1
      public_base_url_is_set || prompt_env_value "PUBLIC_BASE_URL" "PUBLIC_BASE_URL"
      if ! telegram_allowlist_is_set; then
        prompt_env_value "TELEGRAM_ALLOWED_USERNAMES" "TELEGRAM_ALLOWED_USERNAMES (comma-separated, no @)"
      fi
    fi
  fi
}

install_python_deps() {
  ensure_uv
  local -a arguments=(sync --no-dev)
  local extra
  for extra in "${SELECTED_EXTRAS[@]-}"; do
    [[ -n "${extra}" ]] || continue
    arguments+=(--extra "${extra}")
  done
  run_cmd uv "${arguments[@]}"
}

install_tenant_sandbox_image() {
  local engine tenant_image
  if [[ "${DIRECT_ENGINE_AVAILABLE}" != "1" ]]; then
    log "tenant sandbox image build skipped; chat will start with shell execution unavailable."
    return 0
  fi
  engine="${OPENTULPA_CONTAINER_CLI:-docker}"
  tenant_image="$(config_value "SANDBOX_IMAGE" "sandbox_image")"
  tenant_image="${tenant_image:-opentulpa-tenant-sandbox:0.1.0}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    command -v "${engine}" >/dev/null 2>&1 || die "${engine} is required for tenant sandbox execution"
  fi
  run_cmd "${engine}" build \
    --tag "${tenant_image}" \
    --file docker/tenant-sandbox.Dockerfile .
}

install_managed_images() {
  local engine runtime_image sandbox_image evaluator_image tenant_image extras_csv
  engine="${OPENTULPA_CONTAINER_CLI:-docker}"
  runtime_image="${OPENTULPA_RELEASE_BASE_IMAGE:-opentulpa-runtime-base:0.1.0}"
  sandbox_image="${EVOLUTION_SANDBOX_IMAGE:-opentulpa-evolution:0.1.0}"
  evaluator_image="${EVOLUTION_EVALUATOR_IMAGE:-${sandbox_image}}"
  tenant_image="$(config_value "SANDBOX_IMAGE" "sandbox_image")"
  tenant_image="${tenant_image:-opentulpa-tenant-sandbox:0.1.0}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    command -v "${engine}" >/dev/null 2>&1 || die "${engine} is required for managed mode"
  fi
  if ((${#SELECTED_EXTRAS[@]})); then
    extras_csv="$(IFS=,; printf '%s' "${SELECTED_EXTRAS[*]}")"
    run_cmd "${engine}" build --build-arg "OPENTULPA_EXTRAS=${extras_csv}" \
      --tag "${runtime_image}" --file Dockerfile .
  else
    run_cmd "${engine}" build --tag "${runtime_image}" --file Dockerfile .
  fi
  run_cmd "${engine}" build \
    --tag "${sandbox_image}" \
    --tag "${evaluator_image}" \
    --file docker/evolution.Dockerfile .
  run_cmd "${engine}" build \
    --tag "${tenant_image}" \
    --file docker/tenant-sandbox.Dockerfile .
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
    run_cmd uv run --no-sync python -m opentulpa "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run --no-sync python -m opentulpa
}

run_host() {
  ensure_uv
  if ((${#PASSTHRU[@]})); then
    run_cmd uv run --no-sync python -m opentulpa.host "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run --no-sync python -m opentulpa.host
}

listener_pids_for_port() {
  local port="$1"
  lsof -nP -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u
}

stop_existing_server() {
  local port grace_seconds attempts listeners remaining pid command_line
  local -a opentulpa_pids=()

  [[ "${DRY_RUN}" == "0" ]] || return 0
  port="$(config_value "PORT" "port")"
  port="${port:-8000}"
  command -v lsof >/dev/null 2>&1 || return 0
  listeners="$(listener_pids_for_port "${port}" || true)"
  [[ -n "${listeners}" ]] || return 0

  for pid in ${listeners}; do
    [[ "${pid}" =~ ^[0-9]+$ ]] || die "invalid listener PID reported for port ${port}"
    command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    [[ -n "${command_line}" ]] || continue
    if [[ "${command_line}" != *" -m opentulpa"* ]]; then
      die "port ${port} is used by PID ${pid}, which is not OpenTulpa; refusing to stop it"
    fi
    opentulpa_pids+=("${pid}")
  done

  if ((${#opentulpa_pids[@]} == 0)); then
    remaining="$(listener_pids_for_port "${port}" || true)"
    [[ -z "${remaining}" ]] || die "port ${port} is in use and its owner could not be verified"
    return 0
  fi

  log "stopping existing OpenTulpa server on port ${port} (PID(s): ${opentulpa_pids[*]})"
  kill -TERM "${opentulpa_pids[@]}" 2>/dev/null || true

  grace_seconds="${OPENTULPA_RESTART_GRACE_SECONDS:-15}"
  if [[ ! "${grace_seconds}" =~ ^[0-9]+$ ]]; then
    die "OPENTULPA_RESTART_GRACE_SECONDS must be a non-negative integer"
  fi
  ((grace_seconds > 300)) && grace_seconds=300
  attempts=$((grace_seconds * 10))
  while ((attempts > 0)); do
    remaining="$(listener_pids_for_port "${port}" || true)"
    [[ -n "${remaining}" ]] || break
    sleep 0.1
    attempts=$((attempts - 1))
  done

  remaining="$(listener_pids_for_port "${port}" || true)"
  if [[ -n "${remaining}" ]]; then
    warn "existing OpenTulpa did not stop within ${grace_seconds}s; forcing its verified process to stop"
    for pid in "${opentulpa_pids[@]}"; do
      command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
      if [[ "${command_line}" == *" -m opentulpa"* ]]; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    done
    attempts=50
    while ((attempts > 0)); do
      remaining="$(listener_pids_for_port "${port}" || true)"
      [[ -n "${remaining}" ]] || break
      sleep 0.1
      attempts=$((attempts - 1))
    done
  fi

  remaining="$(listener_pids_for_port "${port}" || true)"
  [[ -z "${remaining}" ]] || die "port ${port} is still in use after stopping OpenTulpa"
  log "existing OpenTulpa server stopped"
}

run_bootstrap() {
  ensure_uv
  if ((${#PASSTHRU[@]})); then
    run_cmd uv run --no-sync opentulpa-bootstrap "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run --no-sync opentulpa-bootstrap
}

run_manager() {
  ensure_uv
  if ((${#PASSTHRU[@]})); then
    run_cmd uv run --no-sync python scripts/manager.py "${PASSTHRU[@]}"
    return 0
  fi
  run_cmd uv run --no-sync python scripts/manager.py
}

open_local_web_when_ready() {
  local preference="${OPENTULPA_OPEN_BROWSER:-auto}"
  local url="http://127.0.0.1:${PORT:-8000}/" opener=""
  local_server_bootstrap_enabled || return 0
  if is_falsey "${preference}"; then
    return 0
  fi
  if [[ "${preference}" == "auto" ]] && ! is_interactive; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "open ${url} after the server is healthy"
    return 0
  fi
  command -v curl >/dev/null 2>&1 || {
    log "Open ${url} to administer OpenTulpa."
    return 0
  }
  case "$(uname -s)" in
    Darwin) command -v open >/dev/null 2>&1 && opener="open" ;;
    Linux) command -v xdg-open >/dev/null 2>&1 && opener="xdg-open" ;;
  esac
  if [[ -z "${opener}" ]]; then
    log "Open ${url} to administer OpenTulpa."
    return 0
  fi
  (
    local attempt
    for attempt in $(seq 1 480); do
      if curl -fsS "http://127.0.0.1:${PORT:-8000}/healthz" >/dev/null 2>&1 \
        && { [[ "${SERVE_MODE}" == "1" ]] || curl -fsS "http://127.0.0.1:${PORT:-8000}/agent/healthz" >/dev/null 2>&1; }; then
        "${opener}" "${url}" >/dev/null 2>&1 || true
        exit 0
      fi
      sleep 0.25
    done
  ) &
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

container_engine_is_rootless() {
  local engine="$1" name output
  name="$(basename "${engine}")"
  case "${name}" in
    docker)
      output="$("${engine}" info --format '{{json .SecurityOptions}}' 2>/dev/null || true)"
      output="$(printf '%s' "${output}" | tr '[:upper:]' '[:lower:]')"
      [[ "${output}" == *rootless* ]]
      ;;
    podman)
      output="$("${engine}" info --format '{{.Host.Security.Rootless}}' 2>/dev/null || true)"
      output="$(printf '%s' "${output}" | tr '[:upper:]' '[:lower:]')"
      [[ "${output}" == "true" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

docker_uses_desktop_vm() {
  local engine="$1" context identity
  [[ "$(uname -s)" == "Darwin" && "$(basename "${engine}")" == "docker" ]] || return 1
  context="$("${engine}" context show 2>/dev/null || true)"
  identity="$("${engine}" info --format '{{.OperatingSystem}}|{{.Name}}' 2>/dev/null || true)"
  case "${context}|${identity}" in
    orbstack\|OrbStack\|orbstack|desktop-linux\|Docker\ Desktop\|docker-desktop) return 0 ;;
    *) return 1 ;;
  esac
}

container_engine_is_safe_for_direct() {
  local engine="$1"
  if container_engine_is_rootless "${engine}"; then
    return 0
  fi
  docker_uses_desktop_vm "${engine}"
}

configure_container_engine() {
  local runtime="$1" requested candidate
  requested="${OPENTULPA_CONTAINER_CLI:-}"
  DIRECT_ENGINE_AVAILABLE=0

  if [[ "${DRY_RUN}" == "1" ]]; then
    export OPENTULPA_CONTAINER_CLI="${requested:-docker}"
    DIRECT_ENGINE_AVAILABLE=1
    return 0
  fi

  if [[ -n "${requested}" ]]; then
    if command -v "${requested}" >/dev/null 2>&1; then
      if container_engine_is_rootless "${requested}"; then
        DIRECT_ENGINE_AVAILABLE=1
        return 0
      fi
      if [[ "${runtime}" != "managed" ]] && docker_uses_desktop_vm "${requested}"; then
        export OPENTULPA_ALLOW_DESKTOP_VM=1
        DIRECT_ENGINE_AVAILABLE=1
        log "using Docker inside its recognized macOS desktop VM for tenant commands."
        return 0
      fi
    fi
    if [[ "${runtime}" == "managed" && "${MODE}" != "doctor" ]]; then
      die "managed mode requires a running rootless Docker or Podman engine"
    fi
    warn "${requested} is unavailable or lacks required isolation; chat will start but sandbox shell commands will be unavailable (tenant workspace only; source evolution uses the stable host)."
    return 0
  fi

  for candidate in podman docker; do
    command -v "${candidate}" >/dev/null 2>&1 || continue
    if container_engine_is_rootless "${candidate}"; then
      export OPENTULPA_CONTAINER_CLI="${candidate}"
      DIRECT_ENGINE_AVAILABLE=1
      log "using rootless ${candidate} for tenant commands."
      return 0
    fi
    if [[ "${runtime}" != "managed" ]] && docker_uses_desktop_vm "${candidate}"; then
      export OPENTULPA_CONTAINER_CLI="${candidate}"
      export OPENTULPA_ALLOW_DESKTOP_VM=1
      DIRECT_ENGINE_AVAILABLE=1
      log "using Docker inside its recognized macOS desktop VM for tenant commands."
      return 0
    fi
  done

  export OPENTULPA_CONTAINER_CLI="docker"
  if [[ "${runtime}" == "managed" && "${MODE}" != "doctor" ]]; then
    die "managed mode requires a running rootless Docker or Podman engine"
  fi
  warn "no isolated OCI engine was found; chat will start but sandbox shell commands will be unavailable (tenant workspace only; source evolution uses the stable host)."
}

run_doctor() {
  local runtime="$1"
  local failures=0

  doctor_check "uv is available" "$(command -v uv >/dev/null 2>&1 && echo 1 || echo 0)" "curl -LsSf https://astral.sh/uv/install.sh | sh" || failures=$((failures + 1))
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    echo "[doctor] ok: .env exists"
  else
    echo "[doctor] info: .env is missing; relying on process environment variables"
  fi
  load_dotenv
  doctor_check "OPENAI_COMPATIBLE_API_KEY is set" "$(env_is_set "OPENAI_COMPATIBLE_API_KEY" && echo 1 || echo 0)" "set OPENAI_COMPATIBLE_API_KEY in .env" || failures=$((failures + 1))
  if [[ "${runtime}" == "local" ]] || server_telegram_enabled; then
    doctor_check "TELEGRAM_BOT_TOKEN is set" "$(env_is_set "TELEGRAM_BOT_TOKEN" && echo 1 || echo 0)" "set TELEGRAM_BOT_TOKEN in .env" || failures=$((failures + 1))
    doctor_check "Telegram allowlist is set" "$(telegram_allowlist_is_set && echo 1 || echo 0)" "set TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS in .env" || failures=$((failures + 1))
  else
    echo "[doctor] info: ${runtime} Telegram disabled; skipping Telegram token and allowlist checks"
  fi
  if env_is_set "COMPOSIO_API_KEY"; then
    echo "[doctor] ok: COMPOSIO_API_KEY is set"
  else
    echo "[doctor] warn: COMPOSIO_API_KEY is not set; connector integrations such as Google Sheets and Instagram will be unavailable"
  fi
  emit_model_config_notice
  check_model_catalog
  if [[ "${runtime}" == "server" ]]; then
    doctor_check "OPENTULPA_OWNER_TOKEN is set" "$(env_is_set "OPENTULPA_OWNER_TOKEN" && echo 1 || echo 0)" "set OPENTULPA_OWNER_TOKEN for Agent API access" || failures=$((failures + 1))
    if server_telegram_enabled; then
      doctor_check "TELEGRAM_WEBHOOK_SECRET is set" "$(env_is_set "TELEGRAM_WEBHOOK_SECRET" && echo 1 || echo 0)" "set a stable TELEGRAM_WEBHOOK_SECRET in .env" || failures=$((failures + 1))
      doctor_check "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN is set" "$(public_base_url_is_set && echo 1 || echo 0)" "set PUBLIC_BASE_URL to the public HTTPS URL, or rely on Railway's RAILWAY_PUBLIC_DOMAIN" || failures=$((failures + 1))
    else
      echo "[doctor] info: server Telegram disabled; skipping webhook URL/secret checks"
    fi
    doctor_check "OPENTULPA_DATA_ROOT is set" "$(env_is_set "OPENTULPA_DATA_ROOT" && echo 1 || echo 0)" "set OPENTULPA_DATA_ROOT=/app/opentulpa_data and mount persistent storage there" || failures=$((failures + 1))
    if env_is_set "OPENTULPA_DATA_ROOT"; then
      doctor_check "OPENTULPA_DATA_ROOT is writable" "$(mkdir -p "${OPENTULPA_DATA_ROOT}" 2>/dev/null && [[ -w "${OPENTULPA_DATA_ROOT}" ]] && echo 1 || echo 0)" "mount a writable persistent volume at OPENTULPA_DATA_ROOT" || failures=$((failures + 1))
    fi
  fi
  if [[ "${runtime}" == "managed" ]]; then
    local engine runtime_image sandbox_image tenant_image network recovery_token ingress_token
    engine="${OPENTULPA_CONTAINER_CLI:-docker}"
    runtime_image="${OPENTULPA_RELEASE_BASE_IMAGE:-opentulpa-runtime-base:0.1.0}"
    sandbox_image="${EVOLUTION_SANDBOX_IMAGE:-opentulpa-evolution:0.1.0}"
    tenant_image="$(config_value "SANDBOX_IMAGE" "sandbox_image")"
    tenant_image="${tenant_image:-opentulpa-tenant-sandbox:0.1.0}"
    network="${OPENTULPA_RELEASE_EGRESS_NETWORK:-}"
    recovery_token="${OPENTULPA_RECOVERY_TOKEN:-}"
    ingress_token="${OPENTULPA_INGRESS_TOKEN:-}"
    doctor_check "OPENTULPA_OWNER_TOKEN is set" "$(env_is_set "OPENTULPA_OWNER_TOKEN" && echo 1 || echo 0)" "set OPENTULPA_OWNER_TOKEN" || failures=$((failures + 1))
    doctor_check "OPENTULPA_RECOVERY_TOKEN is at least 32 characters" "$([[ ${#recovery_token} -ge 32 ]] && echo 1 || echo 0)" "set a random OPENTULPA_RECOVERY_TOKEN" || failures=$((failures + 1))
    doctor_check "OPENTULPA_INGRESS_TOKEN is at least 32 characters" "$([[ ${#ingress_token} -ge 32 ]] && echo 1 || echo 0)" "set a random OPENTULPA_INGRESS_TOKEN" || failures=$((failures + 1))
    if server_telegram_enabled; then
      doctor_check "TELEGRAM_WEBHOOK_SECRET is set" "$(env_is_set "TELEGRAM_WEBHOOK_SECRET" && echo 1 || echo 0)" "set a stable TELEGRAM_WEBHOOK_SECRET" || failures=$((failures + 1))
      doctor_check "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN is set" "$(public_base_url_is_set && echo 1 || echo 0)" "set the public HTTPS gateway URL" || failures=$((failures + 1))
    fi
    doctor_check "canonical Git checkout is available" "$([[ -d "${REPO_ROOT}/.git" ]] && echo 1 || echo 0)" "run managed mode from a canonical Git checkout" || failures=$((failures + 1))
    doctor_check "${engine} is available" "$(command -v "${engine}" >/dev/null 2>&1 && echo 1 || echo 0)" "install a rootless Docker or Podman engine" || failures=$((failures + 1))
    if command -v "${engine}" >/dev/null 2>&1; then
      doctor_check "${engine} is rootless" "$(container_engine_is_rootless "${engine}" && echo 1 || echo 0)" "configure a rootless Docker or Podman engine" || failures=$((failures + 1))
      doctor_check "trusted runtime base image exists" "$("${engine}" image inspect "${runtime_image}" >/dev/null 2>&1 && echo 1 || echo 0)" "run ./start.sh install managed" || failures=$((failures + 1))
      doctor_check "evolution sandbox image exists" "$("${engine}" image inspect "${sandbox_image}" >/dev/null 2>&1 && echo 1 || echo 0)" "run ./start.sh install managed" || failures=$((failures + 1))
      doctor_check "tenant sandbox image exists" "$("${engine}" image inspect "${tenant_image}" >/dev/null 2>&1 && echo 1 || echo 0)" "run ./start.sh install managed" || failures=$((failures + 1))
      doctor_check "restricted release network exists" "$([[ -n "${network}" ]] && "${engine}" network inspect "${network}" >/dev/null 2>&1 && echo 1 || echo 0)" "create and restrict OPENTULPA_RELEASE_EGRESS_NETWORK" || failures=$((failures + 1))
    fi
  fi
  if [[ "${runtime}" != "managed" ]]; then
    local direct_engine direct_tenant_image
    direct_engine="${OPENTULPA_CONTAINER_CLI:-docker}"
    direct_tenant_image="$(config_value "SANDBOX_IMAGE" "sandbox_image")"
    direct_tenant_image="${direct_tenant_image:-opentulpa-tenant-sandbox:0.1.0}"
    doctor_check "${direct_engine} is available for tenant commands" "$(command -v "${direct_engine}" >/dev/null 2>&1 && echo 1 || echo 0)" "install a rootless Docker or Podman engine" || failures=$((failures + 1))
    if command -v "${direct_engine}" >/dev/null 2>&1; then
      doctor_check "${direct_engine} has direct-mode isolation" "$(container_engine_is_safe_for_direct "${direct_engine}" && echo 1 || echo 0)" "configure rootless Docker/Podman or use Docker Desktop/OrbStack on macOS" || failures=$((failures + 1))
      doctor_check "tenant sandbox image exists" "$("${direct_engine}" image inspect "${direct_tenant_image}" >/dev/null 2>&1 && echo 1 || echo 0)" "run ./start.sh install ${runtime}" || failures=$((failures + 1))
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
  default_to_serve "$@"
  parse_args "$@"
  load_dotenv
  configure_serve
  RUNTIME_MODE="$(normalize_runtime_mode "${RUNTIME_MODE}")"
  configure_python_extras

  local runtime
  runtime="$(resolve_runtime_mode)"

  if [[ "${MODE}" == "doctor" ]]; then
    configure_container_engine "${runtime}"
    run_doctor "${runtime}"
    return 0
  fi

  if [[ "${MODE}" != "install" ]]; then
    configure_local_server_defaults "${runtime}"
    ensure_required_env "${runtime}"
  fi
  configure_container_engine "${runtime}"

  if [[ "${MODE}" != "run" ]]; then
    install_python_deps
    if [[ "${runtime}" == "managed" ]]; then
      install_managed_images
    else
      install_tenant_sandbox_image
    fi
    if [[ "${runtime}" == "local" ]]; then
      ensure_cloudflared
    fi
  fi

  if [[ "${MODE}" == "install" ]]; then
    return 0
  fi

  if [[ "${runtime}" == "managed" ]]; then
    log "running immutable bootstrap with managed OCI releases."
    run_bootstrap
    return 0
  fi

  if [[ "${runtime}" == "server" ]]; then
    log "running server mode."
    stop_existing_server
    open_local_web_when_ready
    if [[ "${SERVE_MODE}" == "1" ]]; then
      run_host
    else
      run_app
    fi
    return 0
  fi

  log "running local Telegram mode."
  if [[ "${SERVE_MODE}" == "1" ]]; then
    open_local_web_when_ready
  fi
  run_manager
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
