#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

cd "${REPO_ROOT}"

# Load local .env into this shell so mode selection can use it.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -n "${PUBLIC_BASE_URL:-}" || -n "${RAILWAY_PUBLIC_DOMAIN:-}" ]]; then
  echo "[start] public base URL detected; running direct app mode."
  exec uv run python -m opentulpa "$@"
fi

echo "[start] no public base URL detected; running quick-tunnel manager mode."
exec uv run python scripts/manager.py "$@"
