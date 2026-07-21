#!/bin/sh
set -eu

REPOSITORY="${OPENTULPA_INSTALL_REPOSITORY:-https://github.com/kvyb/opentulpa.git}"
REF="${OPENTULPA_INSTALL_REF:-main}"
DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
SOURCE_ROOT="${OPENTULPA_INSTALL_SOURCE:-${DATA_HOME}/opentulpa/source}"

say() {
  printf '%s\n' "[opentulpa] $*"
}

if [ -z "${OPENTULPA_INSTALL_SOURCE:-}" ]; then
  command -v git >/dev/null 2>&1 || {
    printf '%s\n' "OpenTulpa requires git so it can maintain and improve its source." >&2
    exit 1
  }
  if [ -d "${SOURCE_ROOT}/.git" ]; then
    say "using the existing source at ${SOURCE_ROOT}"
  else
    [ ! -e "${SOURCE_ROOT}" ] || {
      printf '%s\n' "Install path exists but is not an OpenTulpa checkout: ${SOURCE_ROOT}" >&2
      exit 1
    }
    mkdir -p "$(dirname "${SOURCE_ROOT}")"
    say "downloading OpenTulpa"
    git clone --branch "${REF}" --single-branch "${REPOSITORY}" "${SOURCE_ROOT}"
  fi
elif [ ! -f "${SOURCE_ROOT}/pyproject.toml" ]; then
  printf '%s\n' "OPENTULPA_INSTALL_SOURCE is not an OpenTulpa source tree: ${SOURCE_ROOT}" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  command -v curl >/dev/null 2>&1 || {
    printf '%s\n' "OpenTulpa requires curl to install its Python runtime." >&2
    exit 1
  }
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="${HOME}/.local/bin/uv"
  [ -x "${UV_BIN}" ] || {
    printf '%s\n' "uv installation did not create ${UV_BIN}" >&2
    exit 1
  }
fi

say "installing the OpenTulpa command"
UV_PYTHON=3.12 "${UV_BIN}" sync --locked --no-dev --project "${SOURCE_ROOT}"
COMMAND_SOURCE="${SOURCE_ROOT}/.venv/bin/opentulpa"
[ -x "${COMMAND_SOURCE}" ] || {
  printf '%s\n' "OpenTulpa installation did not create ${COMMAND_SOURCE}" >&2
  exit 1
}
BIN_DIR="${OPENTULPA_BIN_DIR:-${HOME}/.local/bin}"
mkdir -p "${BIN_DIR}"
for command in opentulpa opentulpa-host opentulpa-bootstrap opentulpa-recovery opentulpa-migrate-deepagents; do
  source_command="${SOURCE_ROOT}/.venv/bin/${command}"
  target_command="${BIN_DIR}/${command}"
  [ -x "${source_command}" ] || continue
  if [ -e "${target_command}" ] && [ ! -L "${target_command}" ]; then
    printf '%s\n' "Refusing to replace existing command: ${target_command}" >&2
    exit 1
  fi
  ln -sfn "${source_command}" "${target_command}"
done

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *)
    LINKED=0
    OLD_IFS="${IFS}"
    IFS=:
    for candidate in ${PATH}; do
      if [ -d "${candidate}" ] && [ -w "${candidate}" ] \
        && [ ! -e "${candidate}/opentulpa" ] && [ ! -L "${candidate}/opentulpa" ]; then
        ln -s "${COMMAND_SOURCE}" "${candidate}/opentulpa"
        LINKED=1
        break
      fi
    done
    IFS="${OLD_IFS}"
    SHELL_RC="${HOME}/.profile"
    case "${SHELL:-}" in
      */zsh) SHELL_RC="${HOME}/.zshrc" ;;
      */bash) SHELL_RC="${HOME}/.bashrc" ;;
    esac
    PATH_LINE="export PATH=\"${BIN_DIR}:\$PATH\""
    if ! grep -F "${BIN_DIR}" "${SHELL_RC}" >/dev/null 2>&1; then
      printf '\n%s\n' "${PATH_LINE}" >> "${SHELL_RC}"
    fi
    say "added ${BIN_DIR} to PATH in ${SHELL_RC}"
    if [ "${LINKED}" -eq 0 ]; then
      say "open a new terminal once to refresh PATH"
    fi
    ;;
esac

printf '\nOpenTulpa installed. Start it with:\n\n  opentulpa\n\n'
