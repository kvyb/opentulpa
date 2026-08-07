FROM node:22-bookworm-slim AS railway-sandbox-bridge

WORKDIR /bridge
COPY railway_sandbox_bridge/package.json railway_sandbox_bridge/package-lock.json ./
RUN npm ci --omit=dev
COPY railway_sandbox_bridge/bridge.mjs ./bridge.mjs

FROM oven/bun:1.3.14 AS terminal-client

WORKDIR /tui
COPY clients/tui/package.json clients/tui/bun.lock ./
RUN bun install --frozen-lockfile
COPY clients/tui/build.ts clients/tui/tsconfig.json ./
COPY clients/tui/src ./src
RUN bun run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS controller-build

ARG OPENTULPA_EXTRAS=""
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=120
ENV UV_HTTP_RETRIES=10

WORKDIR /build
COPY pyproject.toml uv.lock ./

# Pysher 1.0.8 is sdist-only but required by Composio; keep every other
# dependency binary-only and seed setuptools for the later offline build.
RUN extras="$(printf '%s' "${OPENTULPA_EXTRAS}" | tr ',' ' ')" \
    && set -- --extra evaluation \
    && for extra in ${extras}; do \
         case "${extra}" in \
           browser|integrations|documents|research|hosted-sandbox|bundled) ;; \
           *) printf '%s\n' "unsupported OPENTULPA_EXTRAS value: ${extra}" >&2; exit 2 ;; \
         esac; \
         set -- "$@" --extra "${extra}"; \
       done \
    && uv export --frozen --no-dev --no-emit-project --no-header "$@" \
         --output-file /tmp/controller-requirements.txt \
    && uv pip install --system 'hatchling==1.27.0' \
    && uv venv --python /usr/local/bin/python3.12 /tmp/pip-download \
    && uv pip install --python /tmp/pip-download/bin/python 'pip==25.1.1' \
    && uv venv --python /usr/local/bin/python3.12 \
         /opt/opentulpa-install/controller/generations/image \
    && mkdir -p /opt/opentulpa-install/controller/generations/image/wheelhouse \
         /opt/opentulpa-install/controller/generations/image/wheels \
    && /tmp/pip-download/bin/pip download \
         --disable-pip-version-check \
         --retries 10 \
         --resume-retries 10 \
         --timeout 120 \
         --only-binary=:all: \
         --no-deps \
         --dest /opt/opentulpa-install/controller/generations/image/wheelhouse \
         'setuptools==80.9.0' \
    && /tmp/pip-download/bin/pip download \
         --disable-pip-version-check \
         --retries 10 \
         --resume-retries 10 \
         --timeout 120 \
         --require-hashes \
         --only-binary=:all: \
         --no-binary=pysher \
         --dest /opt/opentulpa-install/controller/generations/image/wheelhouse \
         --requirement /tmp/controller-requirements.txt \
    && cp /tmp/controller-requirements.txt \
         /opt/opentulpa-install/controller/generations/image/requirements.txt \
    && uv pip install \
         --python /opt/opentulpa-install/controller/generations/image/bin/python \
         --no-index \
         --find-links /opt/opentulpa-install/controller/generations/image/wheelhouse \
         --require-hashes \
         --requirement /tmp/controller-requirements.txt

COPY README.md opentulpa.config.yaml ./
COPY src ./src
COPY railway_sandbox_bridge ./railway_sandbox_bridge

RUN uv build --wheel --offline --no-build-isolation --out-dir /tmp/controller-wheel . \
    && cp /tmp/controller-wheel/*.whl \
         /opt/opentulpa-install/controller/generations/image/wheels/ \
    && uv pip install \
         --python /opt/opentulpa-install/controller/generations/image/bin/python \
         --no-index \
         --no-deps \
         /tmp/controller-wheel/*.whl \
    && /opt/opentulpa-install/controller/generations/image/bin/python -c \
         'import importlib.metadata, importlib.resources; d=importlib.metadata.distribution("opentulpa"); assert {"opentulpa", "opentulpa-host", "opentulpa-sandbox-worker", "opentulpa-migrate-deepagents"} <= {e.name for e in d.entry_points if e.group == "console_scripts"}; assert importlib.resources.files("opentulpa").joinpath("resources", "release_contract.json").is_file()'

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV OPENTULPA_DATA_ROOT=/app/opentulpa_data
ENV OPENTULPA_SOURCE_ROOT=/app/opentulpa_data/source
ENV EVOLUTION_SOURCE_REPOSITORY=https://github.com/kvyb/opentulpa.git
ENV OPENTULPA_INSTALL_REF=main
ENV OPENTULPA_INSTALL_ROOT=/opt/opentulpa-install
ENV OPENTULPA_SOURCE_SEED_ROOT=/opt/opentulpa-source
ENV OPENTULPA_SOURCE_SEED_PROVENANCE=/opt/opentulpa-install/source-seed-manifest.json
ENV OPENTULPA_TRUSTED_WHEELHOUSE=/opt/opentulpa-install/controller/generations/image/wheelhouse
ENV OPENTULPA_INSTALL_ASSETS_ROOT=/opt/opentulpa-install/controller/generations/image/assets
ENV OPENTULPA_UV_BIN=/usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap curl ffmpeg git openssh-client util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && if getent group 65532 >/dev/null; then \
         test "$(getent group 65532 | cut -d: -f1)" = opentulpa-runtime; \
       else \
         printf '%s\n' 'opentulpa-runtime:x:65532:' >> /etc/group; \
       fi \
    && if getent passwd 65532 >/dev/null; then \
         test "$(getent passwd 65532 | cut -d: -f1)" = opentulpa-runtime; \
       else \
         printf '%s\n' 'opentulpa-runtime:x:65532:65532:OpenTulpa Runtime:/tmp:/usr/sbin/nologin' >> /etc/passwd; \
       fi \
    && if getent group 65533 >/dev/null; then \
         test "$(getent group 65533 | cut -d: -f1)" = opentulpa-candidate; \
       else \
         printf '%s\n' 'opentulpa-candidate:x:65533:' >> /etc/group; \
       fi \
    && if getent passwd 65533 >/dev/null; then \
         test "$(getent passwd 65533 | cut -d: -f1)" = opentulpa-candidate; \
       else \
         printf '%s\n' 'opentulpa-candidate:x:65533:65533:OpenTulpa Candidate:/tmp:/usr/sbin/nologin' >> /etc/passwd; \
       fi \
    && test "$(getent passwd 65532 | cut -d: -f3)" = 65532 \
    && test "$(getent passwd 65533 | cut -d: -f3)" = 65533 \
    && test "$(getent passwd 65532 | cut -d: -f3)" != "$(getent passwd 65533 | cut -d: -f3)" \
    && test -x /usr/bin/bwrap \
    && test ! -L /usr/bin/bwrap \
    && test "$(stat -c %u /usr/bin/bwrap)" = 0 \
    && test -z "$(find /usr/bin/bwrap -perm /022 -print -quit)" \
    && test -x /usr/bin/prlimit \
    && test ! -L /usr/bin/prlimit \
    && test "$(stat -c %u /usr/bin/prlimit)" = 0 \
    && test -z "$(find /usr/bin/prlimit -perm /022 -print -quit)" \
    && test -x /usr/bin/setpriv \
    && test ! -L /usr/bin/setpriv \
    && test "$(stat -c %u /usr/bin/setpriv)" = 0 \
    && test -z "$(find /usr/bin/setpriv -perm /022 -print -quit)"

WORKDIR /app
RUN mkdir -p /app/opentulpa_data
COPY --from=controller-build /opt/opentulpa-install /opt/opentulpa-install
COPY --from=controller-build /usr/local/bin/uv /usr/local/bin/uv
RUN test -x /usr/local/bin/uv && test ! -L /usr/local/bin/uv && /usr/local/bin/uv --version
RUN mkdir -p \
    /opt/opentulpa-install/controller/generations/image/assets/railway_sandbox_bridge \
    /opt/opentulpa-install/controller/generations/image/assets/tui
COPY --from=railway-sandbox-bridge /usr/local/bin/node /usr/local/bin/node
COPY --from=railway-sandbox-bridge /bridge \
    /opt/opentulpa-install/controller/generations/image/assets/railway_sandbox_bridge
COPY --from=terminal-client /tui/dist \
    /opt/opentulpa-install/controller/generations/image/assets/tui
COPY . /opt/opentulpa-source
COPY opentulpa.config.yaml \
    /opt/opentulpa-install/controller/generations/image/assets/opentulpa.config.yaml
RUN /opt/opentulpa-install/controller/generations/image/bin/python -c \
         'import json, pathlib; from opentulpa.host.evolution_composition import _source_seed_sha256; root=pathlib.Path("/opt/opentulpa-source"); pathlib.Path("/opt/opentulpa-install/source-seed-manifest.json").write_text(json.dumps({"format_version":1,"source_seed_sha256":_source_seed_sha256(root)},sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")' \
    && : > /opt/opentulpa-install/controller/generations/image/COMPLETE \
    && chmod -R a-w /opt/opentulpa-install /opt/opentulpa-source \
    && chmod -R a+rX /opt/opentulpa-install/controller/generations/image/wheelhouse

EXPOSE 8000

CMD ["/opt/opentulpa-install/controller/generations/image/bin/opentulpa-host"]
