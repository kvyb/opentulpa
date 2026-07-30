FROM node:22-bookworm-slim AS railway-sandbox-bridge

WORKDIR /bridge

COPY railway_sandbox_bridge/package.json railway_sandbox_bridge/package-lock.json /bridge/
RUN npm ci --omit=dev

COPY railway_sandbox_bridge/bridge.mjs /bridge/bridge.mjs

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY --from=railway-sandbox-bridge /usr/local/ /usr/local/
COPY --from=railway-sandbox-bridge /bridge /app/railway_sandbox_bridge

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git util-linux \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=120

ARG OPENTULPA_EXTRAS=""

COPY pyproject.toml uv.lock README.md opentulpa.config.yaml /app/
COPY src /app/src
COPY scripts /app/scripts
COPY docs /app/docs

RUN mkdir -p /app/tulpa_stuff \
    && printf '%s\n' '"""Agent-created integrations and skills."""' > /app/tulpa_stuff/__init__.py \
    && extras="$(printf '%s' "${OPENTULPA_EXTRAS}" | tr ',' ' ')" \
    && set -- \
    && for extra in ${extras}; do \
         case "${extra}" in \
           browser|integrations|documents|research|hosted-sandbox|bundled) ;; \
           *) printf '%s\n' "unsupported OPENTULPA_EXTRAS value: ${extra}" >&2; exit 2 ;; \
         esac; \
         set -- "$@" --extra "${extra}"; \
       done \
    && uv sync --frozen --no-dev --extra evaluation "$@"

COPY start.sh /app/start.sh
COPY . /opt/opentulpa-source

ENV HOST=0.0.0.0
ENV PORT=8000
ENV OPENTULPA_DATA_ROOT=/app/opentulpa_data

EXPOSE 8000

CMD ["./start.sh", "serve", "--run-only"]
