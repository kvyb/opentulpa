FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=120

COPY pyproject.toml uv.lock README.md opentulpa.config.yaml /app/
COPY src /app/src
COPY scripts /app/scripts
COPY docs /app/docs

RUN mkdir -p /app/tulpa_stuff \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && printf '%s\n' '"""Agent-created integrations and skills."""' > /app/tulpa_stuff/__init__.py \
    && uv sync --frozen --no-dev \
    && uv run playwright install --with-deps chromium

COPY start.sh /app/start.sh

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["./start.sh", "run", "server"]
