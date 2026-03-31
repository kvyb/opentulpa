FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY skills /app/skills
COPY docs /app/docs

RUN mkdir -p /app/tulpa_stuff \
    && printf '%s\n' '"""Agent-created integrations and skills."""' > /app/tulpa_stuff/__init__.py \
    && uv sync --frozen --no-dev \
    && uv run playwright install --with-deps chromium

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "opentulpa"]
