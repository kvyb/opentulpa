FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /opt/opentulpa-evolution

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH="/opt/opentulpa-evolution/.venv/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Build this reviewed tool image once during installation. Candidate evaluation
# then runs with no network and a read-only source mount. Installing the trusted
# project gives contract checks a host-owned validator even when Python runs in
# isolated mode and cannot import from the candidate checkout.
RUN uv sync --frozen --all-extras --dev

WORKDIR /workspace
