FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="OpenTulpa tenant sandbox"
LABEL org.opencontainers.image.description="Reviewed no-network workspace tool image"
LABEL org.opentulpa.sandbox.contract="tenant-workspace-v1"

ENV HOME=/tmp
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Keep the general-purpose workspace image small while retaining the tools an
# agent needs to inspect and modify ordinary source trees without a network.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
