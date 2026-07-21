FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="OpenTulpa tenant sandbox"
LABEL org.opencontainers.image.description="Reviewed networked workspace tool image"
LABEL org.opentulpa.sandbox.contract="tenant-workspace-v1"

ENV HOME=/tmp
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Keep the general-purpose workspace image small while retaining the tools an
# agent needs to inspect, modify, and fetch dependencies for ordinary source trees.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl git ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
