FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PIP_INDEX_URL= \
    PIP_TRUSTED_HOST= \
    UV_DEFAULT_INDEX= \
    UV_INDEX_URL=

RUN /usr/local/bin/uv --version \
    && /usr/local/bin/python3 -I -m pip --version
