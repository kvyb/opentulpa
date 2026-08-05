ARG OPENTULPA_RUNTIME_IMAGE=opentulpa-rootful-e2e

FROM docker:28.3.3-cli@sha256:0135662b510037ea581d99c2e5929c5e01185139c0b86986a418bd4da0b98a44 AS docker-cli

FROM ${OPENTULPA_RUNTIME_IMAGE}

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

RUN /usr/local/bin/docker --version
