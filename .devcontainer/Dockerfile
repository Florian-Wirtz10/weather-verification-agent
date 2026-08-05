# Python runtime for the agent/verification/report code.
#
# Node.js for Codex CLI and the non-root user are provisioned by
# devcontainer.json. This image remains a single-purpose, auditable Python
# application layer.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OFFICECLI_SKIP_UPDATE=1 \
    OFFICECLI_RESIDENT_FLUSH=each

# Minimal OS packages needed to build Python wheels (netCDF4/h5py/scipy pull
# in compiled extensions) and to use git inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        chromium \
        curl \
        fonts-noto-cjk \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# OfficeCLI is a self-contained binary. Pin both the release and checksums so
# Dev Container rebuilds are reproducible on Intel/AMD and Apple Silicon hosts.
ARG OFFICECLI_VERSION=1.0.143
RUN set -eu; \
    officecli_platform="$(dpkg --print-architecture)"; \
    case "${officecli_platform}" in \
        amd64) \
            officecli_arch="x64"; \
            officecli_sha256="6a29c598a789b57c92c03e560907d3f131a4bd0a068785b1d338a86fc31a58a7" \
            ;; \
        arm64) \
            officecli_arch="arm64"; \
            officecli_sha256="c50298e4698fcd1b15fe1a0f096405ad260b5c84d4440882582d0bba1e57bd49" \
            ;; \
        *) \
            echo "Unsupported architecture for OfficeCLI: ${officecli_platform}" >&2; \
            exit 1 \
            ;; \
    esac; \
    curl -fsSL \
        "https://github.com/iOfficeAI/OfficeCLI/releases/download/v${OFFICECLI_VERSION}/officecli-linux-${officecli_arch}" \
        -o /usr/local/bin/officecli; \
    echo "${officecli_sha256}  /usr/local/bin/officecli" | sha256sum -c -; \
    chmod 0755 /usr/local/bin/officecli; \
    officecli --version

# Pinned for reproducible builds: anyone building this image gets the same
# pip/uv, regardless of when or where they build it.
RUN pip install --no-cache-dir "pip==25.1.1" "uv==0.11.26"

WORKDIR /workspace
