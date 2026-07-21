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
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Minimal OS packages needed to build Python wheels (netCDF4/h5py/scipy pull
# in compiled extensions) and to use git inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Pinned for reproducible builds: anyone building this image gets the same
# pip/uv, regardless of when or where they build it.
RUN pip install --no-cache-dir "pip==25.1.1" "uv==0.11.26"

WORKDIR /workspace
