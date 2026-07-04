FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 安装基础系统工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    vim \
    ca-certificates \
    gnupg \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 安装 Node.js 20，用于 Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 安装 Claude Code
# 如果后续官方安装方式变化，以 Anthropic 官方文档为准
RUN npm install -g @anthropic-ai/claude-code

# 安装 Python 包管理辅助工具
RUN pip install --upgrade pip uv

WORKDIR /workspace

CMD ["/bin/bash"]
