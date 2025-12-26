FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装系统工具 + Python 3.12 (之前遇到的问题)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git curl wget vim htop build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 解决 Git 目录权限问题 (当前遇到的问题)
# 这里的 * 表示信任所有目录，开发环境这样设置最省心
RUN git config --global --add safe.directory '*'

WORKDIR /workspace
CMD ["/bin/bash"]