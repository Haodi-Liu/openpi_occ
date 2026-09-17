# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Proxy args: apt uses http_proxy/no_proxy; uv/git use https_proxy.
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG no_proxy
ARG NO_PROXY
ARG UV_HTTP_TIMEOUT=300

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV no_proxy=${no_proxy}
ENV NO_PROXY=${NO_PROXY}
ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}

# Needed because LeRobot uses git-lfs. `libgl1` is required by `opencv-python`
# during DataLoader worker startup (e.g. compute_norm_stats / training).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    ca-certificates \
    git \
    git-lfs \
    libgl1 \
    linux-headers-generic \
    build-essential \
    clang && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev && \
    git config --global http.version HTTP/1.1 && \
    git config --global http.postBuffer 524288000 && \
    git config --global http.maxRequests 1 && \
    git config --global core.compression 0 && \
    rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_PYTHON_DOWNLOADS=never

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python /usr/bin/python3.11 $UV_PROJECT_ENVIRONMENT
COPY uv.lock pyproject.toml /app/
COPY packages/openpi-client/pyproject.toml /app/packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src /app/packages/openpi-client/src
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync -v --frozen --no-install-project --no-dev

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
