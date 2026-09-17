#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/setup_host_layout.sh" >/dev/null

GPU_IDS="${1:-${OPENPI_TRAIN_GPUS:-${OPENPI_TRAIN_GPU:-}}}"
if [[ -z "${GPU_IDS}" ]]; then
  echo "Usage: $0 <gpu-ids>" >&2
  echo "Example: $0 0,1,2,4" >&2
  echo "Or set OPENPI_TRAIN_GPUS in the current shell before running this script." >&2
  exit 1
fi

if ! docker image inspect openpi_server >/dev/null 2>&1; then
  echo "Local image openpi_server not found. Build it first:" >&2
  echo "  docker build -t openpi_server -f ${OPENPI_ROOT}/scripts/docker/serve_policy.Dockerfile ${OPENPI_ROOT}" >&2
  exit 1
fi

CONTAINER_NAME="${OPENPI_CONTAINER_NAME:-openpi_rlbench_shell}"
GPU_REQUEST="\"device=${GPU_IDS}\""

EXTRA_MOUNTS=()
for path in "$OPENPI_DATA_HOME" "$OPENPI_LEROBOT_HOME" "$OPENPI_RLBENCH_EXPORT" \
  "$OPENPI_RUNTIME_DATA" "$OPENPI_OCC_EVAL_LOGDIR" "$OPENPI_RLBENCH_CLOSEDLOOP_ROOT" \
  "$OPENPI_CHECKPOINT_BASE_DIR"; do
  EXTRA_MOUNTS+=(-v "${path}:${path}")
done
for path in "$OPENPI_RLBENCH_TRAIN_ROOT" "$OPENPI_RLBENCH_EVAL_ROOT"; do
  if [[ -n "$path" ]]; then
    EXTRA_MOUNTS+=(-v "${path}:${path}:ro")
  fi
done
if [[ -n "$OCC_TIMING_FILE" ]]; then
  EXTRA_MOUNTS+=(-v "${OCC_TIMING_FILE}:/etc/openpi/OPENPI_POLICY.yaml:ro")
  EXTRA_MOUNTS+=(-e OCC_TIMING_FILE=/etc/openpi/OPENPI_POLICY.yaml)
fi

exec docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus "${GPU_REQUEST}" \
  --network host \
  --shm-size=16g \
  --workdir "${OPENPI_ROOT}" \
  -v "${OPENPI_ROOT}:${OPENPI_ROOT}" \
  -v "${OPENPI_DOCKER_HOME}:${OPENPI_DOCKER_HOME}" \
  -v "${OPENPI_ROOT_CACHE}:/root/.cache" \
  "${EXTRA_MOUNTS[@]}" \
  -e OPENPI_ROOT="${OPENPI_ROOT}" \
  -e OPENPI_DOCKER_HOME="${OPENPI_DOCKER_HOME}" \
  -e OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
  -e OPENPI_ROOT_CACHE="${OPENPI_ROOT_CACHE}" \
  -e OPENPI_LEROBOT_HOME="${OPENPI_LEROBOT_HOME}" \
  -e OPENPI_RLBENCH_EXPORT="${OPENPI_RLBENCH_EXPORT}" \
  -e OPENPI_RUNTIME_DATA="${OPENPI_RUNTIME_DATA}" \
  -e OPENPI_OCC_EVAL_LOGDIR="${OPENPI_OCC_EVAL_LOGDIR}" \
  -e OPENPI_CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR}" \
  -e OPENPI_RLBENCH_TRAIN_ROOT="${OPENPI_RLBENCH_TRAIN_ROOT}" \
  -e OPENPI_RLBENCH_EVAL_ROOT="${OPENPI_RLBENCH_EVAL_ROOT}" \
  -e OPENPI_RLBENCH_CLOSEDLOOP_ROOT="${OPENPI_RLBENCH_CLOSEDLOOP_ROOT}" \
  -e OPENPI_CONTAINER_NAME="${CONTAINER_NAME}" \
  -e OPENPI_RUN_ASSETS="${OPENPI_RUN_ASSETS:-}" \
  -e OPENPI_RUN_CHECKPOINTS="${OPENPI_RUN_CHECKPOINTS:-}" \
  -e OPENPI_SERVE_PORT="${OPENPI_SERVE_PORT:-}" \
  -e OPENPI_SERVE_STEP="${OPENPI_SERVE_STEP:-}" \
  -e RLBENCH_TASK="${RLBENCH_TASK:-}" \
  -e RLBENCH_SINGLE_EXPORT="${RLBENCH_SINGLE_EXPORT:-}" \
  -e RLBENCH_TRAIN_REPO_ID="${RLBENCH_TRAIN_REPO_ID:-}" \
  -e RLBENCH_VAL_REPO_ID="${RLBENCH_VAL_REPO_ID:-}" \
  -e RLBENCH_EXP_NAME="${RLBENCH_EXP_NAME:-}" \
  -e OPENPI_COMPAT_ASSET_ID="${OPENPI_COMPAT_ASSET_ID:-}" \
  -e HF_LEROBOT_HOME="${OPENPI_LEROBOT_HOME}" \
  -e IS_DOCKER=true \
  openpi_server \
  /bin/bash
