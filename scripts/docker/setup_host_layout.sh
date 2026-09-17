#!/usr/bin/env bash

set -euo pipefail

# Source this file to persist the environment variables in your current shell:
#   source scripts/docker/setup_host_layout.sh
#
# Running it as a normal script is also safe: it will create the directories
# and print the resolved paths, but the exports will not persist in the caller.

export OPENPI_ROOT="${OPENPI_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export OPENPI_DOCKER_HOME="${OPENPI_DOCKER_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/openpi_occ}"
# Optional OCC configuration directory; the simulator itself stays outside this container.
export OPENPI_OCC_ROOT="${OPENPI_OCC_ROOT:-}"
export OCC_TIMING_FILE="${OCC_TIMING_FILE:-}"
if [[ -z "${OCC_TIMING_FILE}" && -n "${OPENPI_OCC_ROOT}" ]]; then
  export OCC_TIMING_FILE="${OPENPI_OCC_ROOT}/conf/method/OPENPI_POLICY.yaml"
fi

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OPENPI_DOCKER_HOME/openpi_assets}"
export OPENPI_ROOT_CACHE="${OPENPI_ROOT_CACHE:-$OPENPI_DOCKER_HOME/root_cache}"
export OPENPI_LEROBOT_HOME="${OPENPI_LEROBOT_HOME:-$OPENPI_DOCKER_HOME/lerobot_home}"
export OPENPI_RLBENCH_EXPORT="${OPENPI_RLBENCH_EXPORT:-$OPENPI_DOCKER_HOME/rlbench_export}"
export OPENPI_RUNTIME_DATA="${OPENPI_RUNTIME_DATA:-$OPENPI_DOCKER_HOME/runtime_data}"
export OPENPI_OCC_EVAL_LOGDIR="${OPENPI_OCC_EVAL_LOGDIR:-$OPENPI_DOCKER_HOME/occ_eval_logs}"
export OPENPI_CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-$OPENPI_DOCKER_HOME/checkpoints}"

# Raw datasets are optional for inference. Set these explicitly when exporting data.
export OPENPI_RLBENCH_TRAIN_ROOT="${OPENPI_RLBENCH_TRAIN_ROOT:-}"
export OPENPI_RLBENCH_EVAL_ROOT="${OPENPI_RLBENCH_EVAL_ROOT:-}"
export OPENPI_RLBENCH_CLOSEDLOOP_ROOT="${OPENPI_RLBENCH_CLOSEDLOOP_ROOT:-$OPENPI_DOCKER_HOME/rlbench_closedloop_test}"

for path in "$OPENPI_ROOT" "$OPENPI_OCC_ROOT" "$OPENPI_RLBENCH_TRAIN_ROOT" "$OPENPI_RLBENCH_EVAL_ROOT"; do
  if [[ -n "$path" && ! -d "$path" ]]; then
    echo "Configured directory does not exist: $path" >&2
    exit 1
  fi
done

if [[ -n "$OCC_TIMING_FILE" && ! -f "$OCC_TIMING_FILE" ]]; then
  echo "OCC_TIMING_FILE does not exist: $OCC_TIMING_FILE" >&2
  exit 1
fi

mkdir -p \
  "$OPENPI_DATA_HOME" \
  "$OPENPI_ROOT_CACHE" \
  "$OPENPI_LEROBOT_HOME" \
  "$OPENPI_RLBENCH_EXPORT" \
  "$OPENPI_RUNTIME_DATA" \
  "$OPENPI_OCC_EVAL_LOGDIR" \
  "$OPENPI_CHECKPOINT_BASE_DIR" \
  "$OPENPI_RLBENCH_CLOSEDLOOP_ROOT"

cat <<EOF
Configured host-side layout:
  OPENPI_ROOT=$OPENPI_ROOT
  OPENPI_DOCKER_HOME=$OPENPI_DOCKER_HOME
  OPENPI_OCC_ROOT=$OPENPI_OCC_ROOT
  OCC_TIMING_FILE=$OCC_TIMING_FILE
  OPENPI_DATA_HOME=$OPENPI_DATA_HOME
  OPENPI_ROOT_CACHE=$OPENPI_ROOT_CACHE
  OPENPI_LEROBOT_HOME=$OPENPI_LEROBOT_HOME
  OPENPI_RLBENCH_EXPORT=$OPENPI_RLBENCH_EXPORT
  OPENPI_RUNTIME_DATA=$OPENPI_RUNTIME_DATA
  OPENPI_OCC_EVAL_LOGDIR=$OPENPI_OCC_EVAL_LOGDIR
  OPENPI_CHECKPOINT_BASE_DIR=$OPENPI_CHECKPOINT_BASE_DIR
  OPENPI_RLBENCH_TRAIN_ROOT=$OPENPI_RLBENCH_TRAIN_ROOT
  OPENPI_RLBENCH_EVAL_ROOT=$OPENPI_RLBENCH_EVAL_ROOT
  OPENPI_RLBENCH_CLOSEDLOOP_ROOT=$OPENPI_RLBENCH_CLOSEDLOOP_ROOT
EOF

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Note: this script was executed, not sourced. Run 'source scripts/docker/setup_host_layout.sh' if you also want the exports in your current shell."
fi
