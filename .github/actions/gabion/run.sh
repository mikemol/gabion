#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python is required to run gabion" >&2
  exit 2
fi

python -m pip install --upgrade pip >/dev/null
python -m pip install "${GABION_VERSION}"

cmd=("${GABION_COMMAND}")
if [[ -n "${GABION_ROOT}" ]]; then
  cmd+=("--root" "${GABION_ROOT}")
fi
if [[ -n "${GABION_CONFIG}" ]]; then
  cmd+=("--config" "${GABION_CONFIG}")
fi
if [[ -n "${GABION_REPORT}" ]]; then
  cmd+=("--report" "${GABION_REPORT}")
fi
if [[ -n "${GABION_ARGS}" ]]; then
  # shellcheck disable=SC2206
  cmd+=(${GABION_ARGS})
fi

python -m gabion "${cmd[@]}"
