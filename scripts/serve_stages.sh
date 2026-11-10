#!/usr/bin/env bash
set -euo pipefail
AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PI05_RUNTIME_PYTHON="${PI05_RUNTIME_PYTHON:-/home/user/lpy/pi05-server/.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AGENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$AGENT_ROOT/runtime/cache"
export OPENPI_DATA_HOME="$AGENT_ROOT/runtime/cache/openpi"
cd -- "$AGENT_ROOT"
exec "$PI05_RUNTIME_PYTHON" -B -m tomato_agent.stage_server \
  --contract-out "$AGENT_ROOT/runtime/pi05-stage-model.json" "$@"
