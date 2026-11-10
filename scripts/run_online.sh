#!/usr/bin/env bash
set -euo pipefail
AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MONO_ROOT="${MONO_ROOT:-/home/user/lpy/li-agent-mono}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$AGENT_ROOT/runtime/cache"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
cd -- "$MONO_ROOT"
# Activate the existing installed environment without running pixi install/update.
export CONDA_PREFIX="$MONO_ROOT/.pixi/envs/default"
export PIXI_PROJECT_ROOT="$MONO_ROOT"
export PATH="$CONDA_PREFIX/bin:$PATH"
set +u
source "$CONDA_PREFIX/setup.bash"
source "$MONO_ROOT/ros2/activate.bash"
set -u
export COLCON_LOG_PATH="$AGENT_ROOT/runtime/colcon-log"
export ROS_LOG_DIR="$AGENT_ROOT/runtime/ros-log"
export PYTHONPATH="$AGENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$AGENT_ROOT"
exec "$CONDA_PREFIX/bin/python" -B -m tomato_agent.online.runner "$@"
