#!/usr/bin/env bash
# Local dev: API + static UI on one port. Keep this terminal open.
set -euo pipefail

PLAY_WEB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PLAY_WEB_ROOT/.." && pwd)"
PORT="${PLAY_WEB_API_PORT:-8001}"
HOST="${PLAY_WEB_HOST:-127.0.0.1}"
DEFAULT_VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
CONDA_ENV="${PLAY_WEB_CONDA_ENV:-oracle_craftext}"

resolve_python() {
  if [[ -n "${PLAY_WEB_PYTHON:-}" ]]; then
    echo "$PLAY_WEB_PYTHON"
    return 0
  fi
  if [[ -x "$DEFAULT_VENV_PYTHON" ]]; then
    echo "$DEFAULT_VENV_PYTHON"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_base candidate
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$conda_base/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
      if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
        return 0
      fi
    fi
    for candidate in \
      "$HOME/miniconda3/envs/${CONDA_ENV}/bin/python" \
      "$HOME/anaconda3/envs/${CONDA_ENV}/bin/python"; do
      if [[ -x "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done
  fi
  command -v python3 2>/dev/null || echo python3
}

python_env_label() {
  local play_python="$1"
  if [[ "$play_python" == "$DEFAULT_VENV_PYTHON" ]]; then
    echo "venv (.venv)"
    return 0
  fi
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "$CONDA_DEFAULT_ENV"
    return 0
  fi
  echo "system"
}

PLAY_PYTHON="$(resolve_python)"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PLAY_WEB_PERSIST_KEYS_TO_ENV="${PLAY_WEB_PERSIST_KEYS_TO_ENV:-false}"

cd "$PLAY_WEB_ROOT"
URL="http://${HOST}:${PORT}/trajectory_dashboard.html"

echo ""
echo "TalkingHeads local dev"
echo "  env: $(python_env_label "$PLAY_PYTHON") ($PLAY_PYTHON)"
echo "  dashboard: ${URL}"
echo "  API docs:  http://${HOST}:${PORT}/docs"
echo ""
echo "Press Ctrl+C to stop."
echo ""

if command -v open >/dev/null 2>&1; then
  (sleep 2 && open "$URL") &
fi

exec "$PLAY_PYTHON" -m uvicorn server:app --host "$HOST" --port "$PORT"
