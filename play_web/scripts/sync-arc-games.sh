#!/usr/bin/env bash
# Download local ARC-AGI-3 environment files into play_web/environment_files/.
# After adding games, regenerate onboarding preview icons:
#   bash play_web/scripts/generate-arc-game-previews.sh
set -euo pipefail

PLAY_WEB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PLAY_WEB_ROOT/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
DEFAULT_VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
PLAY_WEB_CONDA_ENV="${PLAY_WEB_CONDA_ENV:-oracle_craftext}"

activate_conda_env() {
  if [[ -n "${PLAY_WEB_PYTHON:-}" ]]; then
    return 0
  fi
  if [[ -x "$DEFAULT_VENV_PYTHON" ]]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    return 0
  fi
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    return 0
  fi
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$PLAY_WEB_CONDA_ENV" ]]; then
    conda activate "$PLAY_WEB_CONDA_ENV" 2>/dev/null || true
  fi
}

resolve_python() {
  if [[ -n "${PLAY_WEB_PYTHON:-}" ]]; then
    echo "$PLAY_WEB_PYTHON"
    return 0
  fi
  if [[ -x "$DEFAULT_VENV_PYTHON" ]]; then
    echo "$DEFAULT_VENV_PYTHON"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "${CONDA_PREFIX}/bin/python"
    return 0
  fi
  local base candidate
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/miniconda3" "/opt/anaconda3" "/opt/homebrew/Caskroom/miniconda/base"; do
    candidate="$base/envs/${PLAY_WEB_CONDA_ENV}/bin/python"
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  command -v python3 2>/dev/null || echo python3
}

if [[ -f "$ENV_FILE" ]]; then
  chmod 600 "$ENV_FILE" 2>/dev/null || true
  # shellcheck disable=SC1090
  set -a && source "$ENV_FILE" && set +a
fi

activate_conda_env
PLAY_PYTHON="$(resolve_python)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

cd "$PLAY_WEB_ROOT"
exec "$PLAY_PYTHON" "$PLAY_WEB_ROOT/scripts/sync_arc_games.py" "$@"
