#!/usr/bin/env bash
# Start/stop TalkingHeads play_web (API + static UI). Idempotent.
set -euo pipefail

PLAY_WEB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PLAY_WEB_ROOT/.." && pwd)"
API_PORT="${PLAY_WEB_API_PORT:-8001}"
WEB_PORT="${PLAY_WEB_WEB_PORT:-8089}"
HOST="${PLAY_WEB_HOST:-127.0.0.1}"
STARTUP_WAIT_SECS="${PLAY_WEB_STARTUP_WAIT_SECS:-120}"
RUN_DIR="$PLAY_WEB_ROOT/.run"
LOG_DIR="$RUN_DIR/logs"
API_PID="$RUN_DIR/play-api.pid"
WEB_PID="$RUN_DIR/play-web.pid"
ENV_FILE="$REPO_ROOT/.env"
DEFAULT_VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
PLAY_WEB_CONDA_ENV="${PLAY_WEB_CONDA_ENV:-oracle_craftext}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

python_env_label() {
  if [[ "$PLAY_PYTHON" == "$DEFAULT_VENV_PYTHON" ]]; then
    echo "venv (.venv)"
    return 0
  fi
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "$CONDA_DEFAULT_ENV"
    return 0
  fi
  echo "system"
}

activate_conda_env() {
  if [[ -n "${PLAY_WEB_PYTHON:-}" ]]; then
    return 0
  fi
  if [[ -x "$DEFAULT_VENV_PYTHON" ]]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "warning: conda not found; using PATH python (expected env: ${PLAY_WEB_CONDA_ENV})" >&2
    return 0
  fi
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    echo "warning: conda.sh not found; using PATH python (expected env: ${PLAY_WEB_CONDA_ENV})" >&2
    return 0
  fi
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$PLAY_WEB_CONDA_ENV" ]]; then
    if ! conda activate "$PLAY_WEB_CONDA_ENV"; then
      echo "error: failed to activate conda env '${PLAY_WEB_CONDA_ENV}'" >&2
      exit 1
    fi
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
  # Fallback: locate the expected conda env python directly, even when the
  # `conda` command is not on PATH (e.g. non-login shells).
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

export PLAY_WEB_PERSIST_KEYS_TO_ENV="${PLAY_WEB_PERSIST_KEYS_TO_ENV:-false}"
export PLAY_WEB_SESSION_TTL_SECONDS="${PLAY_WEB_SESSION_TTL_SECONDS:-7200}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

port_pids() {
  lsof -ti:"$1" 2>/dev/null || true
}

port_busy() {
  [[ -n "$(port_pids "$1")" ]]
}

wait_port_free() {
  local port="$1"
  local attempts="${2:-30}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if ! port_busy "$port"; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

api_ok() {
  curl -sf "http://${HOST}:${API_PORT}/api/campaign_benchmark" >/dev/null 2>&1
}

web_ok() {
  curl -sf -o /dev/null "http://${HOST}:${WEB_PORT}/client/index.html" >/dev/null 2>&1
}

health_ok() {
  api_ok && web_ok
}

kill_pids() {
  local pids="$1"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 0.5
  pids="$(port_pids "${2:-}")"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 0.2
  fi
}

stop_pid_file() {
  local pid_file="$1"
  local port="$2"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 0.5
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
  if port_busy "$port"; then
    kill_pids "$(port_pids "$port")" "$port"
  fi
  wait_port_free "$port" 40 || {
    echo "warning: port ${port} is still busy after stop" >&2
    return 1
  }
}

pid_alive() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

log_tail() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "--- tail ${file} ---" >&2
    tail -n 20 "$file" >&2 || true
  fi
}

start_api() {
  if api_ok && pid_alive "$API_PID"; then
    return 0
  fi
  stop_pid_file "$API_PID" "$API_PORT"
  cd "$PLAY_WEB_ROOT"
  {
    echo ""
    echo "===== API start $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "python: $PLAY_PYTHON ($("$PLAY_PYTHON" --version 2>&1))"
    echo "env: $(python_env_label)"
  } >>"$LOG_DIR/api.log"
  nohup "$PLAY_PYTHON" -m uvicorn server:app --host "$HOST" --port "$API_PORT" \
    >>"$LOG_DIR/api.log" 2>&1 &
  echo $! >"$API_PID"
  sleep 0.3
  if ! kill -0 "$(cat "$API_PID")" 2>/dev/null; then
    echo "API process exited immediately. See $LOG_DIR/api.log" >&2
    log_tail "$LOG_DIR/api.log"
    return 1
  fi
}

start_web() {
  if web_ok && pid_alive "$WEB_PID"; then
    return 0
  fi
  stop_pid_file "$WEB_PID" "$WEB_PORT"
  cd "$PLAY_WEB_ROOT"
  {
    echo ""
    echo "===== WEB start $(date '+%Y-%m-%d %H:%M:%S') ====="
  } >>"$LOG_DIR/web.log"
  nohup "$PLAY_PYTHON" "$PLAY_WEB_ROOT/devserver.py" "$HOST" "$WEB_PORT" \
    >>"$LOG_DIR/web.log" 2>&1 &
  echo $! >"$WEB_PID"
  sleep 0.2
}

cmd_start() {
  if [[ "${PLAY_WEB_SKIP_ARC_SYNC:-false}" != "true" ]]; then
    echo "Syncing ARC-AGI-3 environment files..."
    if ! "$PLAY_WEB_ROOT/scripts/sync-arc-games.sh"; then
      echo "warning: ARC game sync failed; some ARC games may be unavailable offline" >&2
    fi
  fi
  start_web
  start_api || return 1

  local waited=0
  while (( waited < STARTUP_WAIT_SECS )); do
    if health_ok; then
      echo "play_web running"
      echo "  env: $(python_env_label) ($PLAY_PYTHON)"
      echo "  UI:  http://${HOST}:${WEB_PORT}/client/index.html"
      echo "  API: http://${HOST}:${API_PORT}/api/ (docs: /docs)"
      echo "  logs: $LOG_DIR"
      return 0
    fi
    if ! pid_alive "$API_PID"; then
      echo "API process died during startup. See $LOG_DIR/api.log" >&2
      log_tail "$LOG_DIR/api.log"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
    if (( waited % 10 == 0 )); then
      echo "waiting for play_web... (${waited}s / ${STARTUP_WAIT_SECS}s)" >&2
    fi
  done

  echo "play_web failed to become healthy within ${STARTUP_WAIT_SECS}s." >&2
  if ! api_ok; then
    echo "  API not responding on http://${HOST}:${API_PORT}/api/campaign_benchmark" >&2
    log_tail "$LOG_DIR/api.log"
  fi
  if ! web_ok; then
    echo "  Web not responding on http://${HOST}:${WEB_PORT}/client/index.html" >&2
    log_tail "$LOG_DIR/web.log"
  fi
  if pid_alive "$API_PID" || pid_alive "$WEB_PID"; then
    echo "  Processes are still running; the server may finish loading soon." >&2
    echo "  Try: $0 status" >&2
  fi
  return 1
}

cmd_stop() {
  stop_pid_file "$WEB_PID" "$WEB_PORT" || true
  stop_pid_file "$API_PID" "$API_PORT" || true
  rm -f "$API_PID" "$WEB_PID"
  echo "play_web stopped"
}

cmd_status() {
  local api_state="down"
  local web_state="down"
  if api_ok; then api_state="up"; fi
  if web_ok; then web_state="up"; fi
  echo "play_web: API ${api_state} (${API_PORT}), web ${web_state} (${WEB_PORT})"
  if [[ "$api_state" == "up" && "$web_state" == "up" ]]; then
    return 0
  fi
  return 1
}

cmd_restart() {
  cmd_stop
  sleep 1
  cmd_start
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  restart) cmd_restart ;;
  *)
    echo "Usage: $0 {start|stop|status|restart}"
    exit 1
    ;;
esac
