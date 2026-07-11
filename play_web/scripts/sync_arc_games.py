#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLAY_WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLAY_WEB_ROOT.parent
SERVER_DIR = PLAY_WEB_ROOT / "server"


def _load_server_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SERVER_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    _load_server_module("server.arc_agi_adapter", "arc_agi_adapter.py")
    sync_mod = _load_server_module("server.arc_env_sync", "arc_env_sync.py")
    return int(sync_mod._main())


if __name__ == "__main__":
    sys.path[:0] = [str(REPO_ROOT), str(PLAY_WEB_ROOT)]
    raise SystemExit(main())
