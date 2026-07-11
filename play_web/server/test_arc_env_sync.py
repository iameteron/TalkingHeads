from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SERVER_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ADAPTER = _load_module("server.arc_agi_adapter", "arc_agi_adapter.py")
_MOD = _load_module("server.arc_env_sync", "arc_env_sync.py")

scan_local_arc_game_ids = _MOD.scan_local_arc_game_ids
arc_game_options_with_availability = _MOD.arc_game_options_with_availability


def test_scan_local_arc_game_ids_finds_installed_games():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        metadata_dir = base / "ls20" / "9607627b"
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "metadata.json").write_text(
            json.dumps({"game_id": "ls20-9607627b"}),
            encoding="utf-8",
        )
        found = scan_local_arc_game_ids(base)
        assert found == {"ls20"}


def test_arc_game_options_with_availability_marks_local_games():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        metadata_dir = base / "lp85" / "305b61c3"
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "metadata.json").write_text(
            json.dumps({"game_id": "lp85-305b61c3"}),
            encoding="utf-8",
        )
        options = arc_game_options_with_availability(base)
        by_id = {row["id"]: row for row in options}
        assert by_id["lp85"]["available_locally"] is True
        assert by_id["ar25"]["available_locally"] is False


if __name__ == "__main__":
    test_scan_local_arc_game_ids_finds_installed_games()
    test_arc_game_options_with_availability_marks_local_games()
    print("ok")
