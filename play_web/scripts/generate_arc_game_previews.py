#!/usr/bin/env python3
"""Render first-frame PNG previews for ARC onboarding tiles."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path

PLAY_WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLAY_WEB_ROOT.parent
SERVER_DIR = PLAY_WEB_ROOT / "server"
OUTPUT_DIR = PLAY_WEB_ROOT / "client" / "assets" / "arc-games"


def _load_server_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SERVER_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate static ARC game preview PNGs for setup wizard tiles."
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for preview PNG files.",
    )
    parser.add_argument(
        "--game-id",
        action="append",
        dest="game_ids",
        help="Generate only this game id (repeatable). Defaults to all supported games.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable results.",
    )
    args = parser.parse_args(argv)

    _load_server_module("server.arc_agi_adapter", "arc_agi_adapter.py")
    _load_server_module("server.arc_env_sync", "arc_env_sync.py")

    from server.arc_agi_adapter import ARC_GAME_OPTIONS, get_arc_game_preview, normalize_arc_game_id
    from server.arc_env_sync import scan_local_arc_game_ids

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_ids = scan_local_arc_game_ids()
    requested = (
        tuple(normalize_arc_game_id(game_id) for game_id in args.game_ids)
        if args.game_ids
        else tuple(option["id"] for option in ARC_GAME_OPTIONS)
    )

    results: dict[str, str] = {}
    for game_id in requested:
        if game_id not in local_ids:
            results[game_id] = "skipped"
            continue
        try:
            preview = get_arc_game_preview(game_id)
            png_b64 = str(preview.get("png_b64") or "").strip()
            if not png_b64:
                results[game_id] = "failed"
                continue
            out_path = output_dir / f"{game_id}.png"
            out_path.write_bytes(base64.b64decode(png_b64))
            results[game_id] = str(out_path)
        except Exception:
            results[game_id] = "failed"

    if args.json:
        print(json.dumps({"results": results}, ensure_ascii=False))
    else:
        for game_id, status in sorted(results.items()):
            print(f"{game_id}: {status}")

    failed = sum(1 for status in results.values() if status == "failed")
    skipped = sum(1 for status in results.values() if status == "skipped")
    if failed:
        return 1
    if skipped == len(results) and results:
        return 2
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(REPO_ROOT), str(PLAY_WEB_ROOT)]
    raise SystemExit(main())
