from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .arc_agi_adapter import ARC_ENVIRONMENTS_DIR, SUPPORTED_ARC_GAME_IDS


def scan_local_arc_game_ids(environments_dir: Path | None = None) -> set[str]:
    base = Path(environments_dir or ARC_ENVIRONMENTS_DIR)
    found: set[str] = set()
    if not base.is_dir():
        return found
    for game_dir in sorted(base.iterdir()):
        if not game_dir.is_dir():
            continue
        game_id = game_dir.name.strip().lower()
        if len(game_id) != 4:
            continue
        for version_dir in game_dir.iterdir():
            if not version_dir.is_dir():
                continue
            if (version_dir / "metadata.json").is_file():
                found.add(game_id)
                break
    return found


def sync_arc_games(
    *,
    environments_dir: Path | None = None,
    game_ids: tuple[str, ...] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Download missing ARC-AGI-3 environment files into environments_dir."""
    target_dir = Path(environments_dir or ARC_ENVIRONMENTS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(game_ids or SUPPORTED_ARC_GAME_IDS)
    local_ids = scan_local_arc_game_ids(target_dir)
    results: dict[str, str] = {}

    try:
        import arc_agi  # type: ignore
        from arc_agi import OperationMode  # type: ignore
    except Exception as exc:
        for game_id in requested:
            results[game_id] = "failed"
        results["_error"] = (
            "arc-agi is not installed. Install project requirements with "
            f"`pip install -r requirements.txt`: {exc}"
        )
        return results

    arcade = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(target_dir),
    )

    for game_id in requested:
        normalized = game_id.strip().lower()
        if not force and normalized in local_ids:
            results[normalized] = "skipped"
            continue
        try:
            env = arcade.make(normalized, seed=0)
            if env is None:
                results[normalized] = "failed"
                continue
            env.reset()
            results[normalized] = "downloaded" if force or normalized not in local_ids else "ok"
        except Exception:
            results[normalized] = "failed"

    return results


def arc_game_options_with_availability(
    environments_dir: Path | None = None,
) -> list[dict[str, Any]]:
    from .arc_agi_adapter import ARC_GAME_OPTIONS

    local_ids = scan_local_arc_game_ids(environments_dir)
    options: list[dict[str, Any]] = []
    for option in ARC_GAME_OPTIONS:
        row = dict(option)
        row["available_locally"] = row["id"] in local_ids
        options.append(row)
    return options


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download local ARC-AGI-3 environment files for TalkingHeads."
    )
    parser.add_argument(
        "--environments-dir",
        default=str(ARC_ENVIRONMENTS_DIR),
        help="Target directory for ARC environment files.",
    )
    parser.add_argument(
        "--game-id",
        action="append",
        dest="game_ids",
        help="Sync only this game id (repeatable). Defaults to all supported games.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a local copy already exists.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable results.",
    )
    args = parser.parse_args(argv)

    game_ids = tuple(args.game_ids) if args.game_ids else None
    results = sync_arc_games(
        environments_dir=Path(args.environments_dir),
        game_ids=game_ids,
        force=args.force,
    )
    error = results.pop("_error", None)

    if args.json:
        payload = {"results": results}
        if error:
            payload["error"] = error
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for game_id, status in sorted(results.items()):
            print(f"{game_id}: {status}")
        if error:
            print(f"error: {error}", file=sys.stderr)

    if error:
        return 2
    if any(status == "failed" for status in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
