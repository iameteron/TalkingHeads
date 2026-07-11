from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from megaprompt.vocabulary.context import active_vocabulary, label_item, label_mob, label_tile

try:
    from craftax.craftax_classic.constants import Action, BlockType, DIRECTIONS as _CRAFTAX_DIRECTIONS
except Exception:  # pragma: no cover
    Action = None  # type: ignore
    BlockType = None  # type: ignore
    _CRAFTAX_DIRECTIONS = None

# Craftax Classic `player_direction` matches Action indices 1..4 (see DIRECTIONS).
_FALLBACK_DIR_DELTAS: Dict[int, Tuple[int, int]] = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}
_FALLBACK_DIR_NAMES = {1: "left", 2: "right", 3: "up", 4: "down"}


def _direction_delta(dir_idx: int) -> Optional[Tuple[int, int]]:
    if _CRAFTAX_DIRECTIONS is not None and 0 <= dir_idx < len(_CRAFTAX_DIRECTIONS):
        row = _CRAFTAX_DIRECTIONS[dir_idx]
        return (int(row[0]), int(row[1]))
    return _FALLBACK_DIR_DELTAS.get(dir_idx)


def _direction_display_name(dir_idx: int) -> str:
    if Action is not None and 1 <= dir_idx <= 4:
        try:
            return Action(dir_idx).name.lower()
        except Exception:
            pass
    return _FALLBACK_DIR_NAMES.get(dir_idx, "unknown")

_BALROG_FACING_LABELS = {1: "west", 2: "east", 3: "north", 4: "south"}
_BALROG_FACING_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_BALROG_SKIP_ITEMS = {"grass", "sand", "path"}

_ASCII_MAPPING = {
    "out_of_bounds": "#",
    "unknown": "?",
    "water": "~",
    "grass": ".",
    "stone": "%",
    "tree": "T",
    "wood": "w",
    "sand": ":",
    "lava": "L",
    "coal": "c",
    "iron": "i",
    "diamond": "d",
    "gold": "g",
    "path": "_",
    "table": "h",
    "crafting_table": "H",
    "furnace": "F",
    "plant": "*",
    "ripe_plant": "P",
    "bed": "B",
}

_MOB_GROUPS: List[Tuple[str, str, str]] = [
    ("zombies", "Zombie", "Z"),
    ("cows", "Cow", "C"),
    ("skeletons", "Skeleton", "M"),
]
_DANGEROUS_MOBS = {"Zombie", "Skeleton"}


@dataclass(frozen=True)
class SymbolLegend:
    symbol: str
    name: str


def _tile_name_from_id(block_id: int) -> str:
    return label_tile(_tile_name_lower_from_id(block_id))


def _tile_name_lower_from_id(block_id: int) -> str:
    if BlockType is None:
        return "unknown"
    try:
        bt = BlockType(int(block_id))
    except Exception:
        return "unknown"
    return bt.name.lower()


def _iter_non_grass_tiles(state: Any) -> List[Tuple[str, Tuple[int, int], int]]:
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    tiles: List[Tuple[str, Tuple[int, int], int]] = []

    if BlockType is not None:
        oob = int(BlockType.OUT_OF_BOUNDS.value)
        grass = int(getattr(BlockType, "GRASS").value)
        darkness = int(getattr(BlockType, "DARKNESS", BlockType.OUT_OF_BOUNDS).value)
    else:
        oob, grass, darkness = -999999, -999998, -999997

    for x in range(h):
        for y in range(w):
            block_id = int(map_array[x, y])
            if block_id in (oob, grass, darkness):
                continue
            if block_id < 0:
                continue
            tiles.append((_tile_name_from_id(block_id), (x, y), block_id))
    return tiles


def _nearest_unique_tiles(state: Any, k: int = 5) -> List[Tuple[str, Tuple[int, int]]]:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    tiles = _iter_non_grass_tiles(state)
    tiles_sorted = sorted(tiles, key=lambda t: abs(px - t[1][0]) + abs(py - t[1][1]))

    seen: set[str] = set()
    out: List[Tuple[str, Tuple[int, int]]] = []
    for name, pos, _block_id in tiles_sorted:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, pos))
        if len(out) >= k:
            break
    return out


def _iter_active_mobs(state: Any) -> List[Tuple[str, Tuple[int, int]]]:
    mobs: List[Tuple[str, Tuple[int, int]]] = []
    for attr, display_name, _symbol in _MOB_GROUPS:
        group = getattr(state, attr, None)
        if group is None or not hasattr(group, "mask") or not hasattr(group, "position"):
            continue
        mask = group.mask
        positions = group.position
        try:
            count = int(mask.shape[0])
        except Exception:
            count = len(mask)
        for i in range(count):
            try:
                active = bool(mask[i])
            except Exception:
                active = bool(mask[i].item())  # type: ignore[attr-defined]
            if not active:
                continue
            pos = positions[i]
            x = int(pos[0]) if not hasattr(pos[0], "item") else int(pos[0].item())
            y = int(pos[1]) if not hasattr(pos[1], "item") else int(pos[1].item())
            mobs.append((display_name, (x, y)))
    return mobs


def _nearest_unique_mobs(state: Any, k: int = 5) -> List[Tuple[str, Tuple[int, int]]]:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    mobs = _iter_active_mobs(state)
    mobs_sorted = sorted(mobs, key=lambda t: abs(px - t[1][0]) + abs(py - t[1][1]))

    seen: set[str] = set()
    out: List[Tuple[str, Tuple[int, int]]] = []
    for name, pos in mobs_sorted:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, pos))
        if len(out) >= k:
            break
    return out


def _mob_name_at(state: Any, x: int, y: int) -> Optional[str]:
    for display_name, (mx, my) in _iter_active_mobs(state):
        if mx == x and my == y:
            return display_name
    return None


def _mob_symbol_by_pos(state: Any) -> Dict[Tuple[int, int], str]:
    symbols: Dict[Tuple[int, int], str] = {}
    for attr, _display_name, symbol in _MOB_GROUPS:
        group = getattr(state, attr, None)
        if group is None or not hasattr(group, "mask") or not hasattr(group, "position"):
            continue
        mask = group.mask
        positions = group.position
        try:
            count = int(mask.shape[0])
        except Exception:
            count = len(mask)
        for i in range(count):
            try:
                active = bool(mask[i])
            except Exception:
                active = bool(mask[i].item())  # type: ignore[attr-defined]
            if not active:
                continue
            pos = positions[i]
            x = int(pos[0]) if not hasattr(pos[0], "item") else int(pos[0].item())
            y = int(pos[1]) if not hasattr(pos[1], "item") else int(pos[1].item())
            symbols[(x, y)] = symbol
    return symbols


def _front_block(state: Any) -> Tuple[str, Optional[Tuple[int, int]]]:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    dir_idx = int(getattr(state, "player_direction", 0))
    delta = _direction_delta(dir_idx)
    if delta is None or delta == (0, 0):
        return "Unknown", None

    dx, dy = delta
    fx, fy = px + int(dx), py + int(dy)
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    if not (0 <= fx < h and 0 <= fy < w):
        return "Out of bounds", (fx, fy)
    mob_ahead = _mob_name_at(state, fx, fy)
    if mob_ahead is not None:
        return mob_ahead, (fx, fy)
    block_id = int(map_array[fx, fy])
    return _tile_name_from_id(block_id), (fx, fy)


def _current_block(state: Any) -> Tuple[str, Tuple[int, int]]:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    if 0 <= px < h and 0 <= py < w:
        block_id = int(map_array[px, py])
        return _tile_name_from_id(block_id), (px, py)
    return "Out of bounds", (px, py)


def _moves_to_reach(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    fx, fy = from_pos
    tx, ty = to_pos
    dx = tx - fx
    dy = ty - fy

    parts: List[str] = []
    if dx < 0:
        parts.append("upward")
    elif dx > 0:
        parts.append("downward")
    if dy < 0:
        parts.append("left")
    elif dy > 0:
        parts.append("right")

    return " and ".join(parts) if parts else "stay here (already there)"


def _render_inventory(state: Any) -> str:
    inv = getattr(state, "inventory", None)
    if inv is None:
        return "Inventory: unavailable"

    items: List[str] = []

    if isinstance(inv, Mapping):
        for name, val in inv.items():
            try:
                if hasattr(val, "item"):
                    val = val.item()
                val_int = int(val)
            except Exception:
                continue
            if val_int > 0:
                items.append(f"{label_item(name)}={val_int}")
    else:
        fields = getattr(inv.__class__, "__dataclass_fields__", None)
        if fields:
            for name in fields.keys():
                try:
                    val = getattr(inv, name)
                except Exception:
                    continue
                try:
                    if hasattr(val, "item"):
                        val = val.item()
                    val_int = int(val)
                except Exception:
                    continue
                if val_int > 0:
                    items.append(f"{label_item(name)}={val_int}")
        else:
            for name in dir(inv):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(inv, name)
                except Exception:
                    continue
                if callable(val):
                    continue
                try:
                    if hasattr(val, "item"):
                        val = val.item()
                    val_int = int(val)
                except Exception:
                    continue
                if val_int > 0:
                    items.append(f"{label_item(name)}={val_int}")

    if not items:
        return "Inventory: empty"
    return "Inventory: " + ", ".join(sorted(items))


def render_map(state: Any) -> Tuple[str, List[SymbolLegend]]:
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    px = int(state.player_position[0])
    py = int(state.player_position[1])

    size = 10
    half = size // 2
    top_x = px - half
    left_y = py - half

    header = "y\\x " + " ".join(f"{(left_y + j):2d}" for j in range(size))
    rows: List[str] = [header]
    seen_symbols: Dict[str, str] = {}
    legend: List[SymbolLegend] = [
        SymbolLegend(
            active_vocabulary().labels.get("agent_symbol", "@"),
            active_vocabulary().labels.get("agent", "You"),
        )
    ]
    mob_symbols = _mob_symbol_by_pos(state)

    for i in range(size):
        row_x = top_x + i
        row_cells: List[str] = []
        for j in range(size):
            col_y = left_y + j
            if row_x == px and col_y == py:
                row_cells.append("@")
                continue
            if not (0 <= row_x < h and 0 <= col_y < w):
                row_cells.append("#")
                seen_symbols["#"] = "out_of_bounds"
                continue
            if (row_x, col_y) in mob_symbols:
                sym = mob_symbols[(row_x, col_y)]
                row_cells.append(sym)
                if sym == "Z":
                    seen_symbols["Z"] = "zombie"
                elif sym == "C":
                    seen_symbols["C"] = "cow"
                elif sym == "M":
                    seen_symbols["M"] = "skeleton"
                continue

            block_id = int(map_array[row_x, col_y])
            name = _tile_name_lower_from_id(block_id)
            sym = _ASCII_MAPPING.get(name, (name[0].upper() if name else "?"))
            row_cells.append(sym)
            if sym not in ("@",) and sym not in seen_symbols:
                seen_symbols[sym] = name

        rows.append(f"{row_x:3d} " + "  ".join(row_cells))

    for sym in sorted(seen_symbols.keys()):
        if sym == "#":
            legend.append(SymbolLegend(sym, label_tile("out_of_bounds")))
        elif sym == "Z":
            legend.append(SymbolLegend(sym, label_mob("Zombie")))
        elif sym == "C":
            legend.append(SymbolLegend(sym, label_mob("Cow")))
        elif sym == "M":
            legend.append(SymbolLegend(sym, label_mob("Skeleton")))
        else:
            legend.append(SymbolLegend(sym, label_tile(seen_symbols[sym])))

    return "\n".join(rows), legend


def render_coords_text(state: Any, k_nearest: int = 5) -> str:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    dir_idx = int(getattr(state, "player_direction", 0))
    facing = _direction_display_name(dir_idx)
    current_name, (cx, cy) = _current_block(state)
    front_name, front_pos = _front_block(state)

    lines: List[str] = []
    lines.append(f"You are at coord y={px}, x={py}. You are rotated {facing}.")
    lines.append(f"You are standing on {current_name} at y={cx}, x={cy}.")
    if front_pos is not None:
        fx, fy = front_pos
        lines.append(f"In front of you there is {front_name} at y={fx}, x={fy}.")
    else:
        lines.append(f"In front of you there is {front_name}.")

    lines.append("Nearest objects:")
    for name, (x, y) in _nearest_unique_tiles(state, k=k_nearest):
        plan = _moves_to_reach((px, py), (x, y))
        lines.append(f"- You can see {name} at y={x}, x={y}. To reach it, move {plan}.")
    if not _nearest_unique_tiles(state, k=k_nearest):
        lines.append(
            f"- No nearby {active_vocabulary().labels.get('non_ground_objects', 'non-grass objects')} found."
        )

    lines.append("Nearest mobs:")
    nearest_mobs = _nearest_unique_mobs(state, k=k_nearest)
    for name, (x, y) in nearest_mobs:
        plan = _moves_to_reach((px, py), (x, y))
        lines.append(
            f"- You can see {label_mob(name)} at y={x}, x={y}. To reach it, move {plan}."
        )
    if not nearest_mobs:
        lines.append("- No mobs nearby.")

    danger_mobs = [m for m in _iter_active_mobs(state) if m[0] in _DANGEROUS_MOBS]
    danger_mobs = sorted(danger_mobs, key=lambda m: abs(px - m[1][0]) + abs(py - m[1][1]))
    for name, (x, y) in danger_mobs:
        steps = abs(px - x) + abs(py - y)
        step_word = "step" if steps == 1 else "steps"
        lines.append(
            f"CAUTION! {label_mob(name)} is near you at y={x}, x={y}, {steps} {step_word} away. "
            f"Avoid it, or {active_vocabulary().labels.get('combat_hint', 'kill it by turning to face the monster and using DO several times')}."
        )
    return "\n".join(lines).strip()


def render_map_text(state: Any, include_inventory: bool = True) -> str:
    map_text, legend = render_map(state)
    legend_text = ", ".join(f"'{item.symbol}': {item.name}" for item in legend)
    inventory_block = f"\n\n{_render_inventory(state)}" if include_inventory else ""
    return (
        "Symbolic map 10x10 (agent in the middle). Coordinates shown as [y, x].\n"
        f"{map_text}"
        f"{inventory_block}\n\n"
        f"Legend: {legend_text}"
    ).strip()


def render_map_and_coords_text(state: Any) -> str:
    return f"{render_coords_text(state)}\n\n{render_map_text(state, include_inventory=True)}".strip()


def render_balrog_text(state: Any) -> str:
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    center_x = h // 2
    center_y = w // 2

    def _to_int(value: Any) -> int:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    def _describe_loc(dx: int, dy: int) -> str:
        parts: List[str] = []
        if dx < 0:
            parts.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} north")
        elif dx > 0:
            parts.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} south")
        if dy < 0:
            parts.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} west")
        elif dy > 0:
            parts.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} east")
        return " and ".join(parts) if parts else "at your location"

    # Track nearest visible location for each object type.
    nearest: Dict[str, Tuple[int, str]] = {}
    for x in range(h):
        for y in range(w):
            if x == center_x and y == center_y:
                continue
            block_name = _tile_name_lower_from_id(int(map_array[x, y]))
            if block_name in _BALROG_SKIP_ITEMS or block_name == "out_of_bounds":
                continue

            dx, dy = x - center_x, y - center_y
            distance = abs(dx) + abs(dy)
            if block_name not in nearest or distance < nearest[block_name][0]:
                nearest[block_name] = (distance, _describe_loc(dx, dy))

    if nearest:
        sorted_items = sorted(nearest.items(), key=lambda kv: kv[1][0])
        see_text = "You see:\n" + "\n".join(
            f"- {label_tile(name)} {loc}" for name, (_, loc) in sorted_items
        )
    else:
        see_text = "You see nothing away from you."

    try:
        facing = _to_int(getattr(state, "player_direction", 0))
    except Exception:
        facing = 0

    fdx, fdy = _BALROG_FACING_DELTAS.get(facing, (0, 0))
    tx, ty = center_x + fdx, center_y + fdy
    mob_ahead = _mob_name_at(state, tx, ty)
    if mob_ahead is not None:
        front_item = label_mob(mob_ahead)
    elif 0 <= tx < h and 0 <= ty < w:
        front_item = label_tile(_tile_name_lower_from_id(int(map_array[tx, ty])))
        if _tile_name_lower_from_id(int(map_array[tx, ty])) in _BALROG_SKIP_ITEMS:
            front_item = "nothing"
    else:
        front_item = "nothing"

    status_fields = [
        ("health", "player_health", "health"),
        ("food", "player_food", "food"),
        ("drink", "player_drink", "drink"),
        ("energy", "player_energy", "energy"),
    ]
    status_lines: List[str] = []
    for label, primary, fallback in status_fields:
        value = 0
        if hasattr(state, primary):
            value = _to_int(getattr(state, primary))
        elif hasattr(state, fallback):
            value = _to_int(getattr(state, fallback))
        status_lines.append(f"- {active_vocabulary().vital(label)}: {value}/9")
    status_text = "Your status:\n" + "\n".join(status_lines)

    inv = getattr(state, "inventory", None)
    inv_lines: List[str] = []
    vitals = {"health", "food", "drink", "energy", "mana"}
    if inv is not None:
        fields = getattr(inv.__class__, "__dataclass_fields__", None)
        if fields:
            names = fields.keys()
        elif isinstance(inv, Mapping):
            names = inv.keys()
        else:
            names = [n for n in dir(inv) if not n.startswith("_")]

        for field in names:
            if field in vitals:
                continue
            try:
                raw_value = inv[field] if isinstance(inv, Mapping) else getattr(inv, field)
            except Exception:
                continue
            try:
                value = _to_int(raw_value)
            except Exception:
                continue
            if value > 0:
                inv_lines.append(f"- {label_item(field)}: {value}")

    inventory_text = "Your inventory:\n" + "\n".join(inv_lines) if inv_lines else "You have nothing in your inventory."
    facing_label = _BALROG_FACING_LABELS.get(facing, "unknown")
    return (
        f"{see_text}\n\n"
        f"You face {front_item} at your front.\n\n"
        f"You are facing: {facing_label}\n\n"
        f"{status_text}\n\n"
        f"{inventory_text}"
    ).strip()
