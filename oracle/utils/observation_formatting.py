import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .pathfinding import find_path_on_craftax_map

_EXO_TERMINOLOGY_PATH = (
    Path(__file__).resolve().parents[2]
    / "MegaPrompt"
    / "exo-planet_prompt"
    / "world"
    / "terminology.json"
)


def _is_exo_world_mode(world_mode: str | None) -> bool:
    token = str(world_mode or "craftax").strip().lower()
    return token in {"exo", "exo-planet", "exo_planet"}


def _exo_item_labels() -> Dict[str, str]:
    try:
        data = json.loads(_EXO_TERMINOLOGY_PATH.read_text(encoding="utf-8"))
        items = data.get("items")
        if isinstance(items, dict):
            return {str(k): str(v) for k, v in items.items()}
    except Exception:
        pass
    return {}

# Craftax Classic stores mobs outside the tile map (position + mask per group).
_MOB_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ("zombies", "Zombie", "Z"),
    ("cows", "Cow", "C"),
    ("skeletons", "Skeleton", "M"),
)
_MOB_LEGEND = "Z=zombie, C=cow, M=skeleton"
_DANGEROUS_MOB_NAMES = frozenset({"Zombie", "Skeleton"})


def format_inventory_from_env_state(state, *, world_mode: str | None = None) -> str:
    if state is None or not hasattr(state, "inventory"):
        return "Empty"

    inv = state.inventory
    exo_labels = _exo_item_labels() if _is_exo_world_mode(world_mode) else None
    items = []
    for attr, label in (
        ("wood", "wood"),
        ("stone", "stone"),
        ("coal", "coal"),
        ("iron", "iron"),
        ("diamond", "diamond"),
        ("sapling", "sapling"),
        ("wood_pickaxe", "wood_pickaxe"),
        ("stone_pickaxe", "stone_pickaxe"),
        ("iron_pickaxe", "iron_pickaxe"),
        ("wood_sword", "wood_sword"),
        ("stone_sword", "stone_sword"),
        ("iron_sword", "iron_sword"),
    ):
        count = int(getattr(inv, attr, 0) or 0)
        if count <= 0:
            continue
        display = exo_labels.get(label, label) if exo_labels else label
        items.append(f"{display}: {count}")

    if not items:
        return "Empty"
    return ", ".join(items)


def _relative_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    fx, fy = from_pos
    tx, ty = to_pos
    dx = tx - fx
    dy = ty - fy

    if dx == 0 and dy == 0:
        return "you are already on this block"

    moves: List[str] = []
    if dx < 0:
        moves.append("up")
    elif dx > 0:
        moves.append("down")
    if dy < 0:
        moves.append("left")
    elif dy > 0:
        moves.append("right")

    if not moves:
        return "near you"
    if len(moves) == 1:
        return f"to reach this block you need to move {moves[0]}"
    return f"to reach this block you need to move {moves[0]} and {moves[1]}"


def _tile_name_from_id(block_id: int) -> str:
    try:
        constants = importlib.import_module("craftax.craftax_classic.constants")
        block_type = constants.BlockType
        bt = block_type(block_id)
    except Exception:
        return "Unknown"
    return bt.name.title().replace("_", " ")


def _iter_active_mobs(state) -> List[Tuple[str, Tuple[int, int]]]:
    """Return (display_name, (x, y)) for each active mob in the world."""
    if state is None:
        return []

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


def _mob_symbol_map(state) -> Dict[Tuple[int, int], str]:
    """Map cell -> single-char mob symbol for active mobs."""
    symbols: Dict[Tuple[int, int], str] = {}
    if state is None:
        return symbols
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


def _mob_name_at(state, x: int, y: int) -> Optional[str]:
    for display_name, (mx, my) in _iter_active_mobs(state):
        if mx == x and my == y:
            return display_name
    return None


def _manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _nearest_unique(
    items: List[Tuple[str, Tuple[int, int]]],
    player_pos: Tuple[int, int],
    k: int,
) -> List[Tuple[str, Tuple[int, int]]]:
    """Up to k entries: nearest instance of each unique name."""
    sorted_items = sorted(items, key=lambda t: _manhattan_distance(player_pos, t[1]))
    seen_names: Set[str] = set()
    closest: List[Tuple[str, Tuple[int, int]]] = []
    for name, pos in sorted_items:
        if name in seen_names:
            continue
        seen_names.add(name)
        closest.append((name, pos))
        if len(closest) >= k:
            break
    return closest


def _format_dangerous_mob_cautions(
    state,
    player_pos: Tuple[int, int],
) -> List[str]:
    dangerous = [
        (name, pos)
        for name, pos in _iter_active_mobs(state)
        if name in _DANGEROUS_MOB_NAMES
    ]
    if not dangerous:
        return []

    dangerous.sort(key=lambda t: _manhattan_distance(player_pos, t[1]))
    lines: List[str] = []
    for name, (x, y) in dangerous:
        steps = _manhattan_distance(player_pos, (x, y))
        step_word = "step" if steps == 1 else "steps"
        lines.append(
            f"CAUTION! {name} near you at [{x}, {y}], {steps} {step_word} away. "
            "Avoid it, or kill it by turning to face the monster and using DO several times."
        )
    return lines


def _iter_non_grass_tiles(state) -> List[Tuple[str, Tuple[int, int]]]:
    if state is None or not hasattr(state, "map"):
        return []

    grid = state.map
    h, w = grid.shape
    tiles: List[Tuple[str, Tuple[int, int]]] = []

    try:
        constants = importlib.import_module("craftax.craftax_classic.constants")
        block_type = constants.BlockType
        out_of_bounds_id = block_type.OUT_OF_BOUNDS.value
        grass_id = block_type.GRASS.value
    except Exception:
        out_of_bounds_id = None
        grass_id = None

    for x in range(h):
        for y in range(w):
            block_id = int(grid[x, y])
            if block_id < 0:
                continue
            if out_of_bounds_id is not None and block_id == out_of_bounds_id:
                continue
            if grass_id is not None and block_id == grass_id:
                continue
            tiles.append((_tile_name_from_id(block_id), (x, y)))
    return tiles


def _block_to_symbol(block_id: int) -> str:
    try:
        constants = importlib.import_module("craftax.craftax_classic.constants")
        block_type = constants.BlockType
        bt = block_type(block_id)
    except Exception:
        return "?"

    symbol_map = {
        block_type.GRASS: ".",
        block_type.STONE: "O",
        block_type.TREE: "T",
        block_type.WATER: "~",
        block_type.PATH: "=",
        block_type.SAND: ":",
        block_type.OUT_OF_BOUNDS: "#",
    }
    return symbol_map.get(bt, "?")


def render_symbolic_map_from_env_state(
    state,
    player_symbol: str = "P",
    show_top_axis: bool = True,
    show_bottom_axis: bool = False,
    radius: int = 5,
) -> str:
    if state is None or not hasattr(state, "map") or not hasattr(state, "player_position"):
        return ""

    grid = state.map
    h, w = grid.shape
    px, py = map(int, state.player_position.tolist())
    mob_symbols = _mob_symbol_map(state)

    x_min = max(0, px - radius)
    x_max = min(h - 1, px + radius)
    y_min = max(0, py - radius)
    y_max = min(w - 1, py + radius)

    window_w = y_max - y_min + 1
    row_label_width = max(2, len(str(h - 1)))
    col_cell_width = max(2, len(str(w - 1)) + 1)

    lines: List[str] = []

    def build_col_header() -> str:
        return " " * (row_label_width + 3) + "".join(
            f"{y:>{col_cell_width}}" for y in range(y_min, y_max + 1)
        )

    border = " " * (row_label_width + 1) + "+" + "-" * (window_w * col_cell_width)

    if show_top_axis:
        lines.append(build_col_header())
    lines.append(border)

    for x in range(x_min, x_max + 1):
        row_cells: List[str] = []
        for y in range(y_min, y_max + 1):
            if x == px and y == py:
                ch = player_symbol
            elif (x, y) in mob_symbols:
                ch = mob_symbols[(x, y)]
            else:
                ch = _block_to_symbol(int(grid[x, y]))
            row_cells.append(f"{ch:>{col_cell_width}}")
        lines.append(f"{x:>{row_label_width}} |" + "".join(row_cells))

    lines.append(border)
    if show_bottom_axis:
        lines.append(build_col_header())
    return "\n".join(lines)


def render_symbolic_map_with_path_from_env_state(
    state,
    path: Optional[Sequence[Tuple[int, int]]] = None,
    center_coord: Optional[Sequence[int]] = None,
    start_coord: Optional[Sequence[int]] = None,
    goal_coord: Optional[Sequence[int]] = None,
    player_symbol: str = "P",
    path_symbol: str = "X",
    start_symbol: str = "S",
    goal_symbol: str = "G",
    show_top_axis: bool = True,
    show_bottom_axis: bool = False,
    radius: int = 5,
    path_step_markers: bool = False,
    path_step_style: str = "circled_mod10",
) -> str:
    if state is None or not hasattr(state, "map") or not hasattr(state, "player_position"):
        return ""

    grid = state.map
    h, w = grid.shape
    px, py = map(int, state.player_position.tolist())
    mob_symbols = _mob_symbol_map(state)

    if center_coord is None or len(center_coord) != 2:
        cx, cy = px, py
    else:
        cx, cy = int(center_coord[0]), int(center_coord[1])

    x_min = max(0, cx - radius)
    x_max = min(h - 1, cx + radius)
    y_min = max(0, cy - radius)
    y_max = min(w - 1, cy + radius)

    path_cells: Set[Tuple[int, int]] = set()
    path_step_labels: dict = {}
    if path:
        path_cells = {
            (int(x), int(y))
            for x, y in path
            if x_min <= int(x) <= x_max and y_min <= int(y) <= y_max
        }
        if path_step_markers:
            circled_digits = ["⓪", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
            for step_idx, (x, y) in enumerate(path):
                x_i, y_i = int(x), int(y)
                if not (x_min <= x_i <= x_max and y_min <= y_i <= y_max):
                    continue
                if path_step_style == "circled_mod10":
                    path_step_labels[(x_i, y_i)] = circled_digits[step_idx % 10]
                elif path_step_style == "digit_mod10":
                    path_step_labels[(x_i, y_i)] = str(step_idx % 10)
                elif path_step_style == "action_arrow":
                    if step_idx + 1 >= len(path):
                        continue
                    next_x, next_y = int(path[step_idx + 1][0]), int(path[step_idx + 1][1])
                    dx = next_x - x_i
                    dy = next_y - y_i
                    if dx == -1 and dy == 0:
                        path_step_labels[(x_i, y_i)] = "↑"
                    elif dx == 1 and dy == 0:
                        path_step_labels[(x_i, y_i)] = "↓"
                    elif dx == 0 and dy == -1:
                        path_step_labels[(x_i, y_i)] = "←"
                    elif dx == 0 and dy == 1:
                        path_step_labels[(x_i, y_i)] = "→"
                    else:
                        path_step_labels[(x_i, y_i)] = path_symbol
                elif path_step_style == "action_word":
                    if step_idx + 1 >= len(path):
                        continue
                    next_x, next_y = int(path[step_idx + 1][0]), int(path[step_idx + 1][1])
                    dx = next_x - x_i
                    dy = next_y - y_i
                    if dx == -1 and dy == 0:
                        path_step_labels[(x_i, y_i)] = "UP"
                    elif dx == 1 and dy == 0:
                        path_step_labels[(x_i, y_i)] = "DOWN"
                    elif dx == 0 and dy == -1:
                        path_step_labels[(x_i, y_i)] = "LEFT"
                    elif dx == 0 and dy == 1:
                        path_step_labels[(x_i, y_i)] = "RIGHT"
                    else:
                        path_step_labels[(x_i, y_i)] = path_symbol
                else:
                    # Prefix step index to avoid confusion with tile symbols like O (stone).
                    path_step_labels[(x_i, y_i)] = f"#{step_idx}"
    start_cell = (
        (int(start_coord[0]), int(start_coord[1]))
        if start_coord is not None and len(start_coord) == 2
        else None
    )
    goal_cell = (
        (int(goal_coord[0]), int(goal_coord[1]))
        if goal_coord is not None and len(goal_coord) == 2
        else None
    )

    window_w = y_max - y_min + 1
    row_label_width = max(2, len(str(h - 1)))
    col_cell_width = max(2, len(str(w - 1)) + 1)
    if path and path_step_markers and path_step_style == "full_index":
        col_cell_width = max(col_cell_width, len(f"#{len(path) - 1}") + 1)
    elif path and path_step_markers and path_step_style == "action_arrow":
        col_cell_width = max(col_cell_width, len("A→") + 1)
    elif path and path_step_markers and path_step_style == "action_word":
        col_cell_width = max(col_cell_width, len("RIGHT") + 1)

    lines: List[str] = []

    def build_col_header() -> str:
        return " " * (row_label_width + 3) + "".join(
            f"{y:>{col_cell_width}}" for y in range(y_min, y_max + 1)
        )

    border = " " * (row_label_width + 1) + "+" + "-" * (window_w * col_cell_width)

    if show_top_axis:
        lines.append(build_col_header())
    lines.append(border)

    for x in range(x_min, x_max + 1):
        row_cells: List[str] = []
        for y in range(y_min, y_max + 1):
            if x == px and y == py:
                if goal_cell is not None and (x, y) == goal_cell:
                    ch = "AG"
                elif (x, y) in path_step_labels and path_step_style == "action_arrow":
                    ch = f"{player_symbol}{path_step_labels[(x, y)]}"
                else:
                    ch = player_symbol
            elif start_cell is not None and (x, y) == start_cell:
                ch = start_symbol
            elif goal_cell is not None and (x, y) == goal_cell:
                ch = goal_symbol
            elif (x, y) in path_step_labels:
                ch = path_step_labels[(x, y)]
            elif (x, y) in path_cells:
                ch = path_symbol
            elif (x, y) in mob_symbols:
                ch = mob_symbols[(x, y)]
            else:
                ch = _block_to_symbol(int(grid[x, y]))
            row_cells.append(f"{ch:>{col_cell_width}}")
        lines.append(f"{x:>{row_label_width}} |" + "".join(row_cells))

    lines.append(border)
    if show_bottom_axis:
        lines.append(build_col_header())
    return "\n".join(lines)


def render_textual_observation_with_path_from_env_state(
    state,
    start: Sequence[int],
    goal: Sequence[int],
    center_coord: Optional[Sequence[int]] = None,
    radius: int = 5,
    path_symbol: str = "X",
    path_step_markers: bool = True,
    path_step_style: str = "action_arrow",
    path: Optional[Sequence[Tuple[int, int]]] = None,
) -> str:
    if path is None:
        path = find_path_on_craftax_map(state=state, start=start, goal=goal)
    else:
        path = list(path)
    map_text = render_symbolic_map_with_path_from_env_state(
        state=state,
        path=path,
        center_coord=center_coord,
        start_coord=start,
        goal_coord=goal,
        player_symbol="A",
        radius=radius,
        path_symbol=path_symbol,
        path_step_markers=path_step_markers,
        path_step_style=path_step_style,
    )
    path_legend = "path as step numbers"
    if not path_step_markers:
        path_legend = "X=path"
    elif path_step_style == "action_arrow":
        path_legend = "↑/←/↓/→=best next action along path"
    elif path_step_style == "action_word":
        path_legend = "UP/LEFT/DOWN/RIGHT=best next action along path"
    elif path_step_style == "circled_mod10":
        path_legend = "⓪..⑨=step index mod 10"
    elif path_step_style == "digit_mod10":
        path_legend = "0..9=step index mod 10"
    else:
        path_legend = "#<n>=step number"
    
    agent_reached_goal = (
        hasattr(state, "player_position")
        and len(goal) == 2
        and int(state.player_position[0]) == int(goal[0])
        and int(state.player_position[1]) == int(goal[1])
    )

    lines = [
        f"Path from [{int(start[0])}, {int(start[1])}] to [{int(goal[0])}, {int(goal[1])}]",
        f"Path length: {len(path)}",
        f"Path visible around center [{int((center_coord or state.player_position)[0])}, {int((center_coord or state.player_position)[1])}] with radius {radius}:",
        map_text,
        "",
        f"Legend: A=agent, G=goal, AG=agent on goal, S=start, {path_legend}, .=grass, O=stone, T=tree, ~=water, =path, :sand, #=out_of_bounds, {_MOB_LEGEND}",
    ]
    if agent_reached_goal:
        lines.extend(["", "AGENT RICH THE GOAL!!!"])

    map = "\n".join(
        lines
    )
    return map


def format_observation_from_env_state(state, k: int = 5, radius: int = 5) -> str:
    if state is None:
        return "No state available."

    px, py = map(int, state.player_position.tolist())
    dir_idx = int(getattr(state, "player_direction", 0))

    direction_name = "unknown"
    front_x, front_y = px, py

    try:
        constants = importlib.import_module("craftax.craftax_classic.constants")
        action = constants.Action
        directions = constants.DIRECTIONS
        if 1 <= dir_idx <= 4:
            direction_name = action(dir_idx).name.lower()
            dir_vec = directions[dir_idx]
            front_x = px + int(dir_vec[0])
            front_y = py + int(dir_vec[1])
    except Exception:
        pass

    lines = [
        f"You in [{px}, {py}]",
        f"Facing: {direction_name}",
    ]

    if 1 <= dir_idx <= 4 and hasattr(state, "map"):
        h, w = state.map.shape
        if 0 <= front_x < h and 0 <= front_y < w:
            mob_ahead = _mob_name_at(state, front_x, front_y)
            if mob_ahead is not None:
                front_name = mob_ahead
            else:
                block_id = int(state.map[front_x, front_y])
                front_name = _tile_name_from_id(block_id)
            rel = _relative_direction((px, py), (front_x, front_y))
            lines.append(
                f"Block you're turned to: {front_name} at [{front_x}, {front_y}] ({rel})"
            )
        else:
            lines.append("Block you're turned to: out of bounds")
    else:
        lines.append("Block you're turned to: unknown")

    lines.append("")
    lines.append("Map:")
    lines.append(render_symbolic_map_from_env_state(state, radius=radius))
    lines.append("")
    lines.append(
        f"Legend: P=player, .=grass, O=stone, T=tree, ~=water, =path, :sand, #=out_of_bounds, {_MOB_LEGEND}"
    )
    lines.append("")

    player_pos = (px, py)
    closest_objects = _nearest_unique(_iter_non_grass_tiles(state), player_pos, k)
    closest_mobs = _nearest_unique(_iter_active_mobs(state), player_pos, k)

    lines.append("Nearest objects:")
    if closest_objects:
        for name, (x, y) in closest_objects:
            rel = _relative_direction(player_pos, (x, y))
            lines.append(f"{name} in [{x}, {y}] ({rel})")
    else:
        lines.append("No nearby non-grass objects found.")

    lines.append("")
    lines.append("Nearest mobs:")
    if closest_mobs:
        for name, (x, y) in closest_mobs:
            rel = _relative_direction(player_pos, (x, y))
            lines.append(f"{name} in [{x}, {y}] ({rel})")
    else:
        lines.append("No mobs nearby.")

    caution_lines = _format_dangerous_mob_cautions(state, player_pos)
    if caution_lines:
        lines.append("")
        lines.extend(caution_lines)

    lines.append("")
    lines.append("The rest is grass. (You can walk only on grass or path or sand)")
    return "\n".join(lines)
