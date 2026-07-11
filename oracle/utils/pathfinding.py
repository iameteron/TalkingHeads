import importlib
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple


def find_path_on_craftax_map(
    state,
    start: Sequence[int],
    goal: Sequence[int],
    blocked_tile_ids: Optional[Set[int]] = None,
    use_nearest_reachable_fallback: bool = True,
) -> List[Tuple[int, int]]:
    if state is None or not hasattr(state, "map"):
        return []

    grid = state.map
    h, w = grid.shape

    if len(start) != 2 or len(goal) != 2:
        return []

    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < h and 0 <= y < w

    if not in_bounds(sx, sy) or not in_bounds(gx, gy):
        return []

    if blocked_tile_ids is None:
        try:
            constants = importlib.import_module("craftax.craftax_classic.constants")
            block_type = constants.BlockType

            walkable = {
                block_type.GRASS.value,
                block_type.PATH.value,
                block_type.SAND.value,
            }
            blocked_tile_ids = {
                int(grid[x, y])
                for x in range(h)
                for y in range(w)
                if int(grid[x, y]) not in walkable
            }
        except Exception:
            blocked_tile_ids = set()

    def is_walkable(x: int, y: int) -> bool:
        if (x, y) == (sx, sy) or (x, y) == (gx, gy):
            return True
        return int(grid[x, y]) not in blocked_tile_ids

    queue = deque([(sx, sy)])
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sx, sy): None}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        if (x, y) == (gx, gy):
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if (nx, ny) in parent:
                continue
            if not is_walkable(nx, ny):
                continue
            parent[(nx, ny)] = (x, y)
            queue.append((nx, ny))

    target = (gx, gy)
    if target not in parent:
        if not use_nearest_reachable_fallback or not parent:
            return []
        # If goal is unreachable, route to the explored cell that is
        # closest to the goal by Manhattan distance.
        target = min(parent.keys(), key=lambda p: abs(p[0] - gx) + abs(p[1] - gy))

    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path
