"""Build world-view payloads for the fullscreen map client."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN
from external_visualization.load_textures import OBS_DIM
from external_visualization.render import render_tile_patch_np


def _pixels_to_uint8(pixels: Any) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _inventory_payload(state) -> dict[str, int]:
    inv = state.inventory
    fields = (
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapling",
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
    )
    return {name: int(getattr(inv, name, 0) or 0) for name in fields}


def _inventory_items(inventory: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": count}
        for key, count in inventory.items()
        if count > 0
    ]


def _mobs_payload(state) -> list[dict[str, Any]]:
    mobs: list[dict[str, Any]] = []
    groups = (
        ("zombie", getattr(state, "zombies", None)),
        ("cow", getattr(state, "cows", None)),
        ("skeleton", getattr(state, "skeletons", None)),
    )
    for kind, group in groups:
        if group is None:
            continue
        positions = np.asarray(group.position)
        mask = np.asarray(group.mask)
        if positions.ndim != 2 or positions.shape[0] == 0:
            continue
        for i in range(positions.shape[0]):
            if not bool(mask[i]):
                continue
            mobs.append(
                {
                    "type": kind,
                    "x": int(positions[i][0]),
                    "y": int(positions[i][1]),
                }
            )
    return mobs


def build_world_payload(
    session,
    *,
    block_px: int = BLOCK_PIXEL_SIZE_HUMAN,
    force_full_map: bool = False,
    pixels_to_png_base64,
) -> dict[str, Any]:
    state = session.state
    map_grid = np.asarray(state.map, dtype=np.int32)
    px, py = int(state.player_position[0]), int(state.player_position[1])
    obs_h, obs_w = int(OBS_DIM[0]), int(OBS_DIM[1])
    inventory = _inventory_payload(state)

    world: dict[str, Any] = {
        "map_h": int(map_grid.shape[0]),
        "map_w": int(map_grid.shape[1]),
        "block_px": int(block_px),
        "obs_dim": [obs_h, obs_w],
        "player_pos": [px, py],
        "player_direction": int(state.player_direction or 0),
        "stats": {
            "health": int(state.player_health),
            "food": int(state.player_food),
            "drink": int(state.player_drink),
            "energy": int(state.player_energy),
        },
        "inventory": inventory,
        "inventory_items": _inventory_items(inventory),
        "obs_origin": [px - obs_h // 2, py - obs_w // 2],
        "mobs": _mobs_payload(state),
        "map_epoch": int(getattr(session, "map_epoch", 0) or 0),
    }

    last_grid: Optional[np.ndarray] = getattr(session, "_last_map_grid", None)
    if force_full_map or last_grid is None or last_grid.shape != map_grid.shape:
        base_pixels = _pixels_to_uint8(
            session.render_world_base_fn(state, block_pixel_size=block_px)
        )
        world["map_full_b64"] = pixels_to_png_base64(base_pixels)
        world["map_diff"] = []
        world["map_blocks"] = map_grid.astype(int).tolist()
        session._last_map_grid = map_grid.copy()
    else:
        patches: list[dict[str, Any]] = []
        diff_mask = map_grid != last_grid
        if np.any(diff_mask):
            xs, ys = np.where(diff_mask)
            for x, y in zip(xs.tolist(), ys.tolist()):
                patch = render_tile_patch_np(
                    state,
                    int(x),
                    int(y),
                    block_px,
                    session.texture_theme,
                )
                patches.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "block_id": int(map_grid[x, y]),
                        "png_b64": pixels_to_png_base64(patch),
                    }
                )
        world["map_diff"] = patches
        session._last_map_grid = map_grid.copy()

    obs_pixels = _pixels_to_uint8(
        session.render_obs_overlay_fn(state, block_pixel_size=block_px)
    )
    world["obs_overlay_b64"] = pixels_to_png_base64(obs_pixels)
    world["obs_overlay_w"] = int(obs_pixels.shape[1])
    world["obs_overlay_h"] = int(obs_pixels.shape[0])
    return world
