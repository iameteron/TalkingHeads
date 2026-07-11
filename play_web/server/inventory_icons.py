"""Theme-aware inventory item icons (base64 PNG) for the world HUD panel.

Icons are loaded from the same theme-mapped PNG sources as the in-game renderer
(`texture_mapping.txt` for Exo-Planet, Craftax assets otherwise) so HUD slots
always match the active world mode.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

from craftax.craftax_classic.constants import BlockType
from external_visualization.load_textures import (
    BLOCK_PIXEL_SIZE_IMG,
    _resolve_texture_path_for_theme,
    get_texture_bundle,
)

ICON_SIZE = 64
_SPRITE_ICON_SIZE = int(BLOCK_PIXEL_SIZE_IMG * 0.8)

# Fixed slot order for the 4x3 HUD panel: resources, then pickaxes, then swords.
INVENTORY_SLOT_ORDER: list[str] = [
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
]

# Inventory keys mapped to block-tile PNG filenames (same names as load_textures).
_BLOCK_ASSET: dict[str, str] = {
    "wood": "wood.png",
    "stone": "stone.png",
    "coal": "coal.png",
    "iron": "iron.png",
    "diamond": "diamond.png",
    # Placeable structures used by the companion achievement strip.
    "crafting_table": "table.png",
    "furnace": "furnace.png",
}

# Standalone tool / item sprites (inventory-sized in the renderer).
_SPRITE_ASSET: dict[str, str] = {
    "wood_pickaxe": "wood_pickaxe.png",
    "stone_pickaxe": "stone_pickaxe.png",
    "iron_pickaxe": "iron_pickaxe.png",
    "wood_sword": "wood_sword.png",
    "stone_sword": "stone_sword.png",
    "iron_sword": "iron_sword.png",
    "sapling": "sapling.png",
}

_BLOCK_TEXTURE_INDEX: dict[str, int] = {
    "wood": int(BlockType.WOOD.value),
    "stone": int(BlockType.STONE.value),
    "coal": int(BlockType.COAL.value),
    "iron": int(BlockType.IRON.value),
    "diamond": int(BlockType.DIAMOND.value),
    "crafting_table": int(BlockType.CRAFTING_TABLE.value),
    "furnace": int(BlockType.FURNACE.value),
}

_ICON_CACHE: dict[str, dict[str, str]] = {}


def _normalize_theme(theme: str) -> str:
    return (
        "exo-planet"
        if str(theme).strip().lower() in {"exo", "exo-planet"}
        else "craftax"
    )


def _png_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _load_icon_from_png(path: str, *, size: int) -> str | None:
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGBA")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.NEAREST)
    return _png_to_data_url(img)


def _array_to_data_url(arr, *, key_out_black: bool) -> str | None:
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] < 3:
        return None
    rgb = np.clip(a[:, :, :3], 0, 255).astype(np.uint8)
    if key_out_black:
        alpha = (rgb.max(axis=2) > 8).astype(np.uint8) * 255
    else:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([rgb, alpha])
    img = Image.fromarray(rgba, mode="RGBA")
    if img.size != (ICON_SIZE, ICON_SIZE):
        img = img.resize((ICON_SIZE, ICON_SIZE), resample=Image.NEAREST)
    return _png_to_data_url(img)


def _icon_from_bundle_block(bundle: dict, index: int) -> str | None:
    textures = bundle.get("smaller_block_textures")
    if textures is None or not (0 <= index < len(textures)):
        textures = bundle.get("block_textures")
    if textures is None or not (0 <= index < len(textures)):
        return None
    return _array_to_data_url(textures[index], key_out_black=False)


def _icon_from_bundle_sprite(bundle: dict, texture_key: str) -> str | None:
    texture = bundle.get(texture_key)
    if texture is None:
        return None
    return _array_to_data_url(texture, key_out_black=True)


def get_inventory_icons(theme: str) -> dict[str, str]:
    """Return a mapping of inventory key -> base64 PNG data URL for ``theme``."""
    normalized = _normalize_theme(theme)
    cached = _ICON_CACHE.get(normalized)
    if cached is not None:
        return cached

    bundle = get_texture_bundle(normalized).get(BLOCK_PIXEL_SIZE_IMG, {})
    icons: dict[str, str] = {}

    for key, filename in _BLOCK_ASSET.items():
        path = _resolve_texture_path_for_theme(filename, normalized)
        data_url = _load_icon_from_png(path, size=ICON_SIZE)
        if data_url is None:
            data_url = _icon_from_bundle_block(bundle, _BLOCK_TEXTURE_INDEX[key])
        if data_url:
            icons[key] = data_url

    for key, filename in _SPRITE_ASSET.items():
        path = _resolve_texture_path_for_theme(filename, normalized)
        data_url = _load_icon_from_png(path, size=_SPRITE_ICON_SIZE)
        if data_url is None:
            legacy_key = f"{key}_texture" if key != "sapling" else "sapling_texture"
            data_url = _icon_from_bundle_sprite(bundle, legacy_key)
        if data_url:
            icons[key] = data_url

    _ICON_CACHE[normalized] = icons
    return icons
