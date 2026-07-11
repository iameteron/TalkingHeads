import os
import pathlib
from glob import glob

import jax.numpy as jnp
import imageio.v3 as iio
import numpy as np
from PIL import Image
from craftax.environment_base.util import load_compressed_pickle, save_compressed_pickle
from craftax.craftax_classic.constants import BlockType

# GAME CONSTANTS
OBS_DIM = (7, 9)
MAX_OBS_DIM = max(OBS_DIM)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 64
BLOCK_PIXEL_SIZE_AGENT = 7
INVENTORY_OBS_HEIGHT = 2
TEXTURE_CACHE_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache_classic.pbz2"
)
TEXTURE_CACHE_EXO_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache_exo.pbz2"
)
ASSETS_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets")
EXO_MOD_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "exo-planet_mod")
EXO_TEXTURE_MAPPING_FILE = os.path.join(EXO_MOD_DIR, "texture_mapping.txt")
ACTIVE_TEXTURE_THEME = "craftax"
EXO_TEXTURE_MAPPING: dict[str, str] = {}
TEXTURES_BY_THEME: dict[str, dict] = {}


def _parse_exo_texture_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not os.path.exists(EXO_TEXTURE_MAPPING_FILE):
        return mapping
    with open(EXO_TEXTURE_MAPPING_FILE, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ">" not in line:
                continue
            key_raw, value_raw = line.split(">", 1)
            key = key_raw.strip()
            value = value_raw.strip()
            if key and value:
                mapping[key] = value
    return mapping


def _resolve_wildcard_path(raw_path: str) -> str | None:
    if "*" not in raw_path:
        return raw_path if os.path.exists(raw_path) else None
    matches = sorted(glob(raw_path))
    return matches[0] if matches else None


def _resolve_texture_path_for_theme(filename: str, texture_theme: str) -> str:
    theme = (
        "exo-planet"
        if str(texture_theme).strip().lower() in {"exo", "exo-planet"}
        else "craftax"
    )
    if theme != "exo-planet":
        return os.path.join(ASSETS_DIR, filename)

    mapping = _parse_exo_texture_mapping()
    mapped = mapping.get(filename, "").strip()
    if mapped:
        # Keep "A + B" compatibility, but prefer pre-composed file if present.
        if "+" in mapped:
            first = mapped.split("+", 1)[0].strip()
            first_path = os.path.join(pathlib.Path(__file__).parent.resolve(), first)
            first_resolved = _resolve_wildcard_path(first_path)
            if first_resolved:
                return first_resolved
        else:
            full_path = os.path.join(pathlib.Path(__file__).parent.resolve(), mapped)
            resolved = _resolve_wildcard_path(full_path)
            if resolved:
                return resolved

    # Fallback: if exo mod directly has the same texture filename.
    exo_direct = os.path.join(EXO_MOD_DIR, filename)
    if os.path.exists(exo_direct):
        return exo_direct

    # Final fallback to original Craftax asset.
    return os.path.join(ASSETS_DIR, filename)


def resolve_agent_icon_path(texture_theme: str) -> str:
    """Filesystem path to the player/agent sprite for the given world mode."""
    return _resolve_texture_path_for_theme("player.png", texture_theme)


def _resolve_texture_path(filename: str) -> str:
    if ACTIVE_TEXTURE_THEME != "exo-planet":
        return os.path.join(ASSETS_DIR, filename)

    mapped = EXO_TEXTURE_MAPPING.get(filename, "").strip()
    if mapped:
        # Keep "A + B" compatibility, but prefer pre-composed file if present.
        if "+" in mapped:
            first = mapped.split("+", 1)[0].strip()
            first_path = os.path.join(pathlib.Path(__file__).parent.resolve(), first)
            first_resolved = _resolve_wildcard_path(first_path)
            if first_resolved:
                return first_resolved
        else:
            full_path = os.path.join(pathlib.Path(__file__).parent.resolve(), mapped)
            resolved = _resolve_wildcard_path(full_path)
            if resolved:
                return resolved

    # Fallback: if exo mod directly has the same texture filename.
    exo_direct = os.path.join(EXO_MOD_DIR, filename)
    if os.path.exists(exo_direct):
        return exo_direct

    # Final fallback to original Craftax asset.
    return os.path.join(ASSETS_DIR, filename)
# TEXTURES
def load_texture(filename, block_pixel_size, clamp_alpha=True):
    filename = _resolve_texture_path(filename)
    img = iio.imread(filename)
    # Normalize all inputs to RGBA so mixed RGB/RGBA sources do not break stacking.
    pil_img = Image.fromarray(img).convert("RGBA")

    # resize to BLOCK_PIXEL_SIZE_IMGxBLOCK_PIXEL_SIZE_IMG if not already
    if pil_img.size != (BLOCK_PIXEL_SIZE_IMG, BLOCK_PIXEL_SIZE_IMG):
        pil_img = pil_img.resize(
            (BLOCK_PIXEL_SIZE_IMG, BLOCK_PIXEL_SIZE_IMG), resample=Image.NEAREST
        )
    img = np.array(pil_img)

    jnp_img = jnp.array(img).astype(int)
    print(jnp_img.shape)
    assert jnp_img.shape[:2] == (BLOCK_PIXEL_SIZE_IMG, BLOCK_PIXEL_SIZE_IMG)

    if jnp_img.shape[2] == 4 and clamp_alpha:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    if block_pixel_size != BLOCK_PIXEL_SIZE_IMG:
        img = np.array(jnp_img, dtype=np.uint8)
        image = Image.fromarray(img)
        image = image.resize(
            (block_pixel_size, block_pixel_size), resample=Image.NEAREST
        )
        jnp_img = jnp.array(image, dtype=jnp.int32)

    return jnp_img


def load_all_textures(block_pixel_size):
    small_block_pixel_size = int(block_pixel_size * 0.8)

    # blocks
    texture_names = [
        "debug_tile.png",
        "debug_tile.png",
        "grass.png",
        "water.png",
        "stone.png",
        "tree.png",
        "wood.png",
        "path.png",
        "coal.png",
        "iron.png",
        "diamond.png",
        "table.png",
        "furnace.png",
        "sand.png",
        "lava.png",
        "plant_on_grass.png",
        "ripe_plant_on_grass.png",
    ]

    block_textures = jnp.array(
        [load_texture(fname, block_pixel_size)[:, :, :3] for fname in texture_names]
    )
    block_textures = block_textures.at[1].set(
        jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128
    )

    # rng = jax.random.prngkey(0)
    # block_textures = jax.random.permutation(rng, block_textures)

    smaller_block_textures = jnp.array(
        [
            load_texture(fname, int(block_pixel_size * 0.8))[:, :, :3]
            for fname in texture_names
        ]
    )

    full_map_block_textures = jnp.array(
        [jnp.tile(block_textures[block.value], (*OBS_DIM, 1)) for block in BlockType]
    )

    # player
    pad_pixels = (
        (OBS_DIM[0] // 2) * block_pixel_size,
        (OBS_DIM[1] // 2) * block_pixel_size,
    )

    player_textures = [
        load_texture("player-left.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-right.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-up.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-down.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-sleep.png", block_pixel_size, clamp_alpha=False),
    ]

    full_map_player_textures_rgba = [
        jnp.pad(
            player_texture,
            ((pad_pixels[0], pad_pixels[0]), (pad_pixels[1], pad_pixels[1]), (0, 0)),
        )
        for player_texture in player_textures
    ]

    full_map_player_textures = jnp.array(
        [player_texture[:, :, :3] for player_texture in full_map_player_textures_rgba]
    )

    full_map_player_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(player_texture[:, :, 3], axis=-1).astype(float) / 255,
                repeats=3,
                axis=2,
            )
            for player_texture in full_map_player_textures_rgba
        ]
    )

    # inventory

    empty_texture = jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    smaller_empty_texture = jnp.zeros(
        (int(block_pixel_size * 0.8), int(block_pixel_size * 0.8), 3), dtype=jnp.int32
    )

    ones_texture = jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)

    number_size = int(block_pixel_size * 0.6)

    number_textures_rgba = [
        jnp.zeros((number_size, number_size, 3), dtype=jnp.int32),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2
            )
            for number_texture in number_textures_rgba
        ]
    )

    health_texture = jnp.array(
        load_texture("health.png", small_block_pixel_size)[:, :, :3]
    )
    hunger_texture = jnp.array(
        load_texture("food.png", small_block_pixel_size)[:, :, :3]
    )
    thirst_texture = jnp.array(
        load_texture("drink.png", small_block_pixel_size)[:, :, :3]
    )
    energy_texture = jnp.array(
        load_texture("energy.png", small_block_pixel_size)[:, :, :3]
    )

    # get rid of the cow ghost
    def apply_alpha(texture):
        return texture[:, :, :3] * jnp.repeat(
            jnp.expand_dims(texture[:, :, 3], axis=-1), 3, axis=-1
        )

    wood_pickaxe_texture = jnp.array(
        load_texture("wood_pickaxe.png", small_block_pixel_size)[:, :, :3]
    )  # no ghosts :)
    stone_pickaxe_texture = jnp.array(
        load_texture("stone_pickaxe.png", small_block_pixel_size)
    )
    stone_pickaxe_texture = apply_alpha(stone_pickaxe_texture)
    iron_pickaxe_texture = jnp.array(
        load_texture("iron_pickaxe.png", small_block_pixel_size)
    )
    iron_pickaxe_texture = apply_alpha(iron_pickaxe_texture)

    wood_sword_texture = jnp.array(
        load_texture("wood_sword.png", small_block_pixel_size)
    )
    wood_sword_texture = apply_alpha(wood_sword_texture)
    stone_sword_texture = jnp.array(
        load_texture("stone_sword.png", small_block_pixel_size)
    )
    stone_sword_texture = apply_alpha(stone_sword_texture)
    iron_sword_texture = jnp.array(
        load_texture("iron_sword.png", small_block_pixel_size)
    )
    iron_sword_texture = apply_alpha(iron_sword_texture)

    sapling_texture = jnp.array(
        load_texture("sapling.png", small_block_pixel_size)[:, :, :3]
    )

    # entities
    zombie_texture_rgba = jnp.array(
        load_texture("zombie.png", block_pixel_size, clamp_alpha=False)
    )
    zombie_texture = zombie_texture_rgba[:, :, :3]
    zombie_texture_alpha = jnp.repeat(
        jnp.expand_dims(zombie_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    cow_texture_rgba = jnp.array(
        load_texture("cow.png", block_pixel_size, clamp_alpha=False)
    )
    cow_texture = cow_texture_rgba[:, :, :3]
    cow_texture_alpha = jnp.repeat(
        jnp.expand_dims(cow_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    skeleton_texture_rgba = jnp.array(
        load_texture("skeleton.png", block_pixel_size, clamp_alpha=False)
    )
    skeleton_texture = skeleton_texture_rgba[:, :, :3]
    skeleton_texture_alpha = jnp.repeat(
        jnp.expand_dims(skeleton_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    arrow_texture_rgba = jnp.array(load_texture("arrow-up.png", block_pixel_size))
    arrow_texture = apply_alpha(arrow_texture_rgba)
    arrow_texture_alpha = jnp.repeat(
        jnp.expand_dims(arrow_texture_rgba[:, :, 3], axis=-1), repeats=3, axis=2
    )

    night_texture = (
        jnp.array([[[0, BLOCK_PIXEL_SIZE_IMG, 64]]])
        .repeat(OBS_DIM[0] * block_pixel_size, axis=0)
        .repeat(OBS_DIM[1] * block_pixel_size, axis=1)
    )

    xs, ys = np.meshgrid(
        np.linspace(-1, 1, OBS_DIM[0] * block_pixel_size),
        np.linspace(-1, 1, OBS_DIM[1] * block_pixel_size),
    )
    night_noise_intensity_texture = (
        1 - np.exp(-0.5 * (xs**2 + ys**2) / (0.5**2)).T
    )

    night_noise_intensity_texture = jnp.expand_dims(
        night_noise_intensity_texture, axis=-1
    ).repeat(3, axis=-1)

    return {
        "block_textures": block_textures,
        "smaller_block_textures": smaller_block_textures,
        "full_map_block_textures": full_map_block_textures,
        "player_textures": player_textures,
        "full_map_player_textures": full_map_player_textures,
        "full_map_player_textures_alpha": full_map_player_textures_alpha,
        "empty_texture": empty_texture,
        "smaller_empty_texture": smaller_empty_texture,
        "ones_texture": ones_texture,
        "number_textures": number_textures,
        "number_textures_alpha": number_textures_alpha,
        "health_texture": health_texture,
        "hunger_texture": hunger_texture,
        "thirst_texture": thirst_texture,
        "energy_texture": energy_texture,
        "wood_pickaxe_texture": wood_pickaxe_texture,
        "stone_pickaxe_texture": stone_pickaxe_texture,
        "iron_pickaxe_texture": iron_pickaxe_texture,
        "wood_sword_texture": wood_sword_texture,
        "stone_sword_texture": stone_sword_texture,
        "iron_sword_texture": iron_sword_texture,
        "sapling_texture": sapling_texture,
        "zombie_texture": zombie_texture,
        "zombie_texture_alpha": zombie_texture_alpha,
        "cow_texture": cow_texture,
        "cow_texture_alpha": cow_texture_alpha,
        "skeleton_texture": skeleton_texture,
        "skeleton_texture_alpha": skeleton_texture_alpha,
        "arrow_texture": arrow_texture,
        "arrow_texture_alpha": arrow_texture_alpha,
        "night_texture": night_texture,
        "night_noise_intensity_texture": night_noise_intensity_texture,
    }


def _is_valid_texture_cache(textures: dict) -> bool:
    for ts in (BLOCK_PIXEL_SIZE_AGENT, BLOCK_PIXEL_SIZE_IMG, BLOCK_PIXEL_SIZE_HUMAN):
        tex_shape = textures[ts]["full_map_block_textures"].shape
        if (
            tex_shape[0] != len(BlockType)
            or tex_shape[1] != OBS_DIM[0] * ts
            or tex_shape[2] != OBS_DIM[1] * ts
            or tex_shape[3] != 3
        ):
            return False
    return True


def _build_textures(theme: str) -> dict:
    global ACTIVE_TEXTURE_THEME, EXO_TEXTURE_MAPPING
    ACTIVE_TEXTURE_THEME = theme
    EXO_TEXTURE_MAPPING = _parse_exo_texture_mapping() if theme == "exo-planet" else {}
    cache_file = TEXTURE_CACHE_EXO_FILE if theme == "exo-planet" else TEXTURE_CACHE_FILE
    can_use_cache = os.path.exists(cache_file) and not os.environ.get(
        "CRAFTAX_RELOAD_TEXTURES", False
    )

    if can_use_cache:
        print(f"Loading textures ({theme}) from cache.")
        try:
            textures = load_compressed_pickle(cache_file)
        except Exception as exc:
            print(f"Texture cache ({theme}) unreadable ({exc}), reloading textures.")
            try:
                os.remove(cache_file)
            except OSError:
                pass
            textures = None
        if textures is not None and _is_valid_texture_cache(textures):
            print(f"Textures ({theme}) successfully loaded from cache.")
            return textures
        if textures is not None:
            print(f"Invalid texture cache ({theme}), reloading textures.")

    print(f"Processing textures ({theme}). This will be cached for future use.")
    textures = {
        BLOCK_PIXEL_SIZE_AGENT: load_all_textures(BLOCK_PIXEL_SIZE_AGENT),
        BLOCK_PIXEL_SIZE_IMG: load_all_textures(BLOCK_PIXEL_SIZE_IMG),
        BLOCK_PIXEL_SIZE_HUMAN: load_all_textures(BLOCK_PIXEL_SIZE_HUMAN),
    }
    save_compressed_pickle(cache_file, textures)
    print(f"Textures ({theme}) loaded and saved to cache.")
    return textures


def get_texture_bundle(theme: str) -> dict:
    normalized = "exo-planet" if str(theme).strip().lower() in {"exo", "exo-planet"} else "craftax"
    if normalized not in TEXTURES_BY_THEME:
        TEXTURES_BY_THEME[normalized] = _build_textures(normalized)
    return TEXTURES_BY_THEME[normalized]


def set_active_texture_theme(theme: str) -> dict:
    global TEXTURES, ACTIVE_TEXTURE_THEME
    normalized = "exo-planet" if str(theme).strip().lower() in {"exo", "exo-planet"} else "craftax"
    ACTIVE_TEXTURE_THEME = normalized
    TEXTURES = get_texture_bundle(normalized)
    return TEXTURES


TEXTURES = set_active_texture_theme(
    os.environ.get("PLAY_WEB_TEXTURE_THEME", "craftax")
)