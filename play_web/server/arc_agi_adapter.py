from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ARC_GAME_OPTIONS = [
    {
        "id": "ar25",
        "label": "ar25",
        "description": "Interactive ARC-AGI-3 puzzle with human-helper scoring on the ARC leaderboard.",
        "human_operator": True,
        "ai_operator": False,
        "instruction_mode": False,
        "companion_mode": True,
    },
    {
        "id": "bp35",
        "label": "bp35",
        "description": "Interactive ARC-AGI-3 puzzle with human-helper scoring on the ARC leaderboard.",
        "human_operator": True,
        "ai_operator": False,
        "instruction_mode": False,
        "companion_mode": True,
    },
    {
        "id": "lp85",
        "label": "lp85",
        "description": "Interactive puzzle suite with human-helper scoring on the ARC leaderboard.",
        "human_operator": True,
        "ai_operator": False,
        "instruction_mode": False,
        "companion_mode": True,
    },
    {
        "id": "ls20",
        "label": "ls20",
        "description": "Interactive puzzle suite with human-helper scoring on the ARC leaderboard.",
        "human_operator": True,
        "ai_operator": False,
        "instruction_mode": False,
        "companion_mode": True,
    },
]
SUPPORTED_ARC_GAME_IDS = tuple(option["id"] for option in ARC_GAME_OPTIONS)
ARC_ENVIRONMENTS_DIR = Path(__file__).resolve().parent.parent / "environment_files"
ARC_GAME_PREVIEW_ASSETS_DIR = Path(__file__).resolve().parent.parent / "client" / "assets" / "arc-games"
ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}


class ArcAgiUnavailableError(ValueError):
    pass


class ArcAgiGameUnavailableError(ValueError):
    pass


@dataclass
class ArcStepResult:
    reward: float
    done: bool
    frame: dict[str, Any]
    timing: dict[str, float] | None = None


def normalize_arc_game_id(game_id: str | None) -> str:
    normalized = str(game_id or "ls20").strip().lower()
    if normalized not in SUPPORTED_ARC_GAME_IDS:
        raise ValueError(
            f"Unsupported ARC-AGI-3 game '{game_id}'. "
            f"Supported games: {', '.join(SUPPORTED_ARC_GAME_IDS)}"
        )
    return normalized


def parse_arc_action_text(action_text: str) -> tuple[str, dict[str, int]]:
    text = str(action_text or "").strip().upper()
    match = re.search(r"\b(ACTION[1-7])\b", text)
    if not match:
        raise ValueError("ARC action must be one of ACTION1..ACTION7.")
    action_name = match.group(1)
    if action_name != "ACTION6":
        return action_name, {}

    tail = text[match.end() :]
    coords = re.findall(r"-?\d+", tail)
    if len(coords) < 2:
        raise ValueError("ACTION6 requires x and y coordinates.")
    x = int(coords[0])
    y = int(coords[1])
    if not (0 <= x <= 63 and 0 <= y <= 63):
        raise ValueError("ACTION6 coordinates must be in the 0..63 range.")
    return action_name, {"x": x, "y": y}


def _png_b64_from_pixels(pixels: Any) -> tuple[str, int, int] | None:
    if pixels is None:
        return None
    arr = np.asarray(pixels)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None
    if arr.dtype != np.uint8:
        max_value = float(np.max(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), image.width, image.height


def _arc_frame_array(frame_data: Any) -> np.ndarray | None:
    if frame_data is None:
        return None
    frames = frame_data if isinstance(frame_data, (list, tuple)) else [frame_data]
    if not frames:
        return None
    arr = np.asarray(frames[-1])
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr.astype(np.int16, copy=False)


def _png_b64_from_arc_frame(frame_data: Any, *, scale: int = 12) -> tuple[str, int, int] | None:
    frame = _arc_frame_array(frame_data)
    if frame is None:
        return None
    height, width = frame.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in ARC_COLOR_MAP.items():
        rgb[frame == value] = color
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    image = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), image.width, image.height


def _arc_frame_grid_text(frame_data: Any) -> str:
    frame = _arc_frame_array(frame_data)
    if frame is None:
        return ""
    clipped = np.clip(frame, 0, 15)
    rows = []
    for row in clipped:
        rows.append("".join(format(int(value), "x") for value in row))
    return "\n".join(rows)


def _text_png_b64(text: str, *, title: str = "ARC-AGI-3") -> tuple[str, int, int]:
    width, height = 768, 768
    image = Image.new("RGB", (width, height), (15, 17, 22))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, 54), fill=(35, 38, 46))
    draw.text((18, 18), title, fill=(232, 229, 224), font=font)
    y = 76
    line_height = 16
    for raw_line in str(text or "").splitlines()[:42]:
        line = raw_line[:112]
        draw.text((18, y), line, fill=(220, 224, 230), font=font)
        y += line_height
        if y > height - 30:
            break
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), width, height


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _coerce_state_name(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    return str(getattr(value, "name", value))


class ArcAgiAdapter:
    def __init__(
        self,
        game_id: str = "ls20",
        *,
        seed: int = 0,
        arcade_factory: Callable[[], Any] | None = None,
        game_action_cls: Any | None = None,
        game_state_cls: Any | None = None,
    ) -> None:
        self.game_id = normalize_arc_game_id(game_id)
        self.seed = int(seed)
        self._arcade_factory = arcade_factory
        self._game_action_cls = game_action_cls
        self._game_state_cls = game_state_cls
        self.arcade: Any = None
        self.env: Any = None
        self.current_obs: Any = None
        self._init_env()

    def _load_sdk(self) -> tuple[Callable[[], Any], Any, Any]:
        if self._arcade_factory is not None and self._game_action_cls is not None:
            return self._arcade_factory, self._game_action_cls, self._game_state_cls
        try:
            import arc_agi  # type: ignore
            from arc_agi import OperationMode  # type: ignore
            try:
                from arc_agi import GameAction, GameState  # type: ignore
            except Exception:
                from arcengine import GameAction, GameState  # type: ignore
        except Exception as exc:
            raise ArcAgiUnavailableError(
                "arc-agi is not installed. Install project requirements with "
                "`pip install -r requirements.txt`."
            ) from exc

        def _factory() -> Any:
            return arc_agi.Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(ARC_ENVIRONMENTS_DIR),
            )

        return _factory, GameAction, GameState

    def _init_env(self) -> None:
        arcade_factory, game_action_cls, game_state_cls = self._load_sdk()
        self._game_action_cls = game_action_cls
        self._game_state_cls = game_state_cls
        self.arcade = arcade_factory()
        try:
            self.env = self.arcade.make(self.game_id, seed=self.seed)
        except Exception as exc:
            raise ArcAgiGameUnavailableError(
                f"ARC-AGI-3 game '{self.game_id}' is not available locally. "
                "Check that arc-agi local environment files are installed and accessible."
            ) from exc
        if self.env is None:
            raise ArcAgiGameUnavailableError(
                f"ARC-AGI-3 game '{self.game_id}' is not available locally. "
                "Check that arc-agi local environment files are installed and accessible."
            )
        self.current_obs = self.env.reset()

    def reset(self) -> dict[str, Any]:
        self.current_obs = self.env.reset()
        return self.render_frame()

    def available_actions(self) -> list[str]:
        actions = getattr(self.env, "action_space", None) or []
        names: list[str] = []
        for action in actions:
            name = str(getattr(action, "name", action)).strip()
            if name:
                names.append(name)
        return names

    def observation_text(self) -> str:
        try:
            from megaprompt.obs.arc_grid import render as render_arc_grid  # type: ignore

            return render_arc_grid(self.observation_payload(include_image=False))
        except Exception:
            obs = self.current_obs
            info = getattr(self.env, "info", None)
            title = _first_attr(info, ("title",)) or self.game_id
            state = _coerce_state_name(_first_attr(obs, ("state",)))
            levels_completed = _first_attr(obs, ("levels_completed",)) or 0
            return "\n".join(
                [
                    f"Game: {self.game_id} ({title})",
                    f"State: {state}",
                    f"Levels completed: {levels_completed}",
                    f"Available actions: {', '.join(self.available_actions()) or 'none'}",
                ]
            )

    def _rendered_png(self) -> tuple[str, int, int]:
        obs = self.current_obs
        rendered = _png_b64_from_arc_frame(_first_attr(obs, ("frame",)))
        if rendered is None:
            rendered = _png_b64_from_pixels(self._pixels_from_obs())
        if rendered is None:
            title = f"ARC-AGI-3 {self.game_id}"
            rendered = _text_png_b64(self._metadata_text(), title=title)
        return rendered

    def _metadata_text(self) -> str:
        obs = self.current_obs
        info = getattr(self.env, "info", None)
        title = _first_attr(info, ("title",)) or self.game_id
        state = _coerce_state_name(_first_attr(obs, ("state",)))
        levels_completed = _first_attr(obs, ("levels_completed",)) or 0
        return "\n".join(
            [
                f"Game: {self.game_id} ({title})",
                f"State: {state}",
                f"Levels completed: {levels_completed}",
                f"Available actions: {', '.join(self.available_actions()) or 'none'}",
            ]
        )

    def observation_payload(self, *, include_image: bool = True) -> dict[str, Any]:
        obs = self.current_obs
        info = getattr(self.env, "info", None)
        title = _first_attr(info, ("title",)) or self.game_id
        state = _coerce_state_name(_first_attr(obs, ("state",)))
        levels_completed = _first_attr(obs, ("levels_completed",)) or 0
        png_b64 = ""
        width = 0
        height = 0
        if include_image:
            png_b64, width, height = self._rendered_png()
        return {
            "game_id": self.game_id,
            "title": title,
            "state": state,
            "levels_completed": int(levels_completed),
            "available_actions": self.available_actions(),
            "frame_grid": _arc_frame_grid_text(_first_attr(obs, ("frame",))),
            "png_b64": png_b64,
            "w": int(width),
            "h": int(height),
        }

    def _pixels_from_obs(self) -> Any:
        obs = self.current_obs
        for name in ("pixels", "image", "rgb", "screen", "observation"):
            value = _first_attr(obs, (name,))
            if value is not None:
                return value
        return None

    def render_frame(self) -> dict[str, Any]:
        obs = self.current_obs
        png_b64, width, height = self._rendered_png()
        state = _coerce_state_name(_first_attr(obs, ("state",)))
        levels_completed = _first_attr(obs, ("levels_completed",)) or 0
        frame_grid = _arc_frame_grid_text(_first_attr(obs, ("frame",)))
        return {
            "w": int(width),
            "h": int(height),
            "png_b64": png_b64,
            "agent_observation": self.observation_text(),
            "arc": {
                "game_id": self.game_id,
                "state": state,
                "levels_completed": int(levels_completed),
                "available_actions": self.available_actions(),
                "frame_grid": frame_grid,
            },
        }

    def step(self, action_text: str, *, reasoning: str = "") -> ArcStepResult:
        action_name, data = parse_arc_action_text(action_text)
        action = getattr(self._game_action_cls, action_name)
        kwargs: dict[str, Any] = {}
        if data:
            kwargs["data"] = data
        if reasoning:
            kwargs["reasoning"] = {"thought": reasoning}
        env_started = time.perf_counter()
        self.current_obs = self.env.step(action, **kwargs)
        env_ms = round((time.perf_counter() - env_started) * 1000, 1)
        state = _coerce_state_name(_first_attr(self.current_obs, ("state",)))
        done_states = {"WIN", "GAME_OVER", "DONE", "LOSE", "LOST"}
        done = state.upper() in done_states
        render_started = time.perf_counter()
        frame = self.render_frame()
        render_ms = round((time.perf_counter() - render_started) * 1000, 1)
        return ArcStepResult(
            reward=0.0,
            done=done,
            frame=frame,
            timing={"env_step_ms": env_ms, "render_ms": render_ms},
        )


def arc_game_option_meta(game_id: str) -> dict[str, Any]:
    normalized = normalize_arc_game_id(game_id)
    for option in ARC_GAME_OPTIONS:
        if option["id"] == normalized:
            return dict(option)
    raise ValueError(f"Unknown ARC game '{game_id}'")


def _read_static_arc_game_preview_png(game_id: str) -> tuple[str, int, int] | None:
    path = ARC_GAME_PREVIEW_ASSETS_DIR / f"{normalize_arc_game_id(game_id)}.png"
    if not path.is_file():
        return None
    data = path.read_bytes()
    if not data:
        return None
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
    return base64.b64encode(data).decode("ascii"), width, height


def get_arc_game_preview(game_id: str) -> dict[str, Any]:
    meta = arc_game_option_meta(game_id)
    static = _read_static_arc_game_preview_png(meta["id"])
    if static is not None:
        png_b64, width, height = static
        return {
            **meta,
            "game_id": meta["id"],
            "png_b64": png_b64,
            "w": width,
            "h": height,
        }
    adapter = ArcAgiAdapter(meta["id"])
    frame = adapter.render_frame()
    return {
        **meta,
        "game_id": meta["id"],
        "png_b64": str(frame.get("png_b64") or ""),
        "w": int(frame.get("w") or 0),
        "h": int(frame.get("h") or 0),
    }
