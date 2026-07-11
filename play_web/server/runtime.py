import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import jax
import numpy as np
from PIL import Image

from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN, Action
from craftax.craftax_env import make_craftax_env_from_name
from external_visualization.render import make_render_craftax_pixels_jit, make_render_world_jit
from oracle.active_agent_base import ActiveAgent
from oracle.config_loader import load_config
from oracle.configs import GenConfig, OracleConfig

from oracle.intent import Intent
from oracle.openrouter_client import fetch_json
from oracle.oracle import Oracle
from oracle.knowledge import clear_episode_notes, strip_to_database_tags
from oracle.prompts.prompt_generation import (
    build_previous_actions_analysis,
    coerce_megaprompt_config_for_world_mode,
    generate_arc_agent_prompt,
    generate_agent_prompt,
    normalize_world_mode,
)
from oracle.utils.observation_formatting import format_inventory_from_env_state
from oracle.statistics_wrapper import ActiveAgentStatistics

from .active_agent_helpers import (
    format_action_for_ui,
    format_agent_observation_text_from_state,
    parse_agent_answer,
)
from .arc_agi_adapter import ARC_GAME_OPTIONS, ArcAgiAdapter, normalize_arc_game_id
from .campaign_mode import CampaignState, campaign_total_levels, detect_new_episode_achievements
from .deployment_operator_limits import format_operator_call_budget_text, is_deployment_megaprompt
from .env_file import api_keys_status as env_api_keys_status, get_api_secret
from .leaderboard import (
    TALKINGHEADS_PLAYER_NAME,
    append_arc_human_result,
    append_leaderboard_entry,
    utc_now_iso,
)
from .trajectory_logger import TrajectoryLogger
from .world_frame import build_world_payload


logger = logging.getLogger(__name__)

DEFAULT_AGENT_GOAL = "Collect stone"
EXO_PLANET_DEFAULT_AGENT_GOAL = "Collect Biomass"
ARC_DEFAULT_AGENT_GOAL = "Solve the ARC-AGI-3 game efficiently"
GAME_KIND_CRAFTAX = "craftax"
GAME_KIND_ARC_AGI = "arc_agi"
DEFAULT_ARC_GAME_ID = "ls20"
_IMAGE_MARKER_RE = re.compile(
    r"\[\[image:data:image/(?:png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=\r\n]+\]\]",
    re.IGNORECASE,
)


def default_agent_goal_for_texture_theme(theme: str) -> str:
    normalized = "exo-planet" if str(theme).strip().lower() in {"exo", "exo-planet"} else "craftax"
    return EXO_PLANET_DEFAULT_AGENT_GOAL if normalized == "exo-planet" else DEFAULT_AGENT_GOAL


DEFAULT_AGENT_SYSTEM_MESSAGE = "You are the game-playing agent described in the prompt."
ACTIVE_AGENT_MODEL = os.environ.get(
    "ACTIVE_AGENT_MODEL", "qwen/qwen3-next-80b-a3b-instruct"
)
ACTIVE_AGENT_MODE = os.environ.get("ACTIVE_AGENT_MODE", "openrouter").strip().lower()
ACTIVE_AGENT_MAX_NEW_TOKENS = int(os.environ.get("ACTIVE_AGENT_MAX_NEW_TOKENS", "2048"))
ACTIVE_AGENT_DO_SAMPLE = os.environ.get("ACTIVE_AGENT_DO_SAMPLE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ACTIVE_AGENT_TEMPERATURE = float(os.environ.get("ACTIVE_AGENT_TEMPERATURE", "1.0"))
ACTIVE_AGENT_TOP_P = float(os.environ.get("ACTIVE_AGENT_TOP_P", "0.9"))
DEFAULT_INTERACTION_MODE = "oracle"
DEFAULT_TEXTURE_THEME = os.environ.get("PLAY_WEB_DEFAULT_TEXTURE_THEME", "craftax")
DEFAULT_MEGAPROMPT_CONFIG_NAME = "database_formulation"
DEFAULT_FORCED_EXPERT = Intent.GOAL_EXPERT
DEFAULT_ALLOWED_EXPERTS = [Intent.GOAL_EXPERT]
ALL_EXPERTS = list(Intent.ALL_EXPERTS)
ENV_NAME = "Craftax-Classic-Symbolic-v1"


_MODEL_ERROR_HINTS = (
    "is not a valid model",
    "not a valid model id",
    "model_not_found",
    "model not found",
    "no endpoints found",
    "no allowed providers",
    "does not exist",
    "invalid model",
    "unknown model",
)


def describe_model_not_found_error(exc: BaseException) -> str | None:
    """
    Return a short human-readable reason if ``exc`` looks like an
    invalid/unknown model error from the provider (OpenRouter), else None.

    OpenRouter rejects unknown model ids with 400/404 responses whose message
    mentions the model. We match both the known phrasings and generic
    400/404 + "model" responses so the UI can show an actionable message.
    """
    text = (str(exc) or "").strip()
    low = text.lower()
    if any(hint in low for hint in _MODEL_ERROR_HINTS):
        return text
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status in (400, 404) and "model" in low:
        return text
    return None


_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_MODELS_CACHE: dict[str, Any] = {"ts": 0.0, "ids": frozenset()}
_OPENROUTER_MODELS_TTL = 300.0


def fetch_openrouter_model_ids(api_key: str | None = None) -> frozenset[str]:
    """
    Return the set of available OpenRouter model ids (lowercased), cached for a
    few minutes. Best-effort: returns an empty set if the list can't be fetched,
    so callers can treat "unknown" as "do not warn".
    """
    now = time.time()
    cached = _OPENROUTER_MODELS_CACHE
    if cached["ids"] and now - cached["ts"] < _OPENROUTER_MODELS_TTL:
        return cached["ids"]
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = fetch_json(_OPENROUTER_MODELS_URL, headers=headers, timeout=8)
        ids = frozenset(
            str(item.get("id", "")).strip().lower()
            for item in payload.get("data", [])
            if item.get("id")
        )
        if ids:
            _OPENROUTER_MODELS_CACHE.update(ts=now, ids=ids)
            return ids
    except Exception:
        logger.warning("Could not fetch OpenRouter model list", exc_info=True)
    return cached["ids"]


def validate_openrouter_models(
    model_ids: list[str], api_key: str | None = None
) -> dict[str, Any]:
    """
    Check the given model ids against OpenRouter's catalog.

    Returns ``{"checked": bool, "invalid": [...]}``. ``checked`` is False when
    the catalog could not be fetched (so the UI should not show a false warning).
    """
    available = fetch_openrouter_model_ids(api_key)
    if not available:
        return {"checked": False, "invalid": []}
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in model_ids:
        name = str(raw or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        if name.lower() not in available:
            invalid.append(name)
    return {"checked": True, "invalid": invalid}


def resolve_max_agent_steps_per_tick(env_params: Any) -> int:
    """Upper bound for agent ticks per UI click (Craftax env or PLAY_WEB override)."""
    override = os.environ.get("PLAY_WEB_MAX_AGENT_STEPS_PER_TICK", "").strip()
    if override:
        return max(1, int(override))
    return max(1, int(getattr(env_params, "max_timesteps", 1)))


LONG_REASONING_NO_ACTION_FEEDBACK = (
    "Your reasoning took too long and no executable action was detected. "
    "Send your next action only — no further reasoning — as your next message."
)

INVALID_FORMAT_FEEDBACK = (
    "Your last message was not in the required format (no action or question was detected). "
    "Send your next action only — use the required format and no extra reasoning."
)

LOG_DIR = Path(__file__).resolve().parent.parent / "answer"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _agent_ws_knowledge_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if parsed.get("knowledge_updated"):
        fields["knowledge_updated"] = True
    if parsed.get("to_database"):
        fields["to_database"] = parsed["to_database"]
    return fields


def _pixels_to_png_base64(pixels_u8: np.ndarray) -> str:
    img = Image.fromarray(pixels_u8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


EXPERT_CONFIG_KEYS: dict[str, tuple[str, str]] = {
    "map_expert": ("map_expert_model_path", "map_expert_mode"),
    "mechanics_expert": ("mechanics_expert_model_path", "mechanics_expert_mode"),
    "action_expert": ("action_expert_model_path", "action_expert_mode"),
    "question_expert": ("question_expert_model_path", "question_expert_mode"),
    "goal_expert": ("goal_expert_model_path", "goal_expert_mode"),
    "path_expert_helper": ("path_expert_helper_model_path", "path_expert_helper_mode"),
    "path_expert": ("path_expert_model_path", "path_expert_mode"),
}


@dataclass
class Session:
    env: Any
    env_params: Any
    rng: jax.Array
    state: Any
    step_fn: Any
    render_fn: Any
    render_world_base_fn: Any = None
    render_obs_overlay_fn: Any = None
    _last_map_grid: Optional[np.ndarray] = field(default=None, repr=False)
    # Incremented whenever the cached grid is dropped; lets the client detect
    # that its base map belongs to an older world and request a full snapshot.
    map_epoch: int = 0
    oracle: Optional[Oracle] = None
    oracle_config_data: Optional[OracleConfig] = None
    active_agent: Optional[ActiveAgent] = None
    agent_gen_config: Optional[GenConfig] = None
    oracle_chat_history: list[dict[str, Any]] = field(default_factory=list)
    agent_goal: str = DEFAULT_AGENT_GOAL
    hints: list[str] = field(default_factory=list)
    stop_agent_requested: bool = False
    agent_tick_task: Any = field(default=None, repr=False)
    agent_tick_inline: bool = field(default=False, repr=False)
    trajectory_logger: Optional[TrajectoryLogger] = None
    save_trajectory_enabled: bool = False
    interaction_mode: str = DEFAULT_INTERACTION_MODE
    player_name: str = ""
    player_avatar_id: int = 0
    allowed_experts: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_EXPERTS))
    forced_expert: Optional[str] = DEFAULT_FORCED_EXPERT
    pending_agent_question: str = ""
    pending_agent_question_step: int = 0
    pending_human_tick: Optional[dict[str, Any]] = None
    last_agent_user_prompt: str = ""
    last_agent_system_message: str = ""
    default_agent_steps: int = 20
    max_agent_steps_per_tick: int = 1
    last_step_debug_timing: dict[str, float] = field(default_factory=dict)
    megaprompt_config_name: str = DEFAULT_MEGAPROMPT_CONFIG_NAME
    arc_prompt_extra: str = ""
    active_agent_model: str = ACTIVE_AGENT_MODEL
    active_agent_mode: str = ACTIVE_AGENT_MODE
    MAX_OPERATOR_MESSAGES_FOR_AGENT = 6
    # Last primitive env step only: state before, state after, action (for MegaPrompt state_history).
    last_step_transition: Optional[dict[str, Any]] = None
    # Tick turns: observation + templated prompt + raw model response (for direct agent chat history).
    agent_tick_history: list[dict[str, str]] = field(default_factory=list)
    agent_direct_chat_active: bool = False
    agent_direct_chat_turns: list[dict[str, str]] = field(default_factory=list)
    campaign_state: CampaignState = field(default_factory=CampaignState)
    leaderboard_attempt_active: bool = False
    leaderboard_attempt_started_at: str = ""
    leaderboard_phase1_questions: int = 0
    leaderboard_phase2_questions: int = 0
    leaderboard_last_checkpoint_signature: tuple[int, int, tuple[str, ...]] | None = None
    leaderboard_level_steps: dict[str, int] = field(default_factory=dict)
    leaderboard_current_level_steps: int = 0
    leaderboard_total_steps: int = 0
    companion_research_active: bool = False
    companion_research_task_index: int = 0
    companion_research_task_ticks: int = 0
    companion_research_task_max_ticks: int = 0
    texture_theme: str = DEFAULT_TEXTURE_THEME
    game_kind: str = GAME_KIND_CRAFTAX
    arc_game_id: str = DEFAULT_ARC_GAME_ID
    arc_adapter: Optional[ArcAgiAdapter] = None
    arc_attempt_started_at: float = field(default_factory=time.time)
    arc_actions_count: int = 0
    arc_agent_actions_count: int = 0
    arc_manual_actions_count: int = 0
    arc_questions_count: int = 0
    arc_human_answers_count: int = 0
    arc_final_score: Optional[dict[str, Any]] = None
    arc_score_submitted: bool = False
    arc_leaderboard_last_levels_submitted: int = 0
    arc_action_history: list[str] = field(default_factory=list)
    arc_step_index: int = 0
    arc_previous_reasoning: str = ""
    arc_previous_observation_payload: Optional[dict[str, Any]] = None
    hf_token_session: str = ""
    openrouter_api_key_session: str = ""
    operator_call_limit: int | None = None
    episode_initial_state: Any = None
    episode_discovered_keys: set[str] = field(default_factory=set, repr=False)
    # Browser tab session id — namespaces per-user agent knowledge files.
    play_session_id: str = ""

    def agent_tick_is_running(self) -> bool:
        task = self.agent_tick_task
        if task is not None and not task.done():
            return True
        return bool(self.agent_tick_inline)

    def should_skip_oracle_autoreply(self) -> bool:
        """Skip blocking oracle calls once stop was requested."""
        return bool(self.stop_agent_requested)

    def attach_agent_tick_task(self, task: Any) -> None:
        self.agent_tick_task = task

    def detach_agent_tick_task(self, task: Any) -> None:
        if self.agent_tick_task is task:
            self.agent_tick_task = None

    async def cancel_agent_tick_task(self) -> None:
        task = self.agent_tick_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Agent tick task failed while cancelling")

    def set_api_secrets(
        self,
        *,
        hf_token: str | None = None,
        openrouter_api_key: str | None = None,
    ) -> None:
        if hf_token is not None:
            self.hf_token_session = str(hf_token).strip()
        if openrouter_api_key is not None:
            self.openrouter_api_key_session = str(openrouter_api_key).strip()
        self.oracle = None
        self.active_agent = None

    def clear_api_secrets(self) -> None:
        self.hf_token_session = ""
        self.openrouter_api_key_session = ""
        self.oracle = None
        self.active_agent = None

    def api_keys_status(self) -> dict[str, object]:
        return env_api_keys_status(
            hf_token_override=self.hf_token_session or None,
            openrouter_api_key_override=self.openrouter_api_key_session or None,
        )

    def get_api_secret(self, key: str) -> str:
        if key == "HF_TOKEN":
            return self.hf_token_session or get_api_secret("HF_TOKEN")
        if key == "OPENROUTER_API_KEY":
            return self.openrouter_api_key_session or get_api_secret("OPENROUTER_API_KEY")
        raise ValueError(f"Unsupported env key: {key}")

    def validate_active_models(self) -> dict[str, Any]:
        """
        Best-effort check that the configured OpenRouter models exist in the
        OpenRouter catalog. Returns ``{"checked": bool, "invalid": [...]}``.
        """
        mode = str(self.active_agent_mode or "openrouter").strip().lower()
        if mode != "openrouter":
            return {"checked": False, "invalid": []}
        candidates = [self.active_agent_model, *self.get_expert_models().values()]
        return validate_openrouter_models(
            candidates, api_key=self.get_api_secret("OPENROUTER_API_KEY") or None
        )

    def get_oracle_config(self) -> OracleConfig:
        if self.oracle_config_data is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "oracle_config.yaml"
            self.oracle_config_data = load_config(path=str(config_path))
        return self.oracle_config_data

    def _apply_session_api_keys_to_oracle_config(self, cfg: OracleConfig) -> None:
        cfg.hub_api_key = self.get_api_secret("HF_TOKEN") or None
        cfg.openrouter_api_key = self.get_api_secret("OPENROUTER_API_KEY") or None

    @staticmethod
    def _missing_oracle_api_keys(cfg: OracleConfig, get_secret) -> list[str]:
        required: set[str] = set()
        for mode in (
            cfg.map_expert_mode,
            cfg.mechanics_expert_mode,
            cfg.action_expert_mode,
            cfg.question_expert_mode,
            cfg.goal_expert_mode,
            cfg.path_expert_helper_mode,
            cfg.path_expert_mode,
        ):
            if mode == "hub":
                required.add("HF_TOKEN")
            elif mode == "openrouter":
                required.add("OPENROUTER_API_KEY")
        return [key for key in sorted(required) if not get_secret(key)]

    def ensure_oracle(self) -> Oracle:
        cfg = self.get_oracle_config()
        self._apply_session_api_keys_to_oracle_config(cfg)
        missing = self._missing_oracle_api_keys(cfg, self.get_api_secret)
        if missing:
            labels = ", ".join("HF token" if k == "HF_TOKEN" else "OpenRouter API key" for k in missing)
            raise ValueError(f"Configure API keys in settings: {labels}")
        if self.oracle is None:
            self.oracle = Oracle(cfg, world_mode=self.texture_theme)
            return self.oracle
        oracle_mode = normalize_world_mode(getattr(self.oracle, "world_mode", None))
        if oracle_mode != normalize_world_mode(self.texture_theme):
            self.oracle = Oracle(cfg, world_mode=self.texture_theme)
        return self.oracle

    def rebuild_oracle(self, cfg: Optional[OracleConfig] = None) -> None:
        if cfg is not None:
            self.oracle_config_data = cfg
        self.oracle = None
        missing = self._missing_oracle_api_keys(self.get_oracle_config(), self.get_api_secret)
        if not missing:
            self.ensure_oracle()

    def _campaign_world_mode(self) -> str:
        if self.is_arc_game():
            return "arc_agi"
        return self.texture_theme

    def get_companion_research_snapshot(self) -> dict[str, Any]:
        from .campaign_mode import campaign_tasks_for_world_mode

        tasks = campaign_tasks_for_world_mode(self._campaign_world_mode())
        total_tasks = len(tasks)
        if not self.companion_research_active or total_tasks == 0:
            return {"active": False}
        # Step budget applies to the whole run; ticks are counted across all levels.
        total_budget = max(1, int(self.companion_research_task_max_ticks or 1))
        used = max(0, int(self.companion_research_task_ticks))
        task_idx = max(0, min(int(self.companion_research_task_index or 1) - 1, total_tasks - 1))
        progress_pct = int(round(100.0 * min(used, total_budget) / total_budget))
        task = tasks[task_idx]
        return {
            "active": True,
            "task_index": task_idx + 1,
            "task_total": total_tasks,
            "task_key": task.key,
            "task_title": task.title,
            "task_ticks": used,
            "max_ticks_per_task": total_budget,
            "progress_pct": progress_pct,
            "complete": self.campaign_state.is_finished(),
        }

    def get_agent_observation(self) -> str:
        if self.is_arc_game():
            return self.ensure_arc_adapter().observation_text()
        return format_agent_observation_text_from_state(
            self.state,
            world_mode=self.texture_theme,
        )

    def is_arc_game(self) -> bool:
        return self.game_kind == GAME_KIND_ARC_AGI

    def ensure_arc_adapter(self) -> ArcAgiAdapter:
        if self.arc_adapter is None or self.arc_adapter.game_id != self.arc_game_id:
            self.arc_adapter = ArcAgiAdapter(self.arc_game_id)
        self.state = self.arc_adapter.current_obs
        return self.arc_adapter

    def get_arc_game_options(self) -> list[dict[str, str]]:
        from .arc_env_sync import arc_game_options_with_availability
        from .features import filter_arc_game_options

        return filter_arc_game_options(arc_game_options_with_availability())

    def reset_arc_attempt_stats(self) -> None:
        self.arc_attempt_started_at = time.time()
        self.arc_actions_count = 0
        self.arc_agent_actions_count = 0
        self.arc_manual_actions_count = 0
        self.arc_questions_count = 0
        self.arc_human_answers_count = 0
        self.arc_final_score = None
        self.arc_score_submitted = False
        self.arc_leaderboard_last_levels_submitted = 0
        self.arc_action_history.clear()
        self.arc_step_index = 0
        self.arc_previous_reasoning = ""
        self.arc_previous_observation_payload = None

    def next_arc_step_number(self) -> int:
        return max(1, int(self.arc_step_index or 0) + 1)

    def current_arc_step_number(self) -> int:
        return max(1, int(self.arc_step_index or 0))

    def _record_arc_decision_step(self) -> int:
        self.arc_step_index = max(0, int(self.arc_step_index or 0)) + 1
        return self.arc_step_index

    def _record_arc_action_history(self, action: Any) -> None:
        action_text = " ".join(str(action or "").strip().upper().split())
        if action_text:
            self.arc_action_history.append(action_text)

    def _capture_arc_previous_observation(self, adapter: ArcAgiAdapter | None = None) -> None:
        if not self.is_arc_game():
            return
        adapter = adapter or self.ensure_arc_adapter()
        try:
            self.arc_previous_observation_payload = adapter.observation_payload(include_image=True)
        except Exception:
            logger.exception("Failed to capture previous ARC observation")
            self.arc_previous_observation_payload = None

    def _arc_observation_payload_for_prompt(self, adapter: ArcAgiAdapter | None = None) -> dict[str, Any]:
        adapter = adapter or self.ensure_arc_adapter()
        payload = adapter.observation_payload(include_image=True)
        previous = self.arc_previous_observation_payload or {}
        if previous:
            payload.update(
                {
                    "previous_png_b64": str(previous.get("png_b64") or ""),
                    "previous_w": int(previous.get("w") or 0),
                    "previous_h": int(previous.get("h") or 0),
                    "previous_state": str(previous.get("state") or ""),
                    "previous_levels_completed": int(previous.get("levels_completed") or 0),
                    "previous_action": self.arc_action_history[-1] if self.arc_action_history else "",
                }
            )
        return payload

    def _record_arc_previous_reasoning(self, parsed: dict[str, Any]) -> None:
        if not self.is_arc_game():
            return
        self.arc_previous_reasoning = " ".join(
            str(parsed.get("reasoning") or "").strip().split()
        )

    _ARC_TERMINAL_STATES = frozenset({"WIN", "GAME_OVER", "DONE", "LOSE", "LOST"})

    def _arc_keeps_level_progression(self) -> bool:
        from .features import arc_multi_level_progression

        return arc_multi_level_progression() or self.campaign_state.enabled

    @classmethod
    def _arc_terminal_state(cls, score_or_frame: dict[str, Any] | None) -> bool:
        data = score_or_frame if isinstance(score_or_frame, dict) else {}
        arc = data.get("arc") if isinstance(data.get("arc"), dict) else data
        state = str(arc.get("state") or "").strip().upper()
        return state in cls._ARC_TERMINAL_STATES

    def _arc_score_is_submittable(self, score_or_frame: dict[str, Any] | None) -> bool:
        data = score_or_frame if isinstance(score_or_frame, dict) else {}
        arc = data.get("arc") if isinstance(data.get("arc"), dict) else data
        levels_completed = int(arc.get("levels_completed") or 0)
        return levels_completed >= 1 or self._arc_terminal_state(score_or_frame)

    def _arc_score_is_final(self, score_or_frame: dict[str, Any] | None) -> bool:
        if self._arc_terminal_state(score_or_frame):
            return True
        if self._arc_keeps_level_progression():
            return False
        data = score_or_frame if isinstance(score_or_frame, dict) else {}
        arc = data.get("arc") if isinstance(data.get("arc"), dict) else data
        levels_completed = int(arc.get("levels_completed") or 0)
        return levels_completed >= 1

    def _arc_episode_should_end(self, result: Any) -> bool:
        if bool(getattr(result, "done", False)):
            return True
        if self._arc_keeps_level_progression():
            return False
        return self._arc_score_is_final(getattr(result, "frame", None))

    def build_arc_human_score(self, frame: dict[str, Any] | None = None) -> dict[str, Any]:
        arc = (frame or {}).get("arc") if isinstance(frame, dict) else None
        arc = arc if isinstance(arc, dict) else {}
        state = str(arc.get("state") or "UNKNOWN").strip() or "UNKNOWN"
        levels_completed = int(arc.get("levels_completed") or 0)
        elapsed = max(0, int(time.time() - float(self.arc_attempt_started_at or time.time())))
        success = levels_completed >= 1 or state.upper() == "WIN"
        final = self._arc_score_is_final(arc)
        base = 10000 if success else 1000
        score = (
            base
            + levels_completed * 500
            - self.arc_actions_count * 40
            - self.arc_questions_count * 200
            - self.arc_human_answers_count * 100
            - self.arc_manual_actions_count * 250
            - elapsed
        )
        return {
            "available": self.is_arc_game(),
            "submitted": bool(self.arc_score_submitted),
            "game_id": self.arc_game_id,
            "state": state,
            "success": success,
            "final": final,
            "levels_completed": levels_completed,
            "score": max(0, int(score)),
            "actions": int(self.arc_actions_count),
            "agent_actions": int(self.arc_agent_actions_count),
            "manual_actions": int(self.arc_manual_actions_count),
            "questions": int(self.arc_questions_count),
            "human_answers": int(self.arc_human_answers_count),
            "elapsed_seconds": elapsed,
            "active_agent_model": self.active_agent_model,
            "megaprompt_config_name": self.megaprompt_config_name,
        }

    def mark_arc_finished(self, frame: dict[str, Any]) -> None:
        if not self.is_arc_game():
            return
        self.arc_final_score = self.build_arc_human_score(frame)
        self._maybe_auto_submit_arc_human_score(frame)

    def _arc_leaderboard_player_name(self) -> str:
        player_name = str(self.player_name or "").strip()
        if player_name:
            return player_name[:40]
        if self.interaction_mode == "oracle":
            return TALKINGHEADS_PLAYER_NAME
        return ""

    def _maybe_auto_submit_arc_human_score(self, frame: dict[str, Any] | None) -> bool:
        """Append ARC human leaderboard checkpoints as soon as a level is solved."""
        if not self.is_arc_game():
            return False
        score = self.build_arc_human_score(frame)
        levels_completed = int(score.get("levels_completed") or 0)
        if levels_completed <= 0:
            return False
        if levels_completed <= int(self.arc_leaderboard_last_levels_submitted or 0):
            return False
        player_name = self._arc_leaderboard_player_name()
        if not player_name:
            # Keep the existing modal flow for anonymous human players.
            self.arc_final_score = score
            return False
        try:
            entry = append_arc_human_result(
                {
                    **score,
                    "player_name": player_name,
                    "player_avatar_id": self.player_avatar_id,
                }
            )
        except Exception:
            logger.exception("Failed to append ARC human leaderboard checkpoint")
            return False
        self.arc_leaderboard_last_levels_submitted = levels_completed
        self.arc_score_submitted = True
        self.arc_final_score = {**score, **entry, "submitted": True}
        return True

    def _sync_arc_campaign_after_step(self) -> None:
        if not self.is_arc_game() or not self.campaign_state.enabled:
            return
        if self.campaign_state.maybe_advance(self.state):
            self._record_completed_level_steps()
            self._checkpoint_leaderboard_attempt("level_complete")

    def _reset_arc_episode_after_single_level(self) -> dict[str, Any]:
        """
        ARC-AGI-3 SDK games may advance to the next level after a solve. TalkingHeads
        treats each ARC run as a single-level episode, so hide that transition from
        the UI/agent by immediately resetting the same game while preserving the
        completed attempt score for leaderboard submission.
        """
        adapter = self.ensure_arc_adapter()
        completed_score = self.arc_final_score.copy() if isinstance(self.arc_final_score, dict) else self.arc_final_score
        score_submitted = bool(self.arc_score_submitted)
        self.reset_arc_attempt_stats()
        self.arc_final_score = completed_score
        self.arc_score_submitted = score_submitted
        self.oracle_chat_history.clear()
        self.hints.clear()
        self.pending_agent_question = ""
        self.pending_agent_question_step = 0
        self.pending_human_tick = None
        self.last_agent_user_prompt = ""
        self.last_agent_system_message = ""
        self.agent_tick_history.clear()
        self.agent_direct_chat_turns.clear()
        self.agent_direct_chat_active = False
        try:
            frame = adapter.reset()
            self.state = adapter.current_obs
            arc = frame.get("arc") if isinstance(frame, dict) else None
            if isinstance(arc, dict):
                arc["single_level_reset"] = True
            return frame
        except Exception:
            logger.exception("Failed to reset ARC episode after single-level completion")
            return self.render_frame()

    def set_game_kind(self, game_kind: str, *, arc_game_id: str | None = None) -> bool:
        normalized = str(game_kind or GAME_KIND_CRAFTAX).strip().lower()
        if normalized in {"arc", "arc-agi", "arc_agi", "arc-agi-3", "arc_agi_3"}:
            normalized = GAME_KIND_ARC_AGI
        elif normalized in {"craftax", "craftext", ""}:
            normalized = GAME_KIND_CRAFTAX
        else:
            raise ValueError("game_kind must be 'craftax' or 'arc_agi'")

        if normalized == GAME_KIND_ARC_AGI:
            from .features import assert_arc_game_allowed_for_profile

            next_arc_game_id = assert_arc_game_allowed_for_profile(arc_game_id or self.arc_game_id)
            changed = self.game_kind != GAME_KIND_ARC_AGI or next_arc_game_id != self.arc_game_id
            if not changed:
                self.interaction_mode = "human"
                self.forced_expert = None
                self.allowed_experts = []
                self._align_megaprompt_to_texture_theme()
                return False
            self.game_kind = GAME_KIND_ARC_AGI
            self.arc_game_id = next_arc_game_id
            self._align_megaprompt_to_texture_theme()
            self.interaction_mode = "human"
            self.forced_expert = None
            self.allowed_experts = []
            self.campaign_state.set_world_mode("arc_agi")
            self.campaign_state.set_enabled(False, self.state)
            self.companion_research_active = False
            self.oracle = None
            self.arc_adapter = ArcAgiAdapter(self.arc_game_id)
            self.state = self.arc_adapter.current_obs
            self.invalidate_world_map_cache()
            self.reset()
            self.agent_goal = ARC_DEFAULT_AGENT_GOAL
            return True

        changed = self.game_kind != GAME_KIND_CRAFTAX
        if not changed:
            return False
        self.game_kind = GAME_KIND_CRAFTAX
        self.arc_adapter = None
        self._align_megaprompt_to_texture_theme()
        self.interaction_mode = DEFAULT_INTERACTION_MODE
        self.allowed_experts = list(DEFAULT_ALLOWED_EXPERTS)
        self.forced_expert = DEFAULT_FORCED_EXPERT
        self.set_texture_theme(self.texture_theme or DEFAULT_TEXTURE_THEME)
        self.reset()
        return True

    def set_arc_game_id(self, game_id: str) -> bool:
        return self.set_game_kind(GAME_KIND_ARC_AGI, arc_game_id=game_id)

    def _align_megaprompt_to_texture_theme(self) -> None:
        coerced = coerce_megaprompt_config_for_world_mode(
            self.megaprompt_config_name,
            self.texture_theme,
            game_kind=self.game_kind,
        )
        self.megaprompt_config_name = coerced

    def ensure_texture_mode_consistency(self) -> None:
        """Keep oracle prompts, megaprompt template, and campaign tasks on texture_theme."""
        if self.is_arc_game():
            self.campaign_state.set_world_mode("arc_agi")
        else:
            self.campaign_state.set_world_mode(self.texture_theme)
        self._align_megaprompt_to_texture_theme()
        if self.oracle is None:
            return
        oracle_mode = normalize_world_mode(getattr(self.oracle, "world_mode", None))
        if oracle_mode != normalize_world_mode(self.texture_theme):
            self.rebuild_oracle()

    def _current_agent_gen_config(self) -> GenConfig:
        cfg = self.agent_gen_config
        if cfg is None:
            cfg = GenConfig(
                max_new_tokens=ACTIVE_AGENT_MAX_NEW_TOKENS,
                do_sample=ACTIVE_AGENT_DO_SAMPLE,
                temperature=ACTIVE_AGENT_TEMPERATURE,
                top_p=ACTIVE_AGENT_TOP_P,
            )
            self.agent_gen_config = cfg
        return cfg

    def render_frame(self, block_px: int = BLOCK_PIXEL_SIZE_HUMAN) -> dict:
        if self.is_arc_game():
            adapter = self.ensure_arc_adapter()
            frame = adapter.render_frame()
            if self.megaprompt_config_name == "arc_image":
                obs = adapter.observation_payload(include_image=False)
                actions = obs.get("available_actions") or []
                frame["agent_observation"] = "\n".join(
                    [
                        f"Game: {obs.get('game_id') or self.arc_game_id} ({obs.get('title') or self.arc_game_id})",
                        f"State: {obs.get('state') or 'UNKNOWN'}",
                        f"Levels completed: {obs.get('levels_completed') or 0}",
                        f"Available actions: {', '.join(str(a) for a in actions) or 'none'}",
                        f"Frame image: {obs.get('w') or frame.get('w') or 0}x{obs.get('h') or frame.get('h') or 0} PNG attached as image input.",
                        "Coordinate convention: x=column 0..63 left-to-right, y=row 0..63 top-to-bottom.",
                    ]
                )
            elif self.megaprompt_config_name == "arc_grid_image":
                frame["agent_observation"] = "\n".join(
                    [
                        str(frame.get("agent_observation") or "").rstrip(),
                        "",
                        f"Frame image: {frame.get('w') or 0}x{frame.get('h') or 0} PNG attached as image input.",
                        "Use the image for spatial visual layout and the grid for exact coordinates/colors.",
                    ]
                ).strip()
            elif self.megaprompt_config_name == "arc_2_image":
                obs = self._arc_observation_payload_for_prompt(adapter)
                actions = obs.get("available_actions") or []
                previous_available = bool(obs.get("previous_png_b64"))
                frame["agent_observation"] = "\n".join(
                    [
                        f"Game: {obs.get('game_id') or self.arc_game_id} ({obs.get('title') or self.arc_game_id})",
                        f"State: {obs.get('state') or 'UNKNOWN'}",
                        f"Levels completed: {obs.get('levels_completed') or 0}",
                        f"Available actions: {', '.join(str(a) for a in actions) or 'none'}",
                        f"Previous observation image: {'attached' if previous_available else 'unavailable on the first tick or after reset'}.",
                        f"Current observation image: {obs.get('w') or frame.get('w') or 0}x{obs.get('h') or frame.get('h') or 0} PNG attached as image input.",
                        "Coordinate convention: x=column 0..63 left-to-right, y=row 0..63 top-to-bottom.",
                    ]
                )
            return frame
        pixels = self.render_fn(self.state, block_pixel_size=block_px)
        pixels = np.array(pixels, dtype=np.uint8)
        force_full = self._last_map_grid is None
        return {
            "w": int(pixels.shape[1]),
            "h": int(pixels.shape[0]),
            "png_b64": _pixels_to_png_base64(pixels),
            "agent_observation": self.get_agent_observation(),
            "world": build_world_payload(
                self,
                block_px=block_px,
                force_full_map=force_full,
                pixels_to_png_base64=_pixels_to_png_base64,
            ),
        }

    def invalidate_world_map_cache(self) -> None:
        self._last_map_grid = None
        self.map_epoch += 1

    def _finalize_trajectory_logger(self) -> None:
        if self.trajectory_logger is None:
            return
        try:
            if self.save_trajectory_enabled:
                self.trajectory_logger.save()
            else:
                self.trajectory_logger.delete()
        except Exception:
            logger.exception("Failed to finalize trajectory logger")

    def reset(self) -> dict:
        clear_episode_notes()
        self._finalize_leaderboard_attempt("reset")
        if self.trajectory_logger is not None:
            self._finalize_trajectory_logger()
        self.trajectory_logger = TrajectoryLogger(persist_tmp=self.save_trajectory_enabled)
        if self.is_arc_game():
            self.reset_arc_attempt_stats()
            frame = self.ensure_arc_adapter().reset()
            self.state = self.arc_adapter.current_obs if self.arc_adapter is not None else None
            self.invalidate_world_map_cache()
        else:
            self.rng, r = jax.random.split(self.rng)
            _, self.state = self.env.reset(r, self.env_params)
            # Invalidate BEFORE rendering: the new world must go out as one
            # full map snapshot with the new map_epoch. Rendering first would
            # diff the new world against the old grid (thousands of tile
            # patches, seconds of blocking) and stamp the frame with a stale
            # epoch, forcing the client into an extra full-map round trip.
            self.invalidate_world_map_cache()
            frame = self.render_frame()
        self.oracle_chat_history.clear()
        self.agent_goal = ""
        self.hints.clear()
        self.pending_agent_question = ""
        self.pending_agent_question_step = 0
        self.pending_human_tick = None
        self.last_agent_user_prompt = ""
        self.last_agent_system_message = ""
        self.agent_tick_history.clear()
        self.agent_direct_chat_active = False
        self.agent_direct_chat_turns.clear()
        if self.active_agent is not None:
            self.active_agent.clear_actions_history()
        self.last_step_transition = None
        self.campaign_state.reset_progress(self.state)
        self.episode_initial_state = self.state
        self.episode_discovered_keys.clear()
        if self.campaign_state.enabled:
            self._start_leaderboard_attempt()
        return frame

    def _achievement_discovery_payloads(self) -> list[dict[str, Any]]:
        if self.episode_initial_state is None:
            return []
        tasks = detect_new_episode_achievements(
            world_mode=self._campaign_world_mode(),
            episode_initial_state=self.episode_initial_state,
            current_state=self.state,
            already_discovered=self.episode_discovered_keys,
        )
        if not tasks:
            return []
        frame = self.render_frame()
        campaign_state = self.get_campaign_snapshot()
        return [
            {
                "type": "achievement_discovered",
                "achievement": {
                    "key": task.key,
                    "title": task.title,
                    "goal": task.goal,
                },
                "frame": frame,
                "campaign_state": campaign_state,
            }
            for task in tasks
        ]

    def get_campaign_snapshot(self) -> dict[str, Any]:
        return self.campaign_state.snapshot()

    def set_campaign_enabled(self, enabled: bool) -> dict[str, Any]:
        if self.is_arc_game():
            self.campaign_state.set_world_mode("arc_agi")
        if not enabled:
            self._finalize_leaderboard_attempt("campaign_disabled")
        self.campaign_state.set_enabled(enabled, self.state)
        if enabled:
            self.pending_agent_question = ""
            self.pending_agent_question_step = 0
            self._start_leaderboard_attempt()
        return self.get_campaign_snapshot()

    def start_campaign_phase2_level(self, level_key: str) -> dict[str, Any]:
        if not self.campaign_state.enabled:
            raise ValueError("Campaign mode is disabled")
        # Phase 2 starts each selected level from a fresh environment state.
        self.reset()
        self.campaign_state.enabled = True
        self.campaign_state.start_phase2(level_key, self.state)
        self._start_leaderboard_attempt()
        return self.get_campaign_snapshot()

    def _start_leaderboard_attempt(self) -> None:
        self.leaderboard_attempt_active = True
        self.leaderboard_attempt_started_at = utc_now_iso()
        self.leaderboard_phase1_questions = 0
        self.leaderboard_phase2_questions = 0
        self.leaderboard_last_checkpoint_signature = None
        self.leaderboard_level_steps = {}
        self.leaderboard_current_level_steps = 0
        self.leaderboard_total_steps = 0

    def _record_leaderboard_agent_step(self) -> None:
        if not self.leaderboard_attempt_active or not self.campaign_state.enabled:
            return
        self.leaderboard_current_level_steps += 1
        self.leaderboard_total_steps += 1

    def _record_completed_level_steps(self) -> None:
        snapshot = self.get_campaign_snapshot()
        phase2_keys = list(snapshot.get("phase2", {}).get("completed_keys") or [])
        phase1_keys = list(snapshot.get("phase1", {}).get("completed_keys") or [])
        newly_completed = [
            key
            for key in (*phase1_keys, *phase2_keys)
            if str(key or "").strip() and str(key) not in self.leaderboard_level_steps
        ]
        if not newly_completed:
            return
        for index, key in enumerate(newly_completed):
            task_key = str(key)
            if index == 0:
                self.leaderboard_level_steps[task_key] = self.leaderboard_current_level_steps
            else:
                self.leaderboard_level_steps[task_key] = 0
        self.leaderboard_current_level_steps = 0

    def _phase2_highest_level(self) -> int:
        completed = list(self.campaign_state.phase2_completed_keys or [])
        key_to_level = {
            str(task.get("key")): idx + 1
            for idx, task in enumerate(self.campaign_state.snapshot().get("tasks", []))
        }
        level = 0
        for key in completed:
            level = max(level, int(key_to_level.get(key, 0)))
        return level

    def _has_meaningful_campaign_progress(self) -> bool:
        return (
            int(self.campaign_state.snapshot().get("completed_count") or 0) > 0
            or len(self.campaign_state.phase2_completed_keys or []) > 0
            or self.leaderboard_phase1_questions > 0
            or self.leaderboard_phase2_questions > 0
        )

    def set_player_profile(self, *, name: str | None = None, avatar_id: int | None = None) -> None:
        if name is not None:
            self.player_name = str(name or "").strip()[:40]
        if avatar_id is not None:
            try:
                parsed = int(avatar_id)
            except (TypeError, ValueError):
                parsed = 0
            self.player_avatar_id = max(0, min(9, parsed))

    def _campaign_progress_signature(self) -> tuple[int, int, tuple[str, ...]]:
        snapshot = self.get_campaign_snapshot()
        phase2_keys = tuple(
            str(key)
            for key in (snapshot.get("phase2", {}).get("completed_keys") or [])
        )
        return (
            int(snapshot.get("phase1", {}).get("completed_count") or 0),
            int(self._phase2_highest_level()),
            phase2_keys,
        )

    def _build_leaderboard_entry(self, reason: str) -> dict[str, Any] | None:
        if not self._has_meaningful_campaign_progress():
            return None
        snapshot = self.get_campaign_snapshot()
        entry: dict[str, Any] = {
            "started_at": self.leaderboard_attempt_started_at or utc_now_iso(),
            "finished_at": utc_now_iso(),
            "finish_reason": reason,
            "active_agent_model": self.active_agent_model,
            "active_agent_mode": self.active_agent_mode,
            "megaprompt_config_name": self.megaprompt_config_name,
            "world_mode": self._campaign_world_mode(),
            "phase1_completed_levels": int(snapshot.get("phase1", {}).get("completed_count") or 0),
            "phase1_completed_keys": list(snapshot.get("phase1", {}).get("completed_keys") or []),
            "phase1_questions": int(self.leaderboard_phase1_questions),
            "phase2_highest_level": int(self._phase2_highest_level()),
            "phase2_completed_keys": list(snapshot.get("phase2", {}).get("completed_keys") or []),
            "phase2_questions": int(self.leaderboard_phase2_questions),
            "level_steps": dict(self.leaderboard_level_steps),
            "agent_steps": int(self.leaderboard_total_steps),
            "total_levels": int(
                snapshot.get("total_count") or campaign_total_levels(self._campaign_world_mode())
            ),
            "last_goal": self.agent_goal,
            "last_prompt_excerpt": (self.last_agent_user_prompt or "")[:200],
            "interaction_mode": self.interaction_mode,
        }
        player_name = str(self.player_name or "").strip()
        if self.interaction_mode == "oracle":
            entry["player_name"] = TALKINGHEADS_PLAYER_NAME
        elif player_name:
            entry["player_name"] = player_name[:40]
            entry["player_avatar_id"] = int(self.player_avatar_id)
        return entry

    def _checkpoint_leaderboard_attempt(self, reason: str) -> bool:
        if not self.leaderboard_attempt_active:
            return False
        signature = self._campaign_progress_signature()
        if signature == self.leaderboard_last_checkpoint_signature:
            return False
        entry = self._build_leaderboard_entry(reason)
        if entry is None:
            return False
        append_leaderboard_entry(entry)
        self.leaderboard_last_checkpoint_signature = signature
        return True

    def _finalize_leaderboard_attempt(self, reason: str) -> None:
        if not self.leaderboard_attempt_active:
            return
        self._checkpoint_leaderboard_attempt(reason)
        self.leaderboard_attempt_active = False
        self.leaderboard_last_checkpoint_signature = None

    def _resolve_agent_goal(self, manual_goal: str) -> str:
        default_goal = ARC_DEFAULT_AGENT_GOAL if self.is_arc_game() else DEFAULT_AGENT_GOAL
        manual = manual_goal.strip() or default_goal
        if not self.campaign_state.enabled:
            return manual
        if self.campaign_state.maybe_advance(self.state):
            self._record_completed_level_steps()
            self._checkpoint_leaderboard_attempt("level_complete")
        phase2_task = self.campaign_state.phase2_selected_task()
        if phase2_task is not None:
            return phase2_task.goal
        if self.campaign_state.is_finished():
            return "Campaign complete — all tasks finished."
        task = self.campaign_state.current_task()
        if task is None:
            return manual
        return task.goal

    def _on_agent_goal_changed(self, new_goal: str) -> None:
        if new_goal == self.agent_goal:
            return
        self.agent_goal = new_goal
        self.oracle_chat_history.clear()
        self.hints.clear()
        self.last_step_transition = None
        self.last_agent_user_prompt = ""
        self.last_agent_system_message = ""
        self.agent_tick_history.clear()
        self.agent_direct_chat_turns.clear()
        self.agent_direct_chat_active = False
        self.pending_agent_question = ""
        self.pending_agent_question_step = 0
        self.pending_human_tick = None

    def step(self, action: int) -> tuple[float, bool, dict]:
        if self.is_arc_game():
            action_text = str(action)
            adapter = self.ensure_arc_adapter()
            self._capture_arc_previous_observation(adapter)
            result = adapter.step(action_text)
            self.last_step_debug_timing = dict(result.timing or {})
            self.state = self.arc_adapter.current_obs if self.arc_adapter is not None else self.state
            self._record_arc_decision_step()
            self.arc_actions_count += 1
            self.arc_manual_actions_count += 1
            self._record_arc_action_history(action_text)
            self._sync_arc_campaign_after_step()
            self._maybe_auto_submit_arc_human_score(result.frame)
            done = self._arc_episode_should_end(result)
            if done:
                self.mark_arc_finished(result.frame)
                if not self._arc_keeps_level_progression():
                    result.frame = self._reset_arc_episode_after_single_level()
            if done and self.trajectory_logger is not None:
                self._finalize_trajectory_logger()
            return result.reward, done, result.frame
        self.rng, r = jax.random.split(self.rng)
        _, self.state, reward, done, _ = self.step_fn(r, self.state, action, self.env_params)
        if done and self.trajectory_logger is not None:
            self._finalize_trajectory_logger()
        return float(reward), bool(done), self.render_frame()

    def apply_actions(self, action_str: str) -> tuple[float, bool, list[dict]]:
        if self.is_arc_game():
            adapter = self.ensure_arc_adapter()
            self._capture_arc_previous_observation(adapter)
            result = adapter.step(action_str)
            self.last_step_debug_timing = dict(result.timing or {})
            self.state = adapter.current_obs
            self._record_arc_decision_step()
            self.arc_actions_count += 1
            self.arc_agent_actions_count += 1
            self._record_arc_action_history(action_str)
            self.last_step_transition = {
                "before": None,
                "after": adapter.current_obs,
                "action": str(action_str).strip().upper(),
            }
            self._sync_arc_campaign_after_step()
            self._maybe_auto_submit_arc_human_score(result.frame)
            done = self._arc_episode_should_end(result)
            if done:
                self.mark_arc_finished(result.frame)
                if not self._arc_keeps_level_progression():
                    result.frame = self._reset_arc_episode_after_single_level()
            if done and self.trajectory_logger is not None:
                self._finalize_trajectory_logger()
            return result.reward, done, [result.frame]
        tokens = action_str.strip().upper().split()
        reward = 0.0
        done = False
        frames: list[dict] = []
        for token in tokens:
            if token and token in Action.__members__:
                action_value = Action[token].value
                state_before = self.state
                self.rng, r = jax.random.split(self.rng)
                _, self.state, reward, done, _ = self.step_fn(
                    r, self.state, action_value, self.env_params
                )
                self.last_step_transition = {
                    "before": state_before,
                    "after": self.state,
                    "action": token,
                }
                frames.append(self.render_frame())
                if done:
                    if self.trajectory_logger is not None:
                        self._finalize_trajectory_logger()
                    break
        return float(reward), bool(done), frames

    def ensure_active_agent(self) -> ActiveAgent:
        if self.active_agent is None:
            mode = str(self.active_agent_mode or "openrouter").strip().lower()
            env_key = "HF_TOKEN" if mode == "hub" else "OPENROUTER_API_KEY"
            base_agent = ActiveAgent(
                model_name=self.active_agent_model,
                mode=self.active_agent_mode,
                api_key=self.get_api_secret(env_key) or None,
                reasoning=False,
            )
            self.active_agent = ActiveAgentStatistics(base_agent)
        return self.active_agent

    @staticmethod
    def _format_chat_step_prefix(entry: dict[str, Any], key: str, label: str) -> str:
        try:
            step = int(entry.get(key) or 0)
        except (TypeError, ValueError):
            step = 0
        return f"[{label} at step {step}] " if step > 0 else ""

    @staticmethod
    def _format_chat_history_for_agent(history: list[dict[str, Any]]) -> str:
        if not history:
            return "No previous messages yet."
        lines: list[str] = []
        for i, entry in enumerate(history[-Session.MAX_OPERATOR_MESSAGES_FOR_AGENT :], 1):
            q = entry.get("question", "").strip()
            a = entry.get("answer", "").strip()
            if q or a:
                q_prefix = Session._format_chat_step_prefix(entry, "question_step", "asked")
                a_prefix = Session._format_chat_step_prefix(entry, "answer_step", "answered")
                lines.extend([f"Q: {q_prefix}{q}", f"A: {a_prefix}{a}"])
                if i < len(history):
                    lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _is_direct_chat_end_message(message: str) -> bool:
        return str(message or "").strip().upper() == "END."

    def end_agent_direct_chat(self) -> None:
        self.agent_direct_chat_active = False

    def start_agent_direct_chat(self) -> None:
        self.agent_direct_chat_active = True

    def _record_agent_tick_turn(self, observation: str, prompt: str, response: str) -> None:
        self.agent_tick_history.append(
            {
                "observation": observation.strip(),
                "prompt": prompt.strip(),
                "response": response.strip(),
            }
        )

    @staticmethod
    def _omit_image_markers_from_history(text: str) -> str:
        return _IMAGE_MARKER_RE.sub("[Previous ARC frame image omitted]", str(text or ""))

    def _build_direct_chat_llm_history(self) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for entry in self.agent_tick_history:
            observation = entry.get("observation", "").strip()
            prompt = self._omit_image_markers_from_history(entry.get("prompt", "").strip())
            response = entry.get("response", "").strip()
            if not prompt and not response:
                continue
            user_parts: list[str] = []
            if observation:
                user_parts.append(f"## Observation\n{observation}")
            if prompt:
                user_parts.append(f"## Prompt\n{prompt}")
            history.append({"role": "user", "content": "\n\n".join(user_parts) or prompt})
            if response:
                history.append({"role": "assistant", "content": response})
        for entry in self.agent_direct_chat_turns:
            message = entry.get("message", "").strip()
            response = entry.get("response", "").strip()
            if message:
                history.append({"role": "user", "content": message})
            if response:
                history.append({"role": "assistant", "content": response})
        return history

    @staticmethod
    def _format_hints_for_agent(hints: list[str]) -> str:
        if not hints:
            return "No additional hints from the operator yet."
        lines = [f"{i}. {str(h).strip()}" for i, h in enumerate(hints, 1) if str(h).strip()]
        return "\n".join(lines) if lines else "No additional hints from the operator yet."

    @staticmethod
    def _is_long_reasoning_without_action(raw_answer: str, parsed: dict[str, Any]) -> bool:
        if "action" in parsed:
            return False
        text = str(raw_answer or "").strip()
        if not text:
            return False
        question = str(parsed.get("question", "")).strip()
        words = len(text.split())
        # Heuristic: if answer is very long and we still did not get an action,
        # treat it as an overlong reasoning output and request action-only retry.
        return words >= 350 or len(text) >= 2400 or len(question) >= 800

    def _record_operator_feedback(self, feedback: str) -> None:
        if self.is_arc_game():
            return
        self.oracle_chat_history.append(
            {
                "question": "System",
                "answer": feedback,
            }
        )
        self.save_agent_oracle_dialog()

    async def _retry_agent_with_operator_feedback(
        self,
        agent: ActiveAgent,
        observation_text: str,
        feedback: str,
        *,
        step_idx: int,
        goal: str,
        prompt: str,
        retry_kind: str,
    ) -> tuple[str, dict[str, Any]]:
        """Continue the same LLM thread with an operator notice, then action-only reply."""
        history = None if self.is_arc_game() else self._build_direct_chat_llm_history()
        user_message = feedback
        if self.is_arc_game():
            user_message = "\n\n".join(
                [
                    prompt,
                    "## Platform format correction",
                    feedback,
                    "Return exactly one valid `--- Act ---` or `--- Q ---` block now.",
                ]
            )
        raw_answer = await asyncio.to_thread(
            lambda: agent.chat(
                user_message=user_message,
                system_message=DEFAULT_AGENT_SYSTEM_MESSAGE,
                gen=self._current_agent_gen_config(),
                history=history or None,
            )
        )
        parsed = parse_agent_answer(raw_answer)
        self._record_agent_tick_turn(
            observation_text,
            feedback,
            strip_to_database_tags(raw_answer).strip(),
        )
        if self.trajectory_logger is not None:
            try:
                self.trajectory_logger.add_step(
                    agent_prompt=prompt,
                    env_state=self.state,
                    oracle_dialog=self.oracle_chat_history,
                    raw_answer=raw_answer,
                    parsed=parsed,
                    meta={
                        "step_index": step_idx,
                        "goal": goal,
                        "operator_retry_kind": retry_kind,
                        "active_agent_model": self.active_agent_model,
                        "active_agent_mode": self.active_agent_mode,
                        "interaction_mode": self.interaction_mode,
                        "server": "human" if self.interaction_mode == "human" else "oracle",
                    },
                )
            except Exception:
                logger.exception("Failed to log operator-feedback retry trajectory step")
        return raw_answer, parsed

    def save_agent_oracle_dialog(self) -> None:
        if not self.oracle_chat_history:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = LOG_DIR / f"agent_oracle_dialog_{ts}.txt"
        with file_path.open("w", encoding="utf-8") as f:
            f.write(f"Goal: {self.agent_goal}\n\n")
            for i, entry in enumerate(self.oracle_chat_history, 1):
                q_prefix = self._format_chat_step_prefix(entry, "question_step", "asked")
                a_prefix = self._format_chat_step_prefix(entry, "answer_step", "answered")
                f.write(
                    f"Turn {i}\nAgent/Operator: {q_prefix}{str(entry.get('question','')).strip()}\n"
                    f"Oracle: {a_prefix}{str(entry.get('answer','')).strip()}\n\n"
                )

    def set_interaction_mode(self, mode: str) -> None:
        if self.is_arc_game():
            self.interaction_mode = "human"
            self.pending_agent_question = ""
            self.pending_agent_question_step = 0
            self.pending_human_tick = None
            return
        mode_norm = str(mode or "").strip().lower()
        if mode_norm not in {"oracle", "human"}:
            raise ValueError("interaction_mode must be 'oracle' or 'human'")
        self.interaction_mode = mode_norm
        self.pending_agent_question = ""
        self.pending_agent_question_step = 0
        self.pending_human_tick = None

    def set_allowed_experts(self, experts: list[str]) -> None:
        if self.is_arc_game():
            self.allowed_experts = []
            return
        self.allowed_experts = list(ALL_EXPERTS) if not experts else Intent.normalize_allowed_experts(experts)

    def set_forced_expert(self, expert: Optional[str]) -> None:
        if self.is_arc_game():
            self.forced_expert = None
            return
        raw_value = str(expert or "").strip()
        self.forced_expert = Intent.normalize_expert_name(raw_value) if raw_value else None

    def clamp_agent_steps(self, steps: int) -> int:
        return max(1, min(int(steps), self.max_agent_steps_per_tick))

    def set_default_agent_steps(self, steps: int) -> None:
        steps_value = self.clamp_agent_steps(steps)
        if steps_value < 1:
            raise ValueError("default_agent_steps must be at least 1")
        self.default_agent_steps = steps_value

    def set_megaprompt_config_name(self, config_name: str) -> None:
        normalized = str(config_name or "").strip()
        if not normalized:
            raise ValueError("megaprompt_config_name must be non-empty")
        self.megaprompt_config_name = coerce_megaprompt_config_for_world_mode(
            normalized,
            self.texture_theme,
            game_kind=self.game_kind,
        )

    def set_arc_prompt_extra(self, text: str) -> None:
        value = str(text or "").strip()
        if len(value) > 8000:
            raise ValueError("arc_prompt_extra must be at most 8000 characters")
        self.arc_prompt_extra = value

    def set_operator_call_limit(self, limit: int | None) -> None:
        if limit is None:
            self.operator_call_limit = None
            return
        self.operator_call_limit = max(1, int(limit))

    def clear_operator_call_limit(self) -> None:
        self.operator_call_limit = None

    def get_expert_models(self) -> dict[str, str]:
        cfg = self.get_oracle_config()
        return {
            key: str(getattr(cfg, model_attr))
            for key, (model_attr, _) in EXPERT_CONFIG_KEYS.items()
        }

    def get_expert_modes(self) -> dict[str, str]:
        cfg = self.get_oracle_config()
        return {
            key: str(getattr(cfg, mode_attr))
            for key, (_, mode_attr) in EXPERT_CONFIG_KEYS.items()
        }

    def set_expert_models(self, models: dict[str, str]) -> None:
        if not isinstance(models, dict):
            raise ValueError("expert_models must be an object")
        normalized = {str(k): str(v).strip() for k, v in models.items()}
        cfg = self.get_oracle_config()
        updated = False
        for key, (model_attr, _) in EXPERT_CONFIG_KEYS.items():
            if key not in normalized:
                continue
            model_name = normalized[key]
            if not model_name:
                raise ValueError(f"{key} model must be non-empty")
            if str(getattr(cfg, model_attr)) != model_name:
                setattr(cfg, model_attr, model_name)
                updated = True
        if updated:
            self.rebuild_oracle(cfg)

    def set_expert_modes(self, modes: dict[str, str]) -> None:
        if not isinstance(modes, dict):
            raise ValueError("expert_modes must be an object")
        normalized = {str(k): str(v).strip().lower() for k, v in modes.items()}
        allowed = {"hub", "openrouter", "local"}
        cfg = self.get_oracle_config()
        updated = False
        for key, (_, mode_attr) in EXPERT_CONFIG_KEYS.items():
            if key not in normalized:
                continue
            mode = normalized[key]
            if mode not in allowed:
                raise ValueError(f"{key} mode must be one of: {', '.join(sorted(allowed))}")
            if str(getattr(cfg, mode_attr)) != mode:
                setattr(cfg, mode_attr, mode)
                updated = True
        if updated:
            self.rebuild_oracle(cfg)

    def sync_expert_models_to_active_agent(self) -> None:
        """Unified gateway: every expert uses the movement agent model id."""
        model = str(self.active_agent_model or "").strip()
        if not model:
            return
        cfg = self.get_oracle_config()
        updated = False
        for _key, (model_attr, _mode_attr) in EXPERT_CONFIG_KEYS.items():
            if str(getattr(cfg, model_attr)) != model:
                setattr(cfg, model_attr, model)
                updated = True
        if updated:
            self.rebuild_oracle(cfg)

    def sync_expert_modes_to_active_agent(self) -> None:
        """Unified gateway: every expert uses the movement agent provider mode."""
        mode = str(self.active_agent_mode or ACTIVE_AGENT_MODE).strip().lower()
        if mode not in {"hub", "openrouter"}:
            return
        cfg = self.get_oracle_config()
        updated = False
        for _key, (_model_attr, mode_attr) in EXPERT_CONFIG_KEYS.items():
            if str(getattr(cfg, mode_attr)) != mode:
                setattr(cfg, mode_attr, mode)
                updated = True
        if updated:
            self.rebuild_oracle(cfg)

    def apply_unified_agent_gateway(self) -> None:
        self.sync_expert_modes_to_active_agent()
        self.sync_expert_models_to_active_agent()
        self._align_megaprompt_to_texture_theme()

    def sync_client_runtime_config(
        self,
        *,
        exo_planet_enabled: bool | None = None,
        game_kind: str | None = None,
        arc_game_id: str | None = None,
        megaprompt_config_name: str | None = None,
        arc_prompt_extra: str | None = None,
        active_agent_model: str | None = None,
        active_agent_mode: str | None = None,
        player_name: str | None = None,
        player_avatar_id: int | None = None,
    ) -> None:
        if game_kind is not None or arc_game_id is not None:
            self.set_game_kind(game_kind or self.game_kind, arc_game_id=arc_game_id or self.arc_game_id)
        if exo_planet_enabled is not None and not self.is_arc_game():
            self.set_texture_theme("exo-planet" if bool(exo_planet_enabled) else "craftax")
        if megaprompt_config_name:
            self.set_megaprompt_config_name(megaprompt_config_name)
        else:
            self._align_megaprompt_to_texture_theme()
        if arc_prompt_extra is not None:
            self.set_arc_prompt_extra(arc_prompt_extra)
        if active_agent_model:
            self.set_active_agent_model(active_agent_model)
        if active_agent_mode:
            self.set_active_agent_mode(active_agent_mode)
        if player_name is not None or player_avatar_id is not None:
            self.set_player_profile(name=player_name, avatar_id=player_avatar_id)

    def set_active_agent_model(self, model_name: str) -> None:
        normalized = str(model_name or "").strip()
        if not normalized:
            raise ValueError("active_agent_model must be non-empty")
        if normalized == self.active_agent_model:
            return
        self.active_agent_model = normalized
        # Recreate active agent lazily with the updated model.
        self.active_agent = None
        self.sync_expert_models_to_active_agent()

    def set_active_agent_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"hub", "openrouter"}:
            raise ValueError("active_agent_mode must be 'hub' or 'openrouter'")
        if normalized == self.active_agent_mode:
            return
        self.active_agent_mode = normalized
        # Recreate active agent lazily with the updated provider mode.
        self.active_agent = None
        self.sync_expert_modes_to_active_agent()

    def set_agent_sampling(self, do_sample: bool, temperature: Optional[float] = None, top_p: Optional[float] = None) -> None:
        cfg = self._current_agent_gen_config()
        cfg.do_sample = bool(do_sample)
        if temperature is not None:
            temperature_value = float(temperature)
            if temperature_value < 0.0:
                raise ValueError("active_agent_temperature must be >= 0")
            cfg.temperature = temperature_value
        if top_p is not None:
            top_p_value = float(top_p)
            if top_p_value <= 0.0 or top_p_value > 1.0:
                raise ValueError("active_agent_top_p must be in (0, 1]")
            cfg.top_p = top_p_value

    def reset_runtime_config_to_defaults(self) -> None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "oracle_config.yaml"
        cfg = load_config(path=str(config_path))
        self.clear_api_secrets()
        self.active_agent = None
        self.active_agent_model = ACTIVE_AGENT_MODEL
        self.active_agent_mode = ACTIVE_AGENT_MODE
        self.megaprompt_config_name = DEFAULT_MEGAPROMPT_CONFIG_NAME
        self.arc_prompt_extra = ""
        self.arc_previous_reasoning = ""
        self.interaction_mode = DEFAULT_INTERACTION_MODE
        self.allowed_experts = list(DEFAULT_ALLOWED_EXPERTS)
        self.forced_expert = DEFAULT_FORCED_EXPERT
        self.agent_gen_config = GenConfig(
            max_new_tokens=ACTIVE_AGENT_MAX_NEW_TOKENS,
            do_sample=ACTIVE_AGENT_DO_SAMPLE,
            temperature=ACTIVE_AGENT_TEMPERATURE,
            top_p=ACTIVE_AGENT_TOP_P,
        )
        self.oracle_config_data = cfg
        self.oracle = None
        self.game_kind = GAME_KIND_CRAFTAX
        self.arc_adapter = None
        self.arc_game_id = DEFAULT_ARC_GAME_ID
        self.set_texture_theme(DEFAULT_TEXTURE_THEME)
        self.apply_unified_agent_gateway()
        self.ensure_texture_mode_consistency()

    def set_texture_theme(self, theme: str) -> bool:
        normalized = "exo-planet" if str(theme).strip().lower() in {"exo", "exo-planet"} else "craftax"
        theme_changed = normalized != self.texture_theme
        if not theme_changed:
            return False

        previous = self.texture_theme
        self.texture_theme = normalized
        self.render_fn = make_render_craftax_pixels_jit(normalized)
        self.render_world_base_fn, self.render_obs_overlay_fn = make_render_world_jit(normalized)
        self.invalidate_world_map_cache()
        try:
            if self.oracle is not None:
                self.rebuild_oracle()
            self._align_megaprompt_to_texture_theme()
            self.campaign_state.set_world_mode(normalized)
            self.reset()
            self.agent_goal = default_agent_goal_for_texture_theme(normalized)
        except Exception:
            self.texture_theme = previous
            self.render_fn = make_render_craftax_pixels_jit(previous)
            self.render_world_base_fn, self.render_obs_overlay_fn = make_render_world_jit(previous)
            self.invalidate_world_map_cache()
            if self.oracle is not None:
                self.rebuild_oracle()
            self._align_megaprompt_to_texture_theme()
            self.campaign_state.set_world_mode(previous)
            raise
        return True

    def handle_oracle_question(
        self,
        question: str,
        run_code: bool,
        forced_expert: Optional[str] = None,
        question_step: int | None = None,
        answer_step: int | None = None,
    ) -> str:
        if self.is_arc_game():
            raise ValueError("Oracle mode is disabled for ARC-AGI-3 games. Use a human operator.")
        forced_expert_name = (
            Intent.normalize_expert_name(str(forced_expert).strip())
            if str(forced_expert or "").strip()
            else self.forced_expert
        )
        answer_text = str(
            self.ensure_oracle().answer(
                question,
                run_code=run_code,
                module_name="oracle_ws",
                env_state=self.state,
                allowed_experts=self.allowed_experts or None,
                forced_expert=forced_expert_name,
                goal_hint=self.agent_goal,
            )
        )
        entry: dict[str, Any] = {"question": question, "answer": answer_text}
        if question_step is not None:
            entry["question_step"] = int(question_step)
        if answer_step is not None:
            entry["answer_step"] = int(answer_step)
        self.oracle_chat_history.append(entry)
        self.save_agent_oracle_dialog()
        self.pending_agent_question = ""
        self.pending_agent_question_step = 0
        return answer_text

    def _oracle_expert_for_agent_question(self) -> str:
        """
        Agent-originated questions should go through goal_expert aggregation
        pipeline (question -> map/mechanics/action -> goal) unless an explicit
        expert is pinned in session config.
        """
        return self.forced_expert or Intent.GOAL_EXPERT

    def _generate_arc_agent_prompt(
        self,
        *,
        goal: str,
        observation_text: str,
        operator_budget: str = "",
    ) -> str:
        adapter = self.ensure_arc_adapter()
        prompt = generate_arc_agent_prompt(
            goal=goal.strip() or ARC_DEFAULT_AGENT_GOAL,
            arc_observation=self._arc_observation_payload_for_prompt(adapter),
            dialog=self.oracle_chat_history,
            action_history=self.arc_action_history[-30:],
            previous_reasoning=self.arc_previous_reasoning,
            current_step=self.next_arc_step_number(),
            megaprompt_config_name=self.megaprompt_config_name,
            operator_call_budget=operator_budget,
        )
        extra = self.arc_prompt_extra.strip()
        if extra:
            prompt = extra
        return prompt

    def current_agent_prompt_preview(self) -> tuple[str, str, str, bool]:
        goal = self.agent_goal.strip() if self.agent_goal else ""
        if self.is_arc_game():
            resolved_goal = self._resolve_agent_goal(goal or ARC_DEFAULT_AGENT_GOAL)
            operator_budget = ""
            if is_deployment_megaprompt(self.megaprompt_config_name) and self.operator_call_limit is not None:
                operator_budget = format_operator_call_budget_text(
                    used=len(self.oracle_chat_history),
                    limit=self.operator_call_limit,
                )
            prompt = self._generate_arc_agent_prompt(
                goal=resolved_goal,
                observation_text=self.get_agent_observation(),
                operator_budget=operator_budget,
            )
            return prompt, DEFAULT_AGENT_SYSTEM_MESSAGE, resolved_goal, True
        return (
            self.last_agent_user_prompt or "",
            self.last_agent_system_message or "",
            goal or DEFAULT_AGENT_GOAL,
            bool((self.last_agent_user_prompt or "").strip()),
        )

    async def handle_agent_tick(
        self,
        steps: int,
        goal: str,
        *,
        start_step_idx: int = 0,
        step_count: int = 0,
        resume_after_human: bool = False,
    ):
        self._align_megaprompt_to_texture_theme()
        agent = self.ensure_active_agent()
        steps = self.clamp_agent_steps(steps)
        manual_goal = goal.strip() or DEFAULT_AGENT_GOAL
        goal = self._resolve_agent_goal(manual_goal)
        if resume_after_human:
            start_step_idx = max(0, min(int(start_step_idx), steps))
            step_count = max(0, int(step_count))
            if self.stop_agent_requested:
                return
        else:
            self._on_agent_goal_changed(goal)
            step_count = 0
            start_step_idx = 0
            # Companion research runs many ticks in a loop; preserve stop across calls.
            if not self.companion_research_active:
                self.stop_agent_requested = False
            self.pending_human_tick = None
        from .knowledge_paths import play_knowledge_context

        with play_knowledge_context(self):
            async for _event in self._handle_agent_tick_steps(
                agent=agent,
                steps=steps,
                goal=goal,
                manual_goal=manual_goal,
                start_step_idx=start_step_idx,
                step_count=step_count,
                resume_after_human=resume_after_human,
            ):
                yield _event

    async def _handle_agent_tick_steps(
        self,
        *,
        agent,
        steps: int,
        goal: str,
        manual_goal: str,
        start_step_idx: int,
        step_count: int,
        resume_after_human: bool,
    ):
        for step_idx in range(start_step_idx, steps):
            invalid_format_retry_done = False
            if self.stop_agent_requested:
                return
            if self.campaign_state.enabled:
                new_goal = self._resolve_agent_goal(manual_goal)
                if new_goal != goal:
                    goal = new_goal
                    self._on_agent_goal_changed(goal)
            operator_budget = ""
            if is_deployment_megaprompt(self.megaprompt_config_name) and self.operator_call_limit is not None:
                operator_budget = format_operator_call_budget_text(
                    used=len(self.oracle_chat_history),
                    limit=self.operator_call_limit,
                )
            observation_text = self.get_agent_observation()
            if self.is_arc_game():
                prompt = self._generate_arc_agent_prompt(
                    goal=goal,
                    observation_text=observation_text,
                    operator_budget=operator_budget,
                )
            else:
                prompt = generate_agent_prompt(
                    goal=goal,
                    observation=self.state,
                    message_from_operator=self._format_chat_history_for_agent(self.oracle_chat_history),
                    inventory=format_inventory_from_env_state(
                        self.state,
                        world_mode=self.texture_theme,
                    ),
                    hints=self._format_hints_for_agent(self.hints),
                    previous_actions_analysis=build_previous_actions_analysis(
                        agent.actions_history,
                        getattr(agent, "consecutive_questions_count", 0),
                    ),
                    action_history=list(agent.actions_history),
                    state_history=[self.last_step_transition] if self.last_step_transition else [],
                    megaprompt_config_name=self.megaprompt_config_name,
                    world_mode=self.texture_theme,
                    operator_call_budget=operator_budget,
                )
            self.last_agent_user_prompt = prompt
            self.last_agent_system_message = DEFAULT_AGENT_SYSTEM_MESSAGE
            if self.stop_agent_requested:
                return
            raw_answer = await asyncio.to_thread(
                lambda: agent.chat(
                    user_message=prompt,
                    system_message=DEFAULT_AGENT_SYSTEM_MESSAGE,
                    gen=self._current_agent_gen_config(),
                )
            )
            if self.stop_agent_requested:
                return
            parsed = parse_agent_answer(raw_answer)
            self._record_agent_tick_turn(
                observation_text,
                prompt,
                strip_to_database_tags(raw_answer).strip(),
            )
            if self.trajectory_logger is None:
                self.trajectory_logger = TrajectoryLogger(persist_tmp=self.save_trajectory_enabled)
            try:
                self.trajectory_logger.add_step(
                    agent_prompt=prompt,
                    env_state=self.state,
                    oracle_dialog=self.oracle_chat_history,
                    raw_answer=raw_answer,
                    parsed=parsed,
                    meta={
                        "step_index": step_idx,
                        "goal": goal,
                        "active_agent_model": self.active_agent_model,
                        "active_agent_mode": self.active_agent_mode,
                        "interaction_mode": self.interaction_mode,
                        "server": "human" if self.interaction_mode == "human" else "oracle",
                    },
                )
            except Exception:
                logger.exception("Failed to log trajectory step")
            if self._is_long_reasoning_without_action(raw_answer, parsed):
                self._record_operator_feedback(LONG_REASONING_NO_ACTION_FEEDBACK)
                yield {
                    "type": "agent_operator_notice",
                    "message": LONG_REASONING_NO_ACTION_FEEDBACK,
                    "reasoning": parsed.get("reasoning", ""),
                    "tick": step_idx + 1,
                    "total_ticks": steps,
                    "frame": self.render_frame(),
                    "campaign_state": self.get_campaign_snapshot(),
                    **_agent_ws_knowledge_fields(parsed),
                }
                try:
                    raw_answer, parsed = await self._retry_agent_with_operator_feedback(
                        agent,
                        observation_text,
                        LONG_REASONING_NO_ACTION_FEEDBACK,
                        step_idx=step_idx,
                        goal=goal,
                        prompt=prompt,
                        retry_kind="long_reasoning",
                    )
                    if self.stop_agent_requested:
                        return
                except Exception:
                    logger.exception("Long-reasoning retry failed")
                    yield {
                        "type": "agent_system_notice",
                        "message": (
                            "Could not request an action-only follow-up after long reasoning. "
                            "Try again on the next tick."
                        ),
                        "reasoning": parsed.get("reasoning", ""),
                        "tick": step_idx + 1,
                        "total_ticks": steps,
                        "frame": self.render_frame(),
                        "campaign_state": self.get_campaign_snapshot(),
                    }
                    continue
            if (
                "action" not in parsed
                and "question" not in parsed
                and not invalid_format_retry_done
            ):
                invalid_format_retry_done = True
                self._record_operator_feedback(INVALID_FORMAT_FEEDBACK)
                yield {
                    "type": "agent_operator_notice",
                    "message": INVALID_FORMAT_FEEDBACK,
                    "reasoning": parsed.get("reasoning", ""),
                    "tick": step_idx + 1,
                    "total_ticks": steps,
                    "frame": self.render_frame(),
                    "campaign_state": self.get_campaign_snapshot(),
                    **_agent_ws_knowledge_fields(parsed),
                }
                try:
                    raw_answer, parsed = await self._retry_agent_with_operator_feedback(
                        agent,
                        observation_text,
                        INVALID_FORMAT_FEEDBACK,
                        step_idx=step_idx,
                        goal=goal,
                        prompt=prompt,
                        retry_kind="invalid_format",
                    )
                    if self.stop_agent_requested:
                        return
                except Exception:
                    logger.exception("Invalid-format retry failed")
                    yield {
                        "type": "agent_system_notice",
                        "message": (
                            "Could not request an action-only follow-up after invalid format. "
                            "Try again on the next tick."
                        ),
                        "reasoning": parsed.get("reasoning", ""),
                        "tick": step_idx + 1,
                        "total_ticks": steps,
                        "frame": self.render_frame(),
                        "campaign_state": self.get_campaign_snapshot(),
                    }
                    continue
            if self.stop_agent_requested:
                return
            self._record_arc_previous_reasoning(parsed)
            if "action" in parsed:
                display_action = (
                    str(parsed.get("action") or parsed.get("action_raw") or "").strip().upper()
                    if self.is_arc_game()
                    else format_action_for_ui(parsed, world_mode=self.texture_theme)
                )
                agent.record_action(display_action)
                reward, done, frames = self.apply_actions(parsed["action"])
                if self.campaign_state.enabled:
                    new_goal = self._resolve_agent_goal(manual_goal)
                    if new_goal != goal:
                        goal = new_goal
                        self._on_agent_goal_changed(goal)
                step_count += 1
                self._record_leaderboard_agent_step()
                yield {
                    "type": "agent_action",
                    "action": display_action,
                    "reasoning": parsed.get("reasoning", ""),
                    "reward": reward,
                    "done": done,
                    "tick": step_idx + 1,
                    "total_ticks": steps,
                    "player_position": (
                        ""
                        if self.is_arc_game()
                        else str(np.array(self.state.player_position).tolist())
                    ),
                    "frames": frames,
                    "campaign_state": self.get_campaign_snapshot(),
                    **_agent_ws_knowledge_fields(parsed),
                }
                for discovery_payload in self._achievement_discovery_payloads():
                    yield discovery_payload
                if done:
                    return
                continue
            if "question" in parsed:
                if self.is_arc_game():
                    self.arc_questions_count += 1
                if self.campaign_state.phase2_active():
                    self.leaderboard_phase2_questions += 1
                else:
                    self.leaderboard_phase1_questions += 1
                agent.record_question()
                agent.clear_actions_history()
                self.last_step_transition = None
                question = parsed["question"]
                question_step = self._record_arc_decision_step() if self.is_arc_game() else step_idx + 1
                step_count += 1
                self._record_leaderboard_agent_step()
                tick_info = step_idx + 1
                if self.interaction_mode == "human":
                    self.pending_agent_question = question
                    self.pending_agent_question_step = question_step
                    self.pending_human_tick = {
                        "steps": steps,
                        "manual_goal": manual_goal,
                        "resume_step_idx": step_idx + 1,
                        "step_count": step_count,
                        "question_step": question_step,
                    }
                    yield {
                        "type": "agent_question_pending",
                        "question": question,
                        "reasoning": parsed.get("reasoning", ""),
                        "question_step": question_step,
                        "tick": tick_info,
                        "total_ticks": steps,
                        "campaign_state": self.get_campaign_snapshot(),
                        **_agent_ws_knowledge_fields(parsed),
                    }
                    return
                yield {
                    "type": "agent_question_pending",
                    "question": question,
                    "reasoning": parsed.get("reasoning", ""),
                    "question_step": question_step,
                    "tick": tick_info,
                    "total_ticks": steps,
                    "campaign_state": self.get_campaign_snapshot(),
                    **_agent_ws_knowledge_fields(parsed),
                }
                if self.should_skip_oracle_autoreply():
                    return
                try:
                    answer_text = await asyncio.to_thread(
                        self.handle_oracle_question,
                        question,
                        True,
                        self._oracle_expert_for_agent_question(),
                        question_step,
                        question_step,
                    )
                    if self.should_skip_oracle_autoreply():
                        return
                    yield {
                        "type": "agent_message",
                        "question": question,
                        "answer": answer_text,
                        "reasoning": parsed.get("reasoning", ""),
                        "question_step": question_step,
                        "answer_step": question_step,
                        "ok": True,
                        "tick": tick_info,
                        "total_ticks": steps,
                        "frame": self.render_frame(),
                        "campaign_state": self.get_campaign_snapshot(),
                        **_agent_ws_knowledge_fields(parsed),
                    }
                    continue
                except Exception:
                    logger.exception("Error handling agent question")
                    fallback_answer = "Operator couldt answer now. Please make 5 steps in any direction and try again"
                    self.oracle_chat_history.append({
                        "question": question,
                        "answer": fallback_answer,
                        "question_step": question_step,
                        "answer_step": question_step,
                    })
                    self.save_agent_oracle_dialog()
                    yield {
                        "type": "agent_message",
                        "question": question,
                        "answer": fallback_answer,
                        "reasoning": parsed.get("reasoning", ""),
                        "question_step": question_step,
                        "answer_step": question_step,
                        "ok": False,
                        "tick": tick_info,
                        "total_ticks": steps,
                        "frame": self.render_frame(),
                        "campaign_state": self.get_campaign_snapshot(),
                        **_agent_ws_knowledge_fields(parsed),
                    }
                    continue
            yield {
                "type": "agent_system_notice",
                "message": (
                    "Agent still returned an invalid format after the operator notice "
                    "(no action or question)."
                ),
                "reasoning": parsed.get("reasoning", ""),
                "tick": step_idx + 1,
                "total_ticks": steps,
                "frame": self.render_frame(),
                "campaign_state": self.get_campaign_snapshot(),
            }
            return
        if step_count == 0:
            yield {
                "type": "agent_system_notice",
                "message": "Agent did not produce an action or question.",
                "tick": steps,
                "total_ticks": steps,
                "frame": self.render_frame(),
                "campaign_state": self.get_campaign_snapshot(),
            }

    async def handle_agent_direct_chat(self, message: str) -> AsyncIterator[dict]:
        text = str(message or "").strip()
        if not text:
            yield {"type": "agent_direct_chat_status", "active": self.agent_direct_chat_active, "ok": False, "error": "empty message"}
            return

        if self._is_direct_chat_end_message(text):
            self.end_agent_direct_chat()
            yield {
                "type": "agent_direct_chat_status",
                "active": False,
                "ok": True,
                "ended": True,
                "message": "Direct agent chat ended.",
            }
            return

        if not self.agent_direct_chat_active:
            self.start_agent_direct_chat()

        agent = self.ensure_active_agent()
        history = self._build_direct_chat_llm_history()
        self.last_agent_user_prompt = text
        self.last_agent_system_message = DEFAULT_AGENT_SYSTEM_MESSAGE
        raw_answer = await asyncio.to_thread(
            lambda: agent.chat(
                user_message=text,
                system_message=DEFAULT_AGENT_SYSTEM_MESSAGE,
                gen=self._current_agent_gen_config(),
                history=history or None,
            )
        )
        parsed = parse_agent_answer(raw_answer)
        self.agent_direct_chat_turns.append(
            {"message": text, "response": strip_to_database_tags(raw_answer).strip()}
        )

        yield {
            "type": "agent_direct_chat_reply",
            "human_message": text,
            "raw_answer": raw_answer,
            "reasoning": parsed.get("reasoning", ""),
            "active": True,
            "ok": True,
            **_agent_ws_knowledge_fields(parsed),
        }

        if "action" in parsed:
            display_action = (
                str(parsed.get("action") or parsed.get("action_raw") or "").strip().upper()
                if self.is_arc_game()
                else format_action_for_ui(parsed, world_mode=self.texture_theme)
            )
            agent.record_action(display_action)
            reward, done, frames = self.apply_actions(parsed["action"])
            yield {
                "type": "agent_action",
                "action": display_action,
                "reasoning": parsed.get("reasoning", ""),
                "reward": reward,
                "done": done,
                "frames": frames,
                **_agent_ws_knowledge_fields(parsed),
            }
            if done:
                return
            return

        if "question" in parsed:
            if self.is_arc_game():
                self.arc_questions_count += 1
            agent.record_question()
            question = parsed["question"]
            question_step = self._record_arc_decision_step() if self.is_arc_game() else 0
            yield {
                "type": "agent_message",
                "question": question,
                "answer": "",
                "reasoning": parsed.get("reasoning", ""),
                "question_step": question_step,
                "ok": True,
                "pending": True,
            }
            if self.interaction_mode == "human":
                self.pending_agent_question = question
                self.pending_agent_question_step = question_step
                return
            try:
                answer_text = self.handle_oracle_question(
                    question,
                    run_code=True,
                    forced_expert=self._oracle_expert_for_agent_question(),
                    question_step=question_step or None,
                    answer_step=question_step or None,
                )
                yield {
                    "type": "agent_message",
                    "question": question,
                    "answer": answer_text,
                    "reasoning": parsed.get("reasoning", ""),
                    "question_step": question_step,
                    "answer_step": question_step,
                    "ok": True,
                    "frame": self.render_frame(),
                }
            except Exception:
                logger.exception("Error handling agent question in direct chat")
                fallback_answer = "Operator couldt answer now. Please make 5 steps in any direction and try again"
                self.oracle_chat_history.append({
                    "question": question,
                    "answer": fallback_answer,
                    "question_step": question_step,
                    "answer_step": question_step,
                })
                self.save_agent_oracle_dialog()
                yield {
                    "type": "agent_message",
                    "question": question,
                    "answer": fallback_answer,
                    "reasoning": parsed.get("reasoning", ""),
                    "question_step": question_step,
                    "answer_step": question_step,
                    "ok": False,
                    "frame": self.render_frame(),
                }
            return

        yield {
            "type": "agent_direct_chat_status",
            "active": True,
            "ok": False,
            "error": "Agent returned an unrecognized format.",
            "raw_answer": raw_answer,
        }


def create_isolated_session() -> Session:
    """
    Create a fresh independent Session instance for a browser tab or benchmark worker.
    """
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=True)
    env_params = env.default_params
    step_fn = jax.jit(env.step)
    texture_theme = DEFAULT_TEXTURE_THEME
    render_fn = make_render_craftax_pixels_jit(texture_theme)
    render_world_base_fn, render_obs_overlay_fn = make_render_world_jit(texture_theme)
    rng = jax.random.PRNGKey(np.random.randint(2**31 - 1))
    rng, r = jax.random.split(rng)
    _, state = env.reset(r, env_params)
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "oracle_config.yaml"
    oracle_config = load_config(path=str(config_path))
    max_agent_steps = resolve_max_agent_steps_per_tick(env_params)
    sess = Session(
        env=env,
        env_params=env_params,
        rng=rng,
        state=state,
        step_fn=step_fn,
        render_fn=render_fn,
        render_world_base_fn=render_world_base_fn,
        render_obs_overlay_fn=render_obs_overlay_fn,
        oracle_config_data=oracle_config,
        max_agent_steps_per_tick=max_agent_steps,
        texture_theme=texture_theme,
        agent_gen_config=GenConfig(
            max_new_tokens=ACTIVE_AGENT_MAX_NEW_TOKENS,
            do_sample=ACTIVE_AGENT_DO_SAMPLE,
            temperature=ACTIVE_AGENT_TEMPERATURE,
            top_p=ACTIVE_AGENT_TOP_P,
        ),
    )
    sess.ensure_texture_mode_consistency()
    sess.campaign_state.reset_progress(state)
    sess.episode_initial_state = state
    sess.trajectory_logger = TrajectoryLogger(persist_tmp=sess.save_trajectory_enabled)
    return sess
