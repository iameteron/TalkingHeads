import asyncio
import logging
import time
from typing import Any

import numpy as np
from fastapi import WebSocket

from .env_file import should_persist_api_keys_to_env, update_api_keys
from .features import app_capabilities_payload, apply_demo_runtime_defaults, companion_bench_allowed, get_app_features
from .leaderboard import (
    append_arc_human_result,
    append_companion_research_result,
    get_arc_human_leaderboard,
    get_companion_leaderboard,
    get_human_leaderboard,
)
from .companion_bench import BenchmarkRuntimeOverrides, CompanionBenchRunner, DEFAULT_COMPANION_MAX_TICKS_PER_TASK
from .runtime import (
    ALL_EXPERTS,
    DEFAULT_AGENT_GOAL,
    GAME_KIND_ARC_AGI,
    Session,
    describe_model_not_found_error,
)
from oracle.prompts.prompt_generation import list_megaprompt_configs_for_world_mode
from .trajectory_dashboard import (
    delete_trajectories,
    list_trajectories,
    play_history_for_trajectory,
    rename_trajectory,
    short_history_for_trajectory,
    stats_for_trajectories,
)


logger = logging.getLogger(__name__)


def _format_exception_for_client(exc: BaseException, *, max_chars: int = 800) -> str:
    detail = str(exc or "").strip()
    if not detail:
        detail = exc.__class__.__name__
    detail = " ".join(detail.split())
    if len(detail) > max_chars:
        detail = f"{detail[:max_chars].rstrip()}..."
    return f"{exc.__class__.__name__}: {detail}"


class RuntimeConfigService:
    def get_config(self, sess: Session) -> dict:
        sess.ensure_texture_mode_consistency()
        apply_demo_runtime_defaults(sess)
        megaprompt_options = list_megaprompt_configs_for_world_mode(
            sess.texture_theme,
            game_kind=sess.game_kind,
        )
        if sess.megaprompt_config_name and sess.megaprompt_config_name not in megaprompt_options:
            megaprompt_options = sorted(set(megaprompt_options) | {sess.megaprompt_config_name})
        return {
            "interaction_mode": sess.interaction_mode,
            "allowed_experts": sess.allowed_experts,
            "forced_expert": sess.forced_expert,
            "default_agent_steps": sess.default_agent_steps,
            "max_agent_steps_per_tick": sess.max_agent_steps_per_tick,
            "megaprompt_config_name": sess.megaprompt_config_name,
            "arc_prompt_extra": sess.arc_prompt_extra,
            "megaprompt_options": megaprompt_options,
            "all_experts": ALL_EXPERTS,
            "expert_models": sess.get_expert_models(),
            "expert_modes": sess.get_expert_modes(),
            "active_agent_model": sess.active_agent_model,
            "active_agent_mode": sess.active_agent_mode,
            "active_agent_do_sample": bool((sess.agent_gen_config or sess._current_agent_gen_config()).do_sample),
            "active_agent_temperature": float((sess.agent_gen_config or sess._current_agent_gen_config()).temperature),
            "active_agent_top_p": float((sess.agent_gen_config or sess._current_agent_gen_config()).top_p),
            "agent_direct_chat_active": sess.agent_direct_chat_active,
            "campaign_state": sess.get_campaign_snapshot(),
            "exo_planet_enabled": sess.texture_theme == "exo-planet",
            "game_kind": sess.game_kind,
            "arc_game_id": sess.arc_game_id,
            "arc_game_options": sess.get_arc_game_options(),
            "player_name": sess.player_name,
            "player_avatar_id": sess.player_avatar_id,
            "model_check": sess.validate_active_models(),
            **app_capabilities_payload(),
            **sess.api_keys_status(),
        }

    def update_config(self, sess: Session, payload: dict) -> dict:
        world_mode_changed = False
        try:
            features = get_app_features()
            api_key_updates: dict[str, str | None] = {}
            if features.settings_api_keys:
                if "hf_token" in payload:
                    api_key_updates["hf_token"] = payload.get("hf_token")
                if "openrouter_api_key" in payload:
                    api_key_updates["openrouter_api_key"] = payload.get("openrouter_api_key")
            if api_key_updates:
                sess.set_api_secrets(
                    hf_token=api_key_updates.get("hf_token"),
                    openrouter_api_key=api_key_updates.get("openrouter_api_key"),
                )
                update_api_keys(
                    **api_key_updates,
                    persist_to_env=should_persist_api_keys_to_env(),
                )
                sess.active_agent = None

            if "game_kind" in payload or "arc_game_id" in payload:
                sess.set_game_kind(
                    str(payload.get("game_kind") or sess.game_kind),
                    arc_game_id=payload.get("arc_game_id") or sess.arc_game_id,
                )
                world_mode_changed = True

            if "interaction_mode" in payload:
                sess.set_interaction_mode(str(payload.get("interaction_mode")))
            if "player_name" in payload or "player_nickname" in payload or "player_avatar_id" in payload:
                player_name = payload.get("player_name")
                if player_name is None and "player_nickname" in payload:
                    player_name = payload.get("player_nickname")
                sess.set_player_profile(
                    name=player_name if ("player_name" in payload or "player_nickname" in payload) else None,
                    avatar_id=payload.get("player_avatar_id") if "player_avatar_id" in payload else None,
                )
            if "allowed_experts" in payload:
                raw_experts = payload.get("allowed_experts") or []
                if not isinstance(raw_experts, list):
                    raise ValueError("allowed_experts must be a list")
                sess.set_allowed_experts([str(x) for x in raw_experts])
            if "forced_expert" in payload:
                sess.set_forced_expert(payload.get("forced_expert"))
            if "default_agent_steps" in payload:
                sess.set_default_agent_steps(int(payload.get("default_agent_steps")))
            if "exo_planet_enabled" in payload and sess.game_kind != GAME_KIND_ARC_AGI:
                world_mode_changed = sess.set_texture_theme(
                    "exo-planet" if bool(payload.get("exo_planet_enabled")) else "craftax"
                ) or world_mode_changed
            if features.observation_format_selection and "megaprompt_config_name" in payload:
                sess.set_megaprompt_config_name(str(payload.get("megaprompt_config_name")))
            if features.arc_prompt_override and "arc_prompt_extra" in payload:
                sess.set_arc_prompt_extra(str(payload.get("arc_prompt_extra") or ""))
            if features.expert_model_settings and "expert_models" in payload:
                sess.set_expert_models(payload.get("expert_models") or {})
            if features.expert_model_settings and "expert_modes" in payload:
                sess.set_expert_modes(payload.get("expert_modes") or {})
            if features.model_selection and "active_agent_model" in payload:
                sess.set_active_agent_model(str(payload.get("active_agent_model")))
            if features.model_selection and "active_agent_mode" in payload:
                sess.set_active_agent_mode(str(payload.get("active_agent_mode")))
            if features.model_selection and (
                "active_agent_do_sample" in payload
                or "active_agent_temperature" in payload
                or "active_agent_top_p" in payload
            ):
                current_cfg = sess.agent_gen_config or sess._current_agent_gen_config()
                do_sample = (
                    bool(payload.get("active_agent_do_sample"))
                    if "active_agent_do_sample" in payload
                    else bool(current_cfg.do_sample)
                )
                temperature = (
                    float(payload.get("active_agent_temperature"))
                    if "active_agent_temperature" in payload
                    else float(current_cfg.temperature)
                )
                top_p = (
                    float(payload.get("active_agent_top_p"))
                    if "active_agent_top_p" in payload
                    else float(current_cfg.top_p)
                )
                sess.set_agent_sampling(
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            if "campaign_enabled" in payload:
                sess.set_campaign_enabled(bool(payload.get("campaign_enabled")))
            apply_demo_runtime_defaults(sess)
            sess.apply_unified_agent_gateway()
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.exception("session config update failed")
            return {"ok": False, "error": str(e)}
        return self._config_response(sess, ok=True, include_frame=world_mode_changed)

    def reset_config(self, sess: Session) -> dict:
        try:
            sess.reset_runtime_config_to_defaults()
            apply_demo_runtime_defaults(sess, apply_model_default=True)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return self._config_response(sess, ok=True)

    def _config_response(self, sess: Session, ok: bool, *, include_frame: bool = False) -> dict:
        sess.ensure_texture_mode_consistency()
        apply_demo_runtime_defaults(sess)
        response = {
            "ok": ok,
            "interaction_mode": sess.interaction_mode,
            "allowed_experts": sess.allowed_experts,
            "forced_expert": sess.forced_expert,
            "default_agent_steps": sess.default_agent_steps,
            "max_agent_steps_per_tick": sess.max_agent_steps_per_tick,
            "megaprompt_config_name": sess.megaprompt_config_name,
            "arc_prompt_extra": sess.arc_prompt_extra,
            "megaprompt_options": list_megaprompt_configs_for_world_mode(
                sess.texture_theme,
                game_kind=sess.game_kind,
            ),
            "expert_models": sess.get_expert_models(),
            "expert_modes": sess.get_expert_modes(),
            "active_agent_model": sess.active_agent_model,
            "active_agent_mode": sess.active_agent_mode,
            "active_agent_do_sample": bool((sess.agent_gen_config or sess._current_agent_gen_config()).do_sample),
            "active_agent_temperature": float((sess.agent_gen_config or sess._current_agent_gen_config()).temperature),
            "active_agent_top_p": float((sess.agent_gen_config or sess._current_agent_gen_config()).top_p),
            "agent_direct_chat_active": sess.agent_direct_chat_active,
            "campaign_state": sess.get_campaign_snapshot(),
            "exo_planet_enabled": sess.texture_theme == "exo-planet",
            "game_kind": sess.game_kind,
            "arc_game_id": sess.arc_game_id,
            "arc_game_options": sess.get_arc_game_options(),
            "player_name": sess.player_name,
            "player_avatar_id": sess.player_avatar_id,
            "model_check": sess.validate_active_models(),
            **app_capabilities_payload(),
            **sess.api_keys_status(),
        }
        if include_frame:
            response["world_mode_reset"] = True
            response["default_agent_goal"] = sess.agent_goal
            response["frame"] = sess.render_frame()
        return response


class MessagingService:
    def __init__(self, safe_send_json):
        self._safe_send_json = safe_send_json

    async def _stream_agent_tick_payloads(self, ws: WebSocket, sess: Session, tick_stream) -> bool:
        """Stream agent tick events. Returns True when paused for a human operator answer."""
        human_paused = False
        try:
            async for payload in tick_stream:
                if sess.stop_agent_requested:
                    break
                if payload.get("type") in {
                    "agent_action",
                    "agent_message",
                    "agent_system_notice",
                    "agent_operator_notice",
                } and "frame" not in payload and "frames" not in payload:
                    payload["frame"] = sess.render_frame()
                payload.setdefault("campaign_state", sess.get_campaign_snapshot())
                if not await self._safe_send_json(ws, payload):
                    sess.stop_agent_requested = True
                    break
                if payload.get("type") == "agent_question_pending" and sess.interaction_mode == "human":
                    human_paused = True
                if sess.stop_agent_requested:
                    break
                await asyncio.sleep(0.05)
        finally:
            aclose = getattr(tick_stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except (GeneratorExit, asyncio.CancelledError):
                    pass
                except Exception:
                    logger.exception("Failed to close agent tick stream")
        return human_paused

    async def _finish_agent_tick(self, ws: WebSocket, sess: Session, *, steps: int) -> None:
        await self._safe_send_json(
            ws,
            {
                "type": "agent_tick_complete",
                "total_ticks": steps,
                "stopped": sess.stop_agent_requested,
                "campaign_state": sess.get_campaign_snapshot(),
            },
        )

    async def _finalize_agent_tick_run(
        self,
        ws: WebSocket,
        sess: Session,
        *,
        steps: int,
        human_paused: bool,
    ) -> None:
        if human_paused and sess.stop_agent_requested:
            sess.pending_human_tick = None
            sess.pending_agent_question = ""
            sess.pending_agent_question_step = 0
            await self._finish_agent_tick(ws, sess, steps=steps)
        elif not human_paused:
            await self._finish_agent_tick(ws, sess, steps=steps)

    async def run_agent_tick(self, ws: WebSocket, sess: Session, steps: int, goal: str) -> None:
        steps = sess.clamp_agent_steps(steps)
        human_paused = False
        try:
            human_paused = await self._stream_agent_tick_payloads(
                ws,
                sess,
                sess.handle_agent_tick(steps=steps, goal=goal),
            )
            await self._finalize_agent_tick_run(
                ws, sess, steps=steps, human_paused=human_paused
            )
        except asyncio.CancelledError:
            if not human_paused:
                await self._finish_agent_tick(ws, sess, steps=steps)
        except Exception as e:
            logger.exception("Error in agent_tick")
            model_error = describe_model_not_found_error(e)
            if model_error:
                model = sess.active_agent_model
                await self._safe_send_json(
                    ws,
                    {
                        "type": "agent_system_notice",
                        "error_kind": "model_not_found",
                        "model": model,
                        "message": (
                            f'Model "{model}" was rejected by OpenRouter. '
                            "Open Settings and set a valid OpenRouter model id. "
                            f"Details: {model_error}"
                        ),
                    },
                )
            else:
                detail = _format_exception_for_client(e)
                await self._safe_send_json(
                    ws,
                    {
                        "type": "agent_system_notice",
                        "message": (
                            "Agent tick failed unexpectedly. "
                            f"Details: {detail}"
                        ),
                        "error_kind": "agent_tick_failed",
                        "error_detail": detail,
                    },
                )
            await self._finish_agent_tick(ws, sess, steps=steps)
        finally:
            if not human_paused:
                sess.stop_agent_requested = False

    async def run_agent_tick_resume(
        self,
        ws: WebSocket,
        sess: Session,
        *,
        steps: int,
        manual_goal: str,
        resume_step_idx: int,
        step_count: int,
    ) -> None:
        steps = sess.clamp_agent_steps(steps)
        if sess.stop_agent_requested:
            await self._finish_agent_tick(ws, sess, steps=steps)
            return
        human_paused = False
        try:
            human_paused = await self._stream_agent_tick_payloads(
                ws,
                sess,
                sess.handle_agent_tick(
                    steps,
                    manual_goal,
                    start_step_idx=resume_step_idx,
                    step_count=step_count,
                    resume_after_human=True,
                ),
            )
            await self._finalize_agent_tick_run(
                ws, sess, steps=steps, human_paused=human_paused
            )
        except asyncio.CancelledError:
            if not human_paused:
                await self._finish_agent_tick(ws, sess, steps=steps)
        except Exception:
            logger.exception("Error resuming agent tick after human operator answer")
            await self._safe_send_json(
                ws,
                {
                    "type": "agent_system_notice",
                    "message": (
                        "Agent could not continue after the operator answer. "
                        "Try running another agent tick."
                    ),
                },
            )
            await self._finish_agent_tick(ws, sess, steps=steps)
        finally:
            if not human_paused:
                sess.stop_agent_requested = False

    async def run_companion_research(
        self,
        ws: WebSocket,
        sess: Session,
        *,
        knowledge_source: str,
        max_ticks_per_task: int,
        model: str,
        mode: str,
    ) -> None:
        from .campaign_mode import campaign_tasks_for_world_mode
        from .companion_bench import (
            DEFAULT_COMPANION_MAX_TICKS_PER_TASK,
            _prepare_research_knowledge,
        )
        from .knowledge_paths import (
            ensure_play_session_knowledge,
            play_knowledge_paths_for_session,
        )
        from oracle.knowledge import use_knowledge_paths

        model_name = str(model or sess.active_agent_model).strip()
        mode_name = str(mode or sess.active_agent_mode).strip().lower()
        if model_name:
            sess.set_active_agent_model(model_name)
        if mode_name:
            sess.set_active_agent_mode(mode_name)

        world_mode = sess.texture_theme
        baseline_json, baseline_txt = play_knowledge_paths_for_session(sess)
        _prepare_research_knowledge(
            knowledge_source=str(knowledge_source or "base"),
            target_json=baseline_json,
            target_txt=baseline_txt,
            world_mode=world_mode,
        )
        ensure_play_session_knowledge(sess, baseline_json, baseline_txt)

        # Step budget is for the whole run (all levels combined), not per level.
        total_budget = max(1, int(max_ticks_per_task or DEFAULT_COMPANION_MAX_TICKS_PER_TASK))
        sess.companion_research_active = True
        sess.companion_research_task_max_ticks = total_budget
        sess.companion_research_task_ticks = 0
        sess.stop_agent_requested = False

        research_complete = False
        stopped = False
        research_questions = 0
        try:
            with use_knowledge_paths(json_path=baseline_json, txt_path=baseline_txt):
                sess.set_campaign_enabled(True)
                sess.reset()
                campaign_tasks = campaign_tasks_for_world_mode(sess._campaign_world_mode())
                if campaign_tasks:
                    sess.companion_research_task_index = 1

                await self._safe_send_json(
                    ws,
                    {
                        "type": "companion_research_started",
                        "knowledge_source": str(knowledge_source or "base"),
                        "campaign_state": sess.get_campaign_snapshot(),
                        "companion_research": sess.get_companion_research_snapshot(),
                    },
                )

                task_idx = 0
                total_ticks = 0
                task_initial_state = sess.state

                while (
                    task_idx < len(campaign_tasks)
                    and not sess.stop_agent_requested
                    and total_ticks < total_budget
                ):
                    task = campaign_tasks[task_idx]
                    display_idx = task_idx + 1
                    sess.companion_research_task_index = display_idx

                    while (
                        total_ticks < total_budget
                        and not task.check(task_initial_state, sess.state)
                        and not sess.stop_agent_requested
                    ):
                        human_waited = False
                        async for payload in sess.handle_agent_tick(steps=1, goal=task.goal):
                            if payload.get("type") in {
                                "agent_action",
                                "agent_message",
                                "agent_system_notice",
                                "agent_operator_notice",
                                "agent_question_pending",
                            } and "frame" not in payload and "frames" not in payload:
                                payload["frame"] = sess.render_frame()
                            payload.setdefault("campaign_state", sess.get_campaign_snapshot())
                            payload["companion_research"] = sess.get_companion_research_snapshot()
                            if not await self._safe_send_json(ws, payload):
                                sess.stop_agent_requested = True
                                break
                            if payload.get("type") == "agent_question_pending":
                                research_questions += 1
                            if (
                                payload.get("type") == "agent_question_pending"
                                and sess.interaction_mode == "human"
                            ):
                                if sess.pending_human_tick is not None:
                                    sess.pending_human_tick["companion_research"] = True
                                human_waited = True
                                while (
                                    sess.pending_human_tick is not None
                                    and not sess.stop_agent_requested
                                ):
                                    await asyncio.sleep(0.1)
                                if sess.stop_agent_requested:
                                    break
                            await asyncio.sleep(0.05)
                        if sess.stop_agent_requested:
                            break
                        total_ticks += 1
                        sess.companion_research_task_ticks = total_ticks
                        if human_waited:
                            await self._safe_send_json(
                                ws,
                                {
                                    "type": "companion_research_progress",
                                    "companion_research": sess.get_companion_research_snapshot(),
                                    "campaign_state": sess.get_campaign_snapshot(),
                                },
                            )

                    if task.check(task_initial_state, sess.state):
                        sess._record_completed_level_steps()
                        sess._checkpoint_leaderboard_attempt("level_complete")
                        task_idx += 1
                        task_initial_state = sess.state
                    else:
                        break

                research_complete = task_idx >= len(campaign_tasks)
                stopped = sess.stop_agent_requested
                completed_tasks = min(task_idx, len(campaign_tasks))
                if research_complete:
                    completed_tasks = len(campaign_tasks)
                mean_research_q = (
                    round(research_questions / max(completed_tasks, 1), 2)
                    if research_questions
                    else 0.0
                )
                append_companion_research_result(
                    model=model_name,
                    max_task=completed_tasks,
                    total_questions=research_questions,
                    mean_questions=mean_research_q,
                    max_ticks_per_task=total_budget,
                    research_complete=research_complete,
                    source="websocket",
                )
        except asyncio.CancelledError:
            stopped = True
            raise
        except Exception:
            logger.exception("Error in companion_research")
            await self._safe_send_json(
                ws,
                {
                    "type": "agent_system_notice",
                    "message": "Companion research failed unexpectedly.",
                    "campaign_state": sess.get_campaign_snapshot(),
                },
            )
        finally:
            stopped = stopped or sess.stop_agent_requested
            finish_reason = "research_complete" if research_complete else "research_stopped"
            sess._finalize_leaderboard_attempt(finish_reason)
            sess.companion_research_active = False
            await self._safe_send_json(
                ws,
                {
                    "type": "companion_research_complete",
                    "research_complete": research_complete,
                    "stopped": stopped,
                    "has_own_knowledge": baseline_json.exists() or baseline_txt.exists(),
                    "campaign_state": sess.get_campaign_snapshot(),
                    "companion_research": {
                        "active": False,
                        "complete": research_complete,
                        "progress_pct": 100 if research_complete else sess.get_companion_research_snapshot().get("progress_pct", 0),
                    },
                    "frame": sess.render_frame(),
                },
            )
            sess.stop_agent_requested = False

    async def handle_oracle_ask(self, ws: WebSocket, sess: Session, msg: dict) -> None:
        question_raw = str(msg.get("question", ""))
        question = question_raw.strip()
        run_code = bool(msg.get("run_code", True))
        forced_expert = msg.get("forced_expert")
        if question.lower().startswith("hint:"):
            hint_text = question[len("hint:") :].strip()
            if hint_text:
                sess.hints.append(hint_text)
            await self._safe_send_json(
                ws,
                {
                    "type": "oracle_answer",
                    "ok": True,
                    "question": question_raw,
                    "answer": "Hint saved for the active agent.",
                    "campaign_state": sess.get_campaign_snapshot(),
                },
            )
            return
        if not question:
            await self._safe_send_json(
                ws,
                {
                    "type": "oracle_answer",
                    "ok": False,
                    "error": "empty question",
                    "campaign_state": sess.get_campaign_snapshot(),
                },
            )
            return
        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                answer_text = sess.handle_oracle_question(
                    question,
                    run_code=run_code,
                    forced_expert=forced_expert,
                )
                payload: dict[str, Any] = {"type": "oracle_answer", "ok": True, "question": question, "answer": answer_text}
                payload["campaign_state"] = sess.get_campaign_snapshot()
                if run_code:
                    payload["frame"] = sess.render_frame()
                await self._safe_send_json(ws, payload)
                return
            except Exception as e:
                logger.exception("Error while handling oracle_ask")
                model_error = describe_model_not_found_error(e)
                if model_error:
                    model = sess.active_agent_model
                    answer_text = (
                        f'Model "{model}" was rejected by OpenRouter. '
                        "Open Settings and set a valid OpenRouter model id. "
                        f"Details: {model_error}"
                    )
                    entry: dict[str, Any] = {"question": question, "answer": answer_text}
                    if sess.game_kind == GAME_KIND_ARC_AGI:
                        entry["answer_step"] = sess.next_arc_step_number()
                    sess.oracle_chat_history.append(entry)
                    sess.save_agent_oracle_dialog()
                    await self._safe_send_json(
                        ws,
                        {
                            "type": "oracle_answer",
                            "ok": False,
                            "error_kind": "model_not_found",
                            "model": model,
                            "question": question,
                            "answer": answer_text,
                            "campaign_state": sess.get_campaign_snapshot(),
                        },
                    )
                    return
                answer_text = "Operator couldt answer now. Please make 5 steps in any direction and try again"
                entry = {"question": question, "answer": answer_text}
                if sess.game_kind == GAME_KIND_ARC_AGI:
                    entry["answer_step"] = sess.next_arc_step_number()
                sess.oracle_chat_history.append(entry)
                sess.save_agent_oracle_dialog()
                await self._safe_send_json(
                    ws,
                    {
                        "type": "oracle_answer",
                        "ok": False,
                        "question": question,
                        "answer": answer_text,
                        "campaign_state": sess.get_campaign_snapshot(),
                    },
                )

    async def handle_agent_direct_chat(self, ws: WebSocket, sess: Session, msg: dict) -> None:
        message = str(msg.get("message", ""))
        try:
            async for payload in sess.handle_agent_direct_chat(message):
                if payload.get("type") in {"agent_action", "agent_message", "agent_system_notice", "agent_operator_notice"} and "frame" not in payload and "frames" not in payload:
                    payload["frame"] = sess.render_frame()
                payload.setdefault("campaign_state", sess.get_campaign_snapshot())
                if not await self._safe_send_json(ws, payload):
                    break
                await asyncio.sleep(0.05)
        except Exception:
            logger.exception("Error in agent_direct_chat")
            await self._safe_send_json(
                ws,
                {
                    "type": "agent_direct_chat_status",
                    "active": sess.agent_direct_chat_active,
                    "ok": False,
                    "error": "Failed to talk to the agent.",
                },
            )

    async def handle_operator_answer(
        self,
        ws: WebSocket,
        sess: Session,
        msg: dict,
        *,
        spawn_agent_tick_task=None,
    ) -> None:
        question = str(msg.get("question") or "").strip()
        answer = str(msg.get("answer") or "").strip()
        if not question or not answer:
            await self._safe_send_json(ws, {"type": "operator_answer_ack", "ok": False, "error": "Both 'question' and 'answer' must be non-empty."})
            return
        if sess.game_kind == GAME_KIND_ARC_AGI:
            sess.arc_human_answers_count += 1
        pending = sess.pending_human_tick
        try:
            question_step = int(
                msg.get("question_step")
                or getattr(sess, "pending_agent_question_step", 0)
                or (pending or {}).get("question_step")
                or 0
            )
        except (TypeError, ValueError):
            question_step = 0
        answer_step = 0
        if sess.game_kind == GAME_KIND_ARC_AGI:
            answer_step = question_step or sess.current_arc_step_number()
        elif question_step:
            answer_step = question_step
        entry: dict[str, Any] = {"question": question, "answer": answer}
        if question_step > 0:
            entry["question_step"] = question_step
        if answer_step > 0:
            entry["answer_step"] = answer_step
        sess.oracle_chat_history.append(entry)
        sess.save_agent_oracle_dialog()
        sess.pending_agent_question = ""
        sess.pending_agent_question_step = 0
        resuming = bool(pending and not sess.stop_agent_requested)
        await self._safe_send_json(
            ws,
            {
                "type": "operator_answer_ack",
                "ok": True,
                "question": question,
                "answer": answer,
                "question_step": question_step,
                "answer_step": answer_step,
                "resuming": resuming,
            },
        )

        if not pending or sess.stop_agent_requested:
            sess.pending_human_tick = None
            if sess.stop_agent_requested and pending:
                steps = int(pending.get("steps") or sess.default_agent_steps)
                await self._finish_agent_tick(ws, sess, steps=steps)
                sess.stop_agent_requested = False
            return
        if bool(pending.get("companion_research")):
            sess.pending_human_tick = None
            return

        if sess.agent_tick_is_running():
            sess.pending_human_tick = None
            return

        sess.pending_human_tick = None
        steps = int(pending.get("steps") or sess.default_agent_steps)
        manual_goal = str(pending.get("manual_goal") or DEFAULT_AGENT_GOAL)
        resume_step_idx = int(pending.get("resume_step_idx") or 0)
        step_count = int(pending.get("step_count") or 0)
        resume_coro = self.run_agent_tick_resume(
            ws,
            sess,
            steps=steps,
            manual_goal=manual_goal,
            resume_step_idx=resume_step_idx,
            step_count=step_count,
        )
        if spawn_agent_tick_task is not None:
            spawn_agent_tick_task(sess, resume_coro)
        else:
            await resume_coro

    async def handle_step(self, ws: WebSocket, sess: Session, action: Any) -> None:
        started = time.perf_counter()
        if sess.game_kind == GAME_KIND_ARC_AGI:
            if isinstance(action, dict):
                action_name = str(action.get("action") or "").strip().upper()
                if action_name == "ACTION6":
                    action_name = f"ACTION6 {action.get('x')} {action.get('y')}"
                step_started = time.perf_counter()
                reward, done, frame = sess.step(action_name)
            else:
                step_started = time.perf_counter()
                reward, done, frame = sess.step(str(action))
            player_position = ""
        else:
            step_started = time.perf_counter()
            reward, done, frame = sess.step(int(action))
            player_position = str(np.array(sess.state.player_position).tolist())
        step_elapsed_ms = round((time.perf_counter() - step_started) * 1000, 1)
        total_elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        detail_timing = dict(getattr(sess, "last_step_debug_timing", {}) or {})
        await self._safe_send_json(
            ws,
            {
                "type": "frame",
                "reward": reward,
                "player_position": player_position,
                "done": done,
                "frame": frame,
                "campaign_state": sess.get_campaign_snapshot(),
                "debug_timing": {
                    **detail_timing,
                    "step_ms": step_elapsed_ms,
                    "server_total_ms": total_elapsed_ms,
                },
            },
        )

    async def handle_reset(self, ws: WebSocket, sess: Session) -> None:
        frame = sess.reset()
        await self._safe_send_json(
            ws, {"type": "frame", "frame": frame, "campaign_state": sess.get_campaign_snapshot()}
        )


class StatsDashboardService:
    def get_model_leaderboard(
        self,
        extra_test_rows: list[dict[str, Any]] | None = None,
    ) -> dict:
        payload = get_companion_leaderboard(extra_test_rows=extra_test_rows)
        payload["total_runs"] = int(payload.get("total_tests") or 0)
        return payload

    def get_campaign_benchmark(self, since: str | None = None) -> dict:
        from .campaign_benchmark import get_campaign_benchmark

        return get_campaign_benchmark(since=since)

    def get_arc_human_score(self, sess: Session) -> dict:
        if sess.game_kind != GAME_KIND_ARC_AGI:
            return {"ok": False, "error": "ARC human leaderboard is only available in ARC-AGI-3 games."}
        score = sess.arc_final_score or sess.build_arc_human_score(sess.render_frame())
        return {"ok": True, "score": score}

    def submit_arc_human_score(self, sess: Session, payload: dict[str, Any]) -> dict:
        if sess.game_kind != GAME_KIND_ARC_AGI:
            return {"ok": False, "error": "ARC human leaderboard is only available in ARC-AGI-3 games."}
        player_name = str(payload.get("player_name") or payload.get("name") or "").strip()
        if not player_name:
            return {"ok": False, "error": "player_name is required"}
        if sess.arc_score_submitted:
            return {"ok": False, "error": "This ARC attempt was already submitted."}
        score = sess.arc_final_score or sess.build_arc_human_score(sess.render_frame())
        if not sess._arc_score_is_submittable(score):
            return {"ok": False, "error": "ARC score can be submitted after completing at least one level."}
        entry = append_arc_human_result({
            **score,
            "player_name": player_name,
            "player_avatar_id": payload.get("player_avatar_id", sess.player_avatar_id),
        })
        sess.arc_score_submitted = True
        if sess.arc_final_score is not None:
            sess.arc_final_score["submitted"] = True
        return {
            "ok": True,
            "entry": entry,
            "leaderboard": get_arc_human_leaderboard(game_id=str(score.get("game_id") or "")),
        }

    def get_arc_human_leaderboard(
        self,
        limit: int = 100,
        game_id: str | None = None,
    ) -> dict:
        return get_arc_human_leaderboard(limit=limit, game_id=game_id)

    def get_human_leaderboard(
        self,
        *,
        game_kind: str | None = None,
        world_mode: str | None = None,
        arc_game_id: str | None = None,
        limit: int = 100,
    ) -> dict:
        return get_human_leaderboard(
            game_kind=game_kind,
            world_mode=world_mode,
            arc_game_id=arc_game_id,
            limit=limit,
        )

    def get_oracle_statistics(self, sess: Session) -> dict:
        if sess.oracle is None:
            return {"experts": {}}
        return {"experts": sess.oracle.return_statistics()}

    def get_agent_statistics(self, sess: Session) -> dict:
        agent = sess.active_agent
        if agent is None or not hasattr(agent, "total_calls"):
            return {"total_calls": 0, "questions": 0, "actions": 0, "failures": 0, "questions_pct": 0.0, "actions_pct": 0.0}
        total_calls = int(getattr(agent, "total_calls", 0) or 0)
        questions = int(getattr(agent, "questions", 0) or 0)
        actions = int(getattr(agent, "actions", 0) or 0)
        failures = int(getattr(agent, "failures", 0) or 0)
        denom = total_calls if total_calls > 0 else 1
        return {"total_calls": total_calls, "questions": questions, "actions": actions, "failures": failures, "questions_pct": 100.0 * questions / denom, "actions_pct": 100.0 * actions / denom}

    def get_current_trajectory(self, sess: Session) -> dict:
        logger_obj = sess.trajectory_logger
        if logger_obj is None or not logger_obj.steps:
            return {"active": False}
        steps = len(logger_obj.steps)
        actions = 0
        questions = 0
        total_answer_chars = 0
        total_answer_count = 0
        for step in logger_obj.steps:
            parsed = step.parsed or {}
            if "action" in parsed:
                actions += 1
            if "question" in parsed:
                questions += 1
            raw_answer = step.raw_answer or ""
            if isinstance(raw_answer, str) and raw_answer:
                total_answer_chars += len(raw_answer)
                total_answer_count += 1
        mean_answer_len_chars = float(total_answer_chars) / total_answer_count if total_answer_count else 0.0
        return {"active": True, "steps": steps, "actions": actions, "questions": questions, "total_answer_chars": total_answer_chars, "mean_answer_len_chars": mean_answer_len_chars}

    def list_trajectories(self) -> dict:
        return list_trajectories()

    def trajectory_short_history(self, trajectory_id: str) -> dict:
        tid = str(trajectory_id).strip()
        if not tid:
            return {"ok": False, "error": "trajectory_id is required"}
        try:
            payload = short_history_for_trajectory(tid)
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.exception("Failed to load short trajectory history")
            return {"ok": False, "error": str(e)}
        payload["ok"] = True
        return payload

    def trajectory_play_history(self, trajectory_id: str) -> dict:
        tid = str(trajectory_id).strip()
        if not tid:
            return {"ok": False, "error": "trajectory_id is required"}
        try:
            payload = play_history_for_trajectory(tid)
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.exception("Failed to load trajectory play history")
            return {"ok": False, "error": str(e)}
        payload["ok"] = True
        return payload

    def trajectories_stats(self, payload: dict) -> dict:
        ids = payload.get("ids") or []
        if not isinstance(ids, list):
            ids = []
        return stats_for_trajectories([str(x) for x in ids])

    def rename_trajectory(self, payload: dict) -> dict:
        old_id = str(payload.get("id") or "").strip()
        new_name = str(payload.get("display_name") or "").strip()
        if not old_id or not new_name:
            return {"ok": False, "error": "Both 'id' and 'display_name' are required."}
        try:
            meta = rename_trajectory(old_id, new_name)
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        except FileExistsError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.exception("Failed to rename trajectory")
            return {"ok": False, "error": str(e)}
        return {"ok": True, "item": meta}

    def delete_trajectories(self, payload: dict) -> dict:
        ids = payload.get("ids") or []
        if not isinstance(ids, list):
            return {"ok": False, "error": "'ids' must be a list"}
        try:
            deleted = delete_trajectories([str(x) for x in ids])
        except Exception as e:
            logger.exception("Failed to delete trajectories")
            return {"ok": False, "error": str(e)}
        return {"ok": True, "deleted": deleted.get("deleted", [])}

    def get_agent_prompt_preview(self, sess: Session) -> dict:
        if not get_app_features().agent_prompt_debug:
            return {
                "prompt": "",
                "system_message": "",
                "goal": sess.agent_goal,
                "has_prompt": False,
                "disabled": True,
                "error": "Agent prompt debug is disabled in this app profile.",
                "game_kind": sess.game_kind,
                "arc_prompt_extra": "",
            }
        user_prompt, system_message, goal, has_prompt = sess.current_agent_prompt_preview()
        return {
            "prompt": user_prompt,
            "system_message": system_message,
            "goal": goal,
            "has_prompt": has_prompt,
            "game_kind": sess.game_kind,
            "arc_prompt_extra": sess.arc_prompt_extra,
        }


class CompanionBenchService:
    def __init__(self) -> None:
        self._runner = CompanionBenchRunner()

    async def start(self, sess: Session, payload: dict[str, Any]) -> dict[str, Any]:
        if not companion_bench_allowed(sess):
            return {"ok": False, "error": "Companion bench is disabled in this app profile."}
        if sess.game_kind == GAME_KIND_ARC_AGI:
            return {"ok": False, "error": "Companion bench is disabled for ARC-AGI-3 games."}
        model = str(payload.get("model") or sess.active_agent_model).strip()
        mode = str(payload.get("mode") or sess.active_agent_mode).strip().lower()
        megaprompt_config_name = str(
            payload.get("megaprompt_config_name") or sess.megaprompt_config_name
        ).strip()
        phase = str(payload.get("phase") or "research").strip().lower()
        parallel_agents = int(payload.get("parallel_agents") or 3)
        max_ticks_per_task = int(payload.get("max_ticks_per_task") or DEFAULT_COMPANION_MAX_TICKS_PER_TASK)
        cycles = int(payload.get("cycles") or 1)
        task_key = str(payload.get("task_key") or "").strip()
        knowledge_source = str(
            payload.get("knowledge_source") or ("base" if phase == "test" else "own")
        ).strip().lower()
        if not model:
            return {"ok": False, "error": "model is required"}
        if mode not in {"hub", "openrouter"}:
            return {"ok": False, "error": "mode must be 'hub' or 'openrouter'"}
        if not megaprompt_config_name:
            return {"ok": False, "error": "megaprompt_config_name is required"}
        if phase not in {"research", "test"}:
            return {"ok": False, "error": "phase must be 'research' or 'test'"}
        if phase == "test" and not task_key:
            return {"ok": False, "error": "task_key is required for test phase"}
        cfg = sess.agent_gen_config or sess._current_agent_gen_config()
        runtime_overrides = BenchmarkRuntimeOverrides(
            interaction_mode=sess.interaction_mode,
            allowed_experts=list(sess.allowed_experts),
            forced_expert=sess.forced_expert,
            expert_models=sess.get_expert_models(),
            expert_modes=sess.get_expert_modes(),
            active_agent_do_sample=bool(cfg.do_sample),
            active_agent_temperature=float(cfg.temperature),
            active_agent_top_p=float(cfg.top_p),
            hf_token=sess.get_api_secret("HF_TOKEN") or "",
            openrouter_api_key=sess.get_api_secret("OPENROUTER_API_KEY") or "",
        )
        return await self._runner.start(
            phase=phase,
            model=model,
            mode=mode,
            megaprompt_config_name=megaprompt_config_name,
            parallel_agents=parallel_agents,
            max_ticks_per_task=max_ticks_per_task,
            cycles=cycles,
            task_key=task_key,
            knowledge_source=knowledge_source,
            world_mode=sess.texture_theme,
            runtime_overrides=runtime_overrides,
        )

    async def status(self, sess: Session) -> dict[str, Any]:
        world_mode = "arc_agi" if sess.is_arc_game() else sess.texture_theme
        return await self._runner.status(model=sess.active_agent_model, world_mode=world_mode)

    async def stop(self) -> dict[str, Any]:
        return await self._runner.stop()
