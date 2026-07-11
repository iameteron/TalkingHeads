import asyncio
import logging
from pathlib import Path

from .env_file import load_project_env

load_project_env()

from fastapi import APIRouter, Depends, FastAPI, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .play_sessions import (
    SESSION_ID_HEADER,
    attach_session_id_header,
    resolve_play_session,
    session_id_from_header,
)
from .features import apply_demo_runtime_defaults, companion_bench_allowed, get_app_features
from .runtime import DEFAULT_AGENT_GOAL
from .services import (
    CompanionBenchService,
    MessagingService,
    RuntimeConfigService,
    StatsDashboardService,
)


logger = logging.getLogger(__name__)

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"


async def _safe_send_json(ws: WebSocket, data: dict) -> bool:
    """Send JSON over WebSocket; return False if connection already closed."""
    try:
        await ws.send_json(data)
        return True
    except (RuntimeError, WebSocketDisconnect) as e:
        logger.debug("WebSocket send failed (connection likely closed): %s", e)
        return False


def _spawn_agent_tick_task(sess, coro) -> asyncio.Task:
    task = asyncio.create_task(coro)

    def _on_done(done_task: asyncio.Task) -> None:
        sess.detach_agent_tick_task(done_task)

    task.add_done_callback(_on_done)
    sess.attach_agent_tick_task(task)
    return task


async def _handle_agent_stop(
    ws: WebSocket,
    sess,
    messaging_service: MessagingService,
) -> None:
    sess.stop_agent_requested = True
    if sess.pending_human_tick is not None and not sess.companion_research_active:
        pending_steps = int(
            sess.pending_human_tick.get("steps") or sess.default_agent_steps
        )
        sess.pending_human_tick = None
        await messaging_service._finish_agent_tick(ws, sess, steps=pending_steps)
        sess.stop_agent_requested = False
        await _safe_send_json(
            ws,
            {
                "type": "agent_stop_ack",
                "message": "Agent is already stopped.",
                "running": False,
                "campaign_state": sess.get_campaign_snapshot(),
            },
        )
        return

    if sess.pending_human_tick is not None:
        sess.pending_human_tick = None

    task = sess.agent_tick_task
    if task is not None and not task.done():
        await sess.cancel_agent_tick_task()

    running = sess.companion_research_active or sess.agent_tick_is_running()
    if sess.companion_research_active:
        message = (
            "Stop requested. Companion research is stopping."
            if running
            else "Companion research stopped."
        )
    else:
        message = (
            "Stop requested. Any in-flight model response will be ignored "
            "before another action is applied."
            if running
            else "Agent is already stopped."
        )
    await _safe_send_json(
        ws,
        {
            "type": "agent_stop_ack",
            "message": message,
            "running": running,
            "campaign_state": sess.get_campaign_snapshot(),
        },
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[SESSION_ID_HEADER],
)

api_router = APIRouter(prefix="/api")

runtime_config_service = RuntimeConfigService()
stats_service = StatsDashboardService()
companion_bench_service = CompanionBenchService()


def _play_session(
    response: Response,
    session_id: str | None = None,
):
    resolved_id, sess = resolve_play_session(session_id)
    attach_session_id_header(response, resolved_id)
    return sess


@api_router.get("/session_config")
def get_session_config(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return runtime_config_service.get_config(sess)


@api_router.post("/session_config")
def post_session_config(
    payload: dict,
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    # Plain sync endpoint on purpose: FastAPI runs it in the threadpool, so a
    # slow world-mode switch (texture load + JIT warmup, tens of seconds)
    # doesn't freeze the event loop for every other request and websocket.
    sess = _play_session(response, session_id)
    return runtime_config_service.update_config(sess, payload)


@api_router.post("/session_config/reset")
def reset_session_config(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return runtime_config_service.reset_config(sess)


@api_router.get("/reset")
def reset(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    frame = sess.reset()
    return {"frame": frame, "campaign_state": sess.get_campaign_snapshot()}


@api_router.get("/agent_prompt")
def get_agent_prompt(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return stats_service.get_agent_prompt_preview(sess)


@api_router.get("/agent_knowledge")
def get_agent_knowledge(
    response: Response,
    scope: str = "default",
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    from .knowledge_paths import load_session_knowledge, play_knowledge_paths_for_session

    sess = _play_session(response, session_id)
    if sess.is_arc_game():
        return {
            "knowledge": (
                "ARC-AGI-3 sessions do not use the Craftax/Exo knowledge base. "
                "Human operator messages are passed through the dialog history instead."
            ),
            "path": "",
            "json_path": "",
            "disabled": True,
            "game_kind": sess.game_kind,
        }
    json_path, txt_path = play_knowledge_paths_for_session(sess)
    knowledge = load_session_knowledge(sess)

    return {
        "knowledge": knowledge,
        "path": str(txt_path),
        "json_path": str(json_path),
        "scope": "session",
    }


@api_router.get("/oracle_statistics")
def get_oracle_statistics(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return stats_service.get_oracle_statistics(sess)


@api_router.get("/agent_statistics")
def get_agent_statistics(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return stats_service.get_agent_statistics(sess)


@api_router.get("/inventory_icons")
def get_inventory_icons_route(theme: str = "craftax") -> dict:
    from .inventory_icons import INVENTORY_SLOT_ORDER, get_inventory_icons

    return {
        "theme": theme,
        "order": INVENTORY_SLOT_ORDER,
        "icons": get_inventory_icons(theme),
    }


@api_router.get("/model_leaderboard")
def get_model_leaderboard() -> dict:
    return stats_service.get_model_leaderboard()


@api_router.get("/campaign_benchmark")
def get_campaign_benchmark(since: str | None = None) -> dict:
    return stats_service.get_campaign_benchmark(since=since)


@api_router.get("/arc_game_preview")
def get_arc_game_preview_route(game_id: str = Query(..., min_length=1)) -> dict:
    from .arc_agi_adapter import (
        ArcAgiGameUnavailableError,
        ArcAgiUnavailableError,
        get_arc_game_preview,
    )

    try:
        return get_arc_game_preview(game_id)
    except (ArcAgiUnavailableError, ArcAgiGameUnavailableError) as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get("/arc_human_score")
def get_arc_human_score(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return stats_service.get_arc_human_score(sess)


@api_router.post("/arc_human_score")
def post_arc_human_score(
    payload: dict,
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return stats_service.submit_arc_human_score(sess, payload)


@api_router.get("/arc_human_leaderboard")
def get_arc_human_leaderboard(
    limit: int = 100,
    game_id: str | None = None,
) -> dict:
    return stats_service.get_arc_human_leaderboard(limit=limit, game_id=game_id)


@api_router.get("/human_leaderboard")
def get_human_leaderboard(
    game_kind: str | None = None,
    world_mode: str | None = None,
    arc_game_id: str | None = None,
    limit: int = 100,
) -> dict:
    return stats_service.get_human_leaderboard(
        game_kind=game_kind,
        world_mode=world_mode,
        arc_game_id=arc_game_id,
        limit=limit,
    )


@api_router.get("/trajectory_current")
def get_current_trajectory(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return stats_service.get_current_trajectory(sess)


@api_router.get("/companion_bench/status")
async def get_companion_bench_status(
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return await companion_bench_service.status(sess)


@api_router.post("/companion_bench/start")
async def post_companion_bench_start(
    payload: dict,
    response: Response,
    session_id: str | None = Depends(session_id_from_header),
) -> dict:
    sess = _play_session(response, session_id)
    return await companion_bench_service.start(sess, payload)


@api_router.post("/companion_bench/stop")
async def post_companion_bench_stop() -> dict:
    return await companion_bench_service.stop()


@api_router.get("/trajectories")
def get_trajectories() -> dict:
    return stats_service.list_trajectories()


@api_router.get("/trajectories/{trajectory_id}/short_history")
def get_trajectory_short_history(trajectory_id: str) -> dict:
    return stats_service.trajectory_short_history(trajectory_id)


@api_router.get("/trajectories/{trajectory_id}/play_history")
def get_trajectory_play_history(trajectory_id: str) -> dict:
    return stats_service.trajectory_play_history(trajectory_id)


@api_router.post("/trajectories/stats")
async def post_trajectories_stats(payload: dict) -> dict:
    return stats_service.trajectories_stats(payload)


@api_router.post("/trajectories/rename")
async def post_trajectories_rename(payload: dict) -> dict:
    return stats_service.rename_trajectory(payload)


@api_router.post("/trajectories/delete")
async def post_trajectories_delete(payload: dict) -> dict:
    return stats_service.delete_trajectories(payload)


app.include_router(api_router)


@app.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    session_id: str | None = Query(default=None),
) -> None:
    await ws.accept()
    resolved_id, sess = resolve_play_session(session_id)
    messaging_service = MessagingService(_safe_send_json)

    # Each websocket attachment needs a full map snapshot — HTTP config saves may
    # have rendered the world server-side without the browser ever applying it.
    sess.invalidate_world_map_cache()

    await _safe_send_json(
        ws,
        {
            "type": "frame",
            "frame": sess.render_frame(),
            "campaign_state": sess.get_campaign_snapshot(),
            "play_session_id": resolved_id,
        },
    )

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "agent_stop":
                await _handle_agent_stop(ws, sess, messaging_service)
                continue

            if mtype == "reset":
                await messaging_service.handle_reset(ws, sess)
                continue

            if mtype == "request_full_map":
                # Client detected its base map belongs to an older world
                # snapshot (map_epoch mismatch) and needs a fresh full render.
                sess.invalidate_world_map_cache()
                await _safe_send_json(
                    ws,
                    {
                        "type": "frame",
                        "frame": sess.render_frame(),
                        "campaign_state": sess.get_campaign_snapshot(),
                    },
                )
                continue

            if mtype == "trajectory_save_toggle":
                enabled = bool(msg.get("enabled", False))
                sess.save_trajectory_enabled = enabled
                if sess.trajectory_logger is not None:
                    sess.trajectory_logger.persist_tmp = enabled
                await _safe_send_json(
                    ws,
                    {
                        "type": "trajectory_save_status",
                        "enabled": sess.save_trajectory_enabled,
                    },
                )
                continue

            if mtype == "campaign_toggle":
                enabled = bool(msg.get("enabled", False))
                snapshot = sess.set_campaign_enabled(enabled)
                await _safe_send_json(
                    ws,
                    {
                        "type": "campaign_status",
                        "ok": True,
                        "campaign_state": snapshot,
                    },
                )
                continue

            if mtype == "campaign_phase2_start":
                level_key = str(msg.get("level_key", "")).strip()
                try:
                    snapshot = sess.start_campaign_phase2_level(level_key)
                    await _safe_send_json(
                        ws,
                        {
                            "type": "campaign_status",
                            "ok": True,
                            "campaign_state": snapshot,
                            "frame": sess.render_frame(),
                        },
                    )
                except ValueError as e:
                    await _safe_send_json(
                        ws,
                        {"type": "campaign_status", "ok": False, "error": str(e)},
                    )
                continue

            if mtype == "step":
                if getattr(sess, "game_kind", "") == "arc_agi":
                    payload = {"action": msg.get("action")}
                    if msg.get("x") is not None:
                        payload["x"] = msg.get("x")
                    if msg.get("y") is not None:
                        payload["y"] = msg.get("y")
                    await messaging_service.handle_step(ws, sess, payload)
                else:
                    action = int(msg["action"])
                    await messaging_service.handle_step(ws, sess, action)
                continue

            if mtype == "oracle_ask":
                await messaging_service.handle_oracle_ask(ws, sess, msg)
                continue

            if mtype == "operator_answer":
                await messaging_service.handle_operator_answer(
                    ws,
                    sess,
                    msg,
                    spawn_agent_tick_task=_spawn_agent_tick_task,
                )
                continue

            if mtype == "agent_direct_chat":
                await messaging_service.handle_agent_direct_chat(ws, sess, msg)
                continue

            if mtype == "companion_research_start":
                features = get_app_features()
                if not companion_bench_allowed(sess):
                    await _safe_send_json(
                        ws,
                        {"type": "error", "error": "Companion research is disabled in this app profile."},
                    )
                    continue
                from .companion_bench import DEFAULT_COMPANION_MAX_TICKS_PER_TASK

                knowledge_source = str(msg.get("knowledge_source") or "base").strip().lower()
                max_ticks_raw = msg.get("max_ticks_per_task", DEFAULT_COMPANION_MAX_TICKS_PER_TASK)
                try:
                    max_ticks_per_task = max(1, int(max_ticks_raw))
                except (TypeError, ValueError):
                    max_ticks_per_task = DEFAULT_COMPANION_MAX_TICKS_PER_TASK
                active_agent_model = str(msg.get("active_agent_model") or sess.active_agent_model).strip()
                active_agent_mode = str(msg.get("active_agent_mode") or sess.active_agent_mode).strip().lower()
                megaprompt_config_name = str(msg.get("megaprompt_config_name") or "").strip()
                arc_prompt_extra = msg.get("arc_prompt_extra")
                exo_planet_enabled = msg.get("exo_planet_enabled")
                game_kind = msg.get("game_kind")
                arc_game_id = msg.get("arc_game_id")
                player_name = msg.get("player_name")
                if player_name is None:
                    player_name = msg.get("player_nickname")
                player_avatar_id = msg.get("player_avatar_id")
                if sess.agent_tick_is_running():
                    sess.stop_agent_requested = True
                    await _safe_send_json(
                        ws,
                        {
                            "type": "agent_stop_ack",
                            "message": (
                                "Previous agent request is still finishing after stop. "
                                "Try again in a moment."
                            ),
                            "running": True,
                            "campaign_state": sess.get_campaign_snapshot(),
                        },
                    )
                    continue
                try:
                    runtime_megaprompt = megaprompt_config_name or None
                    runtime_arc_prompt_extra = str(arc_prompt_extra) if arc_prompt_extra is not None else None
                    runtime_model = active_agent_model or None
                    runtime_mode = active_agent_mode or None
                    if not features.observation_format_selection:
                        runtime_megaprompt = None
                    if not features.arc_prompt_override:
                        runtime_arc_prompt_extra = None
                    if not features.model_selection:
                        runtime_model = None
                        runtime_mode = None
                    sess.sync_client_runtime_config(
                        exo_planet_enabled=(
                            bool(exo_planet_enabled) if exo_planet_enabled is not None else None
                        ),
                        game_kind=str(game_kind) if game_kind is not None else None,
                        arc_game_id=str(arc_game_id) if arc_game_id is not None else None,
                        megaprompt_config_name=runtime_megaprompt,
                        arc_prompt_extra=runtime_arc_prompt_extra,
                        active_agent_model=runtime_model,
                        active_agent_mode=runtime_mode,
                        player_name=str(player_name).strip() if player_name is not None else None,
                        player_avatar_id=(
                            int(player_avatar_id)
                            if player_avatar_id is not None
                            else None
                        ),
                    )
                    apply_demo_runtime_defaults(sess)
                except ValueError as e:
                    await _safe_send_json(
                        ws,
                        {
                            "type": "error",
                            "error": str(e) or "Invalid runtime configuration for companion research",
                        },
                    )
                    continue
                sess.stop_agent_requested = False
                _spawn_agent_tick_task(
                    sess,
                    messaging_service.run_companion_research(
                        ws,
                        sess,
                        knowledge_source=knowledge_source,
                        max_ticks_per_task=max_ticks_per_task,
                        model=active_agent_model,
                        mode=active_agent_mode,
                    ),
                )
                continue

            if mtype == "agent_tick":
                features = get_app_features()
                raw_steps = msg.get("steps", 1)
                try:
                    steps = (
                        int(raw_steps)
                        if "steps" in msg
                        else int(sess.default_agent_steps)
                    )
                except (TypeError, ValueError):
                    steps = int(sess.default_agent_steps)
                steps = sess.clamp_agent_steps(steps)
                goal = str(msg.get("goal", DEFAULT_AGENT_GOAL))
                active_agent_model = str(msg.get("active_agent_model", "")).strip()
                active_agent_mode = str(msg.get("active_agent_mode", "")).strip().lower()
                megaprompt_config_name = str(msg.get("megaprompt_config_name", "")).strip()
                arc_prompt_extra = msg.get("arc_prompt_extra")
                exo_planet_enabled = msg.get("exo_planet_enabled")
                game_kind = msg.get("game_kind")
                arc_game_id = msg.get("arc_game_id")

                if sess.agent_tick_is_running():
                    sess.stop_agent_requested = True
                    await _safe_send_json(
                        ws,
                        {
                            "type": "agent_stop_ack",
                            "message": (
                                "Previous agent request is still finishing after stop. "
                                "Try again in a moment."
                            ),
                            "running": True,
                            "campaign_state": sess.get_campaign_snapshot(),
                        },
                    )
                    continue
                if sess.pending_human_tick is not None:
                    await _safe_send_json(
                        ws,
                        {
                            "type": "error",
                            "error": (
                                "Agent is waiting for an operator answer. "
                                "Press Stop to cancel the current run."
                            ),
                        },
                    )
                    continue
                try:
                    runtime_megaprompt = megaprompt_config_name or None
                    runtime_arc_prompt_extra = str(arc_prompt_extra) if arc_prompt_extra is not None else None
                    runtime_model = active_agent_model or None
                    runtime_mode = active_agent_mode or None
                    if not features.observation_format_selection:
                        runtime_megaprompt = None
                    if not features.arc_prompt_override:
                        runtime_arc_prompt_extra = None
                    if not features.model_selection:
                        runtime_model = None
                        runtime_mode = None
                    sess.sync_client_runtime_config(
                        exo_planet_enabled=(
                            bool(exo_planet_enabled) if exo_planet_enabled is not None else None
                        ),
                        game_kind=str(game_kind) if game_kind is not None else None,
                        arc_game_id=str(arc_game_id) if arc_game_id is not None else None,
                        megaprompt_config_name=runtime_megaprompt,
                        arc_prompt_extra=runtime_arc_prompt_extra,
                        active_agent_model=runtime_model,
                        active_agent_mode=runtime_mode,
                    )
                    apply_demo_runtime_defaults(sess)
                except ValueError as e:
                    await _safe_send_json(
                        ws,
                        {
                            "type": "error",
                            "error": str(e) or "Invalid runtime configuration provided for agent_tick",
                        },
                    )
                    continue
                sess.stop_agent_requested = False
                _spawn_agent_tick_task(
                    sess,
                    messaging_service.run_agent_tick(ws, sess, steps, goal),
                )
                continue

            await _safe_send_json(
                ws, {"type": "error", "error": "unknown message type"}
            )
    except WebSocketDisconnect:
        sess.stop_agent_requested = True
        if sess.agent_tick_is_running():
            await sess.cancel_agent_tick_task()
        return
    except RuntimeError as e:
        if "not connected" in str(e).lower() or "accept" in str(e).lower():
            return
        raise


class _RevalidatedStaticFiles(StaticFiles):
    """Static files with Cache-Control: no-cache.

    Browsers otherwise apply heuristic caching (there is no Cache-Control
    header by default) and may keep serving stale js/css for hours after a
    deploy. `no-cache` still allows 304 revalidation via ETag, so unchanged
    files are not re-downloaded.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


if CLIENT_DIR.is_dir():
    app.mount("/", _RevalidatedStaticFiles(directory=str(CLIENT_DIR), html=True), name="static")
