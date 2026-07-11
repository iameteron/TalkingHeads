import ast
import importlib
import logging
import re
from inspect import signature
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .observation_formatting import render_textual_observation_with_path_from_env_state
from .pathfinding import find_path_on_craftax_map

logger = logging.getLogger(__name__)

# Operator length repair: repeat same-dialog + prompt-resend cycles until within budget or max rounds.
MAX_LENGTH_REPAIR_ROUNDS = 5


class PathExpertPipeline:
    """Encapsulates all path expert specific preprocessing and rendering."""

    def __init__(self, helper_expert, path_expert, operator_max_answer_chars: Optional[int] = None):
        self.helper_expert = helper_expert
        self.path_expert = path_expert
        self.operator_max_answer_chars = operator_max_answer_chars

    @staticmethod
    def _stringify_position(value) -> str:
        if value is None:
            return "not provided"
        if isinstance(value, (list, tuple, np.ndarray)):
            return str(np.array(value).tolist())
        return str(value)

    def _extract_agent_position(self, env_state) -> str:
        if env_state is not None and hasattr(env_state, "player_position"):
            return self._stringify_position(env_state.player_position)
        return "not provided"

    def _extract_target_location(self, env_state) -> str:
        if env_state is None:
            return "not provided"
        for attr_name in ("target_location", "goal_location", "target_position", "goal_position"):
            if hasattr(env_state, attr_name):
                return self._stringify_position(getattr(env_state, attr_name))
        return "not provided"

    def _extract_target_location_from_text(self, text: str) -> str:
        if not text:
            return "not provided"
        bracket_match = re.search(r"(\[\s*-?\d+\s*,\s*-?\d+\s*\]|\(\s*-?\d+\s*,\s*-?\d+\s*\))", text)
        if bracket_match:
            pos = self._position_from_string(bracket_match.group(1))
            if pos is not None:
                return str(pos)

        xy_match = re.search(
            r"(?:x|row)\s*[:=]?\s*(-?\d+)\D+(?:y|col|column)\s*[:=]?\s*(-?\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if xy_match:
            return str([int(xy_match.group(1)), int(xy_match.group(2))])
        return "not provided"

    def _extract_target_location_from_helper_answer(self, helper_answer: str) -> str:
        if not helper_answer:
            return "not provided"

        goal_match = re.search(
            r"goal\s*=\s*(\[[^\]]+\]|\([^)]+\)|agent_position|target_location)",
            helper_answer,
            flags=re.IGNORECASE,
        )
        if not goal_match:
            return "not provided"

        goal_raw = goal_match.group(1).strip()
        goal_lower = goal_raw.lower()
        if goal_lower in {"agent_position", "target_location"}:
            return "not provided"

        pos = self._position_from_string(goal_raw)
        if pos is None:
            return "not provided"
        return str(pos)

    def _infer_goal_via_path_helper(self, question: str, agent_position: str) -> Optional[List[int]]:
        """
        When CrafText / caller did not supply goal coordinates, ask the path
        helper model to infer goal=[row, col] from the agent's question alone.
        """
        raw = self.helper_expert.chat_with_retry(
            question,
            agent_position=agent_position,
            target_location="not provided",
            infer_goal_from_question=True,
        )
        if not raw:
            return None
        if "TARGET_LOCATION_NOT_PROVIDED" in raw:
            return None
        via_goal = self._extract_target_location_from_helper_answer(raw)
        if via_goal != "not provided":
            return self._position_from_string(via_goal)
        via_text = self._extract_target_location_from_text(raw)
        if via_text != "not provided":
            return self._position_from_string(via_text)
        return None

    @staticmethod
    def _position_from_string(value: str):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
            return [int(value[0]), int(value[1])]
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return [int(parsed[0]), int(parsed[1])]
        except Exception:
            pass
        nums = re.findall(r"-?\d+", str(value))
        if len(nums) >= 2:
            return [int(nums[0]), int(nums[1])]
        return None

    @staticmethod
    def _tile_name_from_id(block_id: int) -> str:
        try:
            constants = importlib.import_module("craftax.craftax_classic.constants")
            block_type = constants.BlockType
            return block_type(block_id).name.title().replace("_", " ")
        except Exception:
            return f"Block_{int(block_id)}"

    @staticmethod
    def find_nearest_most_colorful_path_point(
        state,
        predicted_path: Sequence[Sequence[int]],
        near_radius: int = 2,
    ) -> Dict[str, object]:
        """
        Find the nearest path point that has the highest diversity of nearby blocks.

        Colorfulness is defined as the number of distinct block names found in the
        square neighborhood of radius `near_radius` around a path point.
        """
        empty_result: Dict[str, object] = {
            "point": None,
            "point_index": -1,
            "colorfulness": 0,
            "secondary_colorfulness": 0,
            "used_secondary_fallback": False,
            "blocks": [],
        }
        if (
            state is None
            or not hasattr(state, "map")
            or predicted_path is None
            or len(predicted_path) == 0
        ):
            return empty_result

        grid = state.map
        h, w = grid.shape
        agent_xy: Optional[Tuple[int, int]] = None
        if hasattr(state, "player_position"):
            ap = np.array(state.player_position).tolist()
            if isinstance(ap, list) and len(ap) >= 2:
                agent_xy = (int(ap[0]), int(ap[1]))

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < h and 0 <= y < w

        point_stats: List[Dict[str, object]] = []

        for i, raw_point in enumerate(predicted_path):
            if raw_point is None or len(raw_point) != 2:
                continue
            x, y = int(raw_point[0]), int(raw_point[1])
            if not in_bounds(x, y):
                continue
            if agent_xy is not None and (x, y) == agent_xy:
                # Do not use the current agent tile as a landmark waypoint.
                continue

            by_name: Dict[str, Dict[str, object]] = {}
            non_grass_counts: Dict[str, int] = {}

            # Only consider direct tiles from the waypoint:
            # up/down/left/right up to `near_radius` steps (no diagonals).
            ray_cells: List[Tuple[int, int, int]] = []
            for step in range(1, near_radius + 1):
                candidates = [
                    (x - step, y, step),  # up
                    (x + step, y, step),  # down
                    (x, y - step, step),  # left
                    (x, y + step, step),  # right
                ]
                for nx, ny, dist in candidates:
                    if in_bounds(nx, ny):
                        ray_cells.append((nx, ny, dist))

            for nx, ny, dist in ray_cells:
                block_id = int(grid[nx, ny])
                block_name = PathExpertPipeline._tile_name_from_id(block_id)
                if block_name not in by_name:
                    by_name[block_name] = {"name": block_name, "coord": [nx, ny], "distance": dist}
                elif dist < int(by_name[block_name]["distance"]):
                    by_name[block_name] = {"name": block_name, "coord": [nx, ny], "distance": dist}
                if block_name.strip().lower() != "grass":
                    non_grass_counts[block_name] = non_grass_counts.get(block_name, 0) + 1

            colorfulness = len(non_grass_counts)
            secondary_colorfulness = sum(non_grass_counts.values())
            blocks = sorted(
                [by_name[name] for name in non_grass_counts.keys()],
                key=lambda b: (int(b.get("distance", 9999)), str(b["name"])),
            )
            point_stats.append(
                {
                    "point": [x, y],
                    "point_index": i,
                    "colorfulness": int(colorfulness),
                    "secondary_colorfulness": int(secondary_colorfulness),
                    "blocks": blocks,
                }
            )

        if not point_stats:
            return empty_result

        max_primary = max(int(s["colorfulness"]) for s in point_stats)
        use_secondary_fallback = max_primary <= 1

        if use_secondary_fallback:
            ranked_stats = sorted(
                point_stats,
                key=lambda s: (
                    int(s["secondary_colorfulness"]),
                    int(s["colorfulness"]),
                    -int(s["point_index"]),
                ),
                reverse=True,
            )
        else:
            ranked_stats = sorted(
                point_stats,
                key=lambda s: (
                    int(s["colorfulness"]),
                    int(s["secondary_colorfulness"]),
                    -int(s["point_index"]),
                ),
                reverse=True,
            )

        top_candidates = ranked_stats[:3]
        if agent_xy is None:
            best = top_candidates[0]
        else:
            best = min(
                top_candidates,
                key=lambda s: (
                    abs(int(s["point"][0]) - agent_xy[0]) + abs(int(s["point"][1]) - agent_xy[1]),
                    int(s["point_index"]),
                ),
            )

        return {
            "point": best["point"],
            "point_index": best["point_index"],
            "colorfulness": best["colorfulness"],
            "secondary_colorfulness": best["secondary_colorfulness"],
            "used_secondary_fallback": use_secondary_fallback,
            "blocks": best["blocks"],
        }

    @staticmethod
    def _format_colorful_waypoint_section(
        state,
        predicted_path: Sequence[Sequence[int]],
        near_radius: int = 2,
    ) -> str:
        """
        Build prompt text for the next landmark waypoint (coords for expert only;
        answer must not leak coordinates per path expert guidelines).
        """
        if not predicted_path:
            return ""
        result = PathExpertPipeline.find_nearest_most_colorful_path_point(
            state=state,
            predicted_path=predicted_path,
            near_radius=near_radius,
        )
        if result.get("point") is None:
            return ""

        point = result["point"]
        idx = int(result["point_index"])
        path_len = len(predicted_path)
        colorfulness = int(result["colorfulness"])
        secondary = int(result["secondary_colorfulness"])
        used_fallback = bool(result["used_secondary_fallback"])
        mode_line = (
            "fallback by total non-grass count (same object types count)"
            if used_fallback
            else "primary colorfulness (distinct non-grass types)"
        )

        blocks = result.get("blocks") or []
        block_names = [str(b.get("name", "")) for b in blocks if isinstance(b, dict)]
        names_line = ", ".join(block_names) if block_names else "(none — path only through grass in neighborhood)"

        def _steps_word(n: int) -> str:
            words = {
                0: "zero",
                1: "one",
                2: "two",
                3: "three",
                4: "four",
                5: "five",
                6: "six",
                7: "seven",
                8: "eight",
                9: "nine",
                10: "ten",
            }
            return words.get(n, str(n))

        def _direction_phrase(dx: int, dy: int, uppercase: bool = True) -> str:
            parts: List[str] = []
            if dx < 0:
                parts.append("TOP" if uppercase else "top")
            elif dx > 0:
                parts.append("DOWN" if uppercase else "down")
            if dy < 0:
                parts.append("LEFT" if uppercase else "left")
            elif dy > 0:
                parts.append("RIGHT" if uppercase else "right")
            return " and ".join(parts) if parts else ("ON" if uppercase else "on")

        def _relative_phrase(from_xy: Sequence[int], to_xy: Sequence[int], reference_block_name: str) -> str:
            dx = int(to_xy[0]) - int(from_xy[0])
            dy = int(to_xy[1]) - int(from_xy[1])
            steps = abs(dx) + abs(dy)
            if steps == 0:
                return f"on the {reference_block_name} block"
            if dx > 0 and dy == 0:
                direction = "below"
            elif dx < 0 and dy == 0:
                direction = "above"
            elif dx == 0 and dy > 0:
                direction = "to the right"
            elif dx == 0 and dy < 0:
                direction = "to the left"
            else:
                vertical = "down" if dx > 0 else "up"
                horizontal = "right" if dy > 0 else "left"
                direction = f"{vertical}-{horizontal} of"
            step_word = _steps_word(steps)
            step_noun = "step" if steps == 1 else "steps"
            return f"{step_word} {step_noun} {direction} the {reference_block_name} block"

        waypoint_x, waypoint_y = int(point[0]), int(point[1])
        waypoint_block_id = int(state.map[waypoint_x, waypoint_y]) if hasattr(state, "map") else -1
        waypoint_block_name = PathExpertPipeline._tile_name_from_id(waypoint_block_id)

        waypoint_from_agent = "at the agent"
        if state is not None and hasattr(state, "player_position"):
            ap = np.array(state.player_position).tolist()
            if len(ap) >= 2:
                axy = [int(ap[0]), int(ap[1])]
                adx = waypoint_x - axy[0]
                ady = waypoint_y - axy[1]
                if adx == 0 and ady == 0:
                    waypoint_from_agent = "at the agent"
                elif adx != 0 and ady != 0:
                    vertical = "down" if adx > 0 else "up"
                    horizontal = "right" if ady > 0 else "left"
                    waypoint_from_agent = f"{vertical}-{horizontal} of the agent"
                elif adx > 0:
                    waypoint_from_agent = "below the agent"
                elif adx < 0:
                    waypoint_from_agent = "above the agent"
                elif ady > 0:
                    waypoint_from_agent = "to the right of the agent"
                else:
                    waypoint_from_agent = "to the left of the agent"

        block_relation_lines: List[str] = []
        for b in blocks:
            b_name = str(b.get("name", "Object"))
            b_coord = b.get("coord", None)
            if not isinstance(b_coord, list) or len(b_coord) != 2:
                continue
            relation = _relative_phrase(
                [waypoint_x, waypoint_y],
                b_coord,
                waypoint_block_name.upper(),
            )
            block_relation_lines.append(f"The {b_name} block is {relation}.")

        _ = (path_len, idx, colorfulness, secondary, mode_line, names_line)  # keep computed debug values stable
        base_sentence = (
            f"The nearest keypoint on the agent's path is a {waypoint_block_name.upper()} block. "
            f"It is located {waypoint_from_agent}."
        )
        if block_relation_lines:
            return " ".join([base_sentence, *block_relation_lines])
        return base_sentence

    def _render_path_observation(
        self,
        helper_answer: str,
        env_state,
        agent_position: str,
        target_location: str,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        if env_state is None or not hasattr(env_state, "map"):
            return ("Environment map is unavailable, path rendering is unavailable.", [])
        if "TARGET_LOCATION_NOT_PROVIDED" in helper_answer:
            return ("Target location is not provided, path rendering is unavailable.", [])

        start_pos = self._position_from_string(agent_position)
        target_pos = self._position_from_string(target_location)
        if start_pos is None:
            logger.warning(
                "Path rendering skipped: agent position unavailable. agent_position=%r target_location=%r",
                agent_position,
                target_location,
            )
            return ("Agent position is unavailable, path rendering is unavailable.", [])
        if target_pos is None:
            logger.warning(
                "Path rendering skipped: target position unavailable. agent_position=%r target_location=%r",
                agent_position,
                target_location,
            )
            return ("Target position is unavailable, path rendering is unavailable.", [])

        center_pos = start_pos
        normalized = helper_answer.replace(" ", "").lower()
        if "center_coord=target_location" in normalized or "center=target_location" in normalized:
            center_pos = target_pos
        elif "center_coord=agent_position" in normalized or "center=agent_position" in normalized:
            center_pos = start_pos
        else:
            center_match = re.search(
                r"(?:center_coord|center)\s*=\s*(\[[^\]]+\]|\([^)]+\)|agent_position|target_location)",
                helper_answer,
                flags=re.IGNORECASE,
            )
            if center_match:
                center_raw = center_match.group(1).strip()
                center_raw_lower = center_raw.lower()
                if center_raw_lower == "target_location":
                    center_pos = target_pos
                elif center_raw_lower == "agent_position":
                    center_pos = start_pos
                else:
                    parsed_center = self._position_from_string(center_raw)
                    if parsed_center is not None:
                        center_pos = parsed_center

        predicted_path = find_path_on_craftax_map(
            state=env_state,
            start=start_pos,
            goal=target_pos,
        )
        try:
            rendered = render_textual_observation_with_path_from_env_state(
                state=env_state,
                start=start_pos,
                goal=target_pos,
                center_coord=center_pos,
                radius=5,
                path=predicted_path,
            )
            return (rendered, predicted_path)
        except Exception as e:
            logger.exception("Failed to render path observation")
            return (f"Path rendering failed: {e}", predicted_path)

    @staticmethod
    def _contains_forbidden_path_visual_references(answer: str) -> bool:
        if not answer:
            return False
        answer_lower = answer.lower()
        forbidden_phrases = (
            "arrow",
            "arrows",
            "path on the map",
            "path on map",
            "marked path",
            "follow the path",
        )
        if any(phrase in answer_lower for phrase in forbidden_phrases):
            return True
        return bool(re.search(r"[↑↓←→]", answer))

    def _agent_chat_accepts_history(self, agent) -> bool:
        try:
            return "history" in signature(agent.chat).parameters
        except Exception:
            return False

    def _repair_path_expert_answer(
        self,
        *,
        question: str,
        initial_answer: str,
        rendered_map: str,
        agent_position: str,
        target_location: str,
        colorful_waypoint: str,
    ) -> str:
        if not self._contains_forbidden_path_visual_references(initial_answer):
            return initial_answer

        correction_message = (
            "The agent cannot see arrows or the path itself. "
            "Describe the route again without mentioning arrows, symbols, or map path marks."
        )
        analysis_note = (
            'Analysis of previous answer: "The agent cannot see arrows or the path itself. '
            'Describe the route again without mentioning arrows, symbols, or map path marks."'
        )

        prompt = self.path_expert.build_prompt(
            question,
            rendered_map=rendered_map,
            agent_position=agent_position,
            target_location=target_location,
            colorful_waypoint=colorful_waypoint,
        )

        # Case 1: same dialog repair with history, if the backend supports it.
        try:
            if self._agent_chat_accepts_history(self.path_expert.agent):
                repaired_same_dialog = self.path_expert.agent.chat(
                    correction_message,
                    gen=self.path_expert.gen_config,
                    history=[{"role": "assistant", "content": initial_answer}],
                )
                if not self._contains_forbidden_path_visual_references(repaired_same_dialog):
                    logger.info("Path expert answer repaired in same dialog.")
                    return repaired_same_dialog
        except Exception:
            logger.exception("Same-dialog repair attempt failed for path expert answer.")

        # Case 2: resend previous prompt and include analysis of the previous answer.
        try:
            insertion_anchor = "Your answer:"
            if insertion_anchor in prompt:
                repaired_prompt = prompt.replace(
                    insertion_anchor,
                    f"{analysis_note}\n\n{insertion_anchor}",
                    1,
                )
            else:
                repaired_prompt = f"{prompt}\n\n{analysis_note}"
            repaired_answer = self.path_expert.agent.chat(
                repaired_prompt,
                gen=self.path_expert.gen_config,
            )
            if not self._contains_forbidden_path_visual_references(repaired_answer):
                logger.info("Path expert answer repaired via prompt resend with analysis note.")
                return repaired_answer
        except Exception:
            logger.exception("Prompt-resend repair attempt failed for path expert answer.")

        logger.warning("Path expert answer still contains forbidden references after repair attempts.")
        return initial_answer

    @staticmethod
    def _answer_exceeds_length_budget(answer: str, max_chars: int) -> bool:
        return bool(answer) and len(answer) > max_chars

    @staticmethod
    def _truncate_answer_to_char_budget(answer: str, max_chars: int) -> str:
        """Deterministic last resort: agent must receive substantive text within the channel limit."""
        if max_chars <= 0:
            return ""
        if len(answer) <= max_chars:
            return answer
        return answer[:max_chars].rstrip()

    @staticmethod
    def _embed_previous_answer_for_repair(initial_answer: str, max_embed_chars: int = 12000) -> str:
        """Include full prior answer in repair prompts (truncate only if absurdly long for the API)."""
        if len(initial_answer) <= max_embed_chars:
            return initial_answer
        return (
            initial_answer[:max_embed_chars]
            + "\n\n[... truncated when attaching to repair prompt; rewrite from this fragment ...]"
        )

    def _length_repair_same_dialog_message(self, initial_answer: str, max_chars: int) -> str:
        """
        Same pattern as arrow repair: user message explains the constraint and embeds the prior reply
        so the model rewrites from it—no silent shortening.
        """
        embedded = self._embed_previous_answer_for_repair(initial_answer)
        return (
            "The previous answer is too long for the agent's observation channel "
            f"(rough limit about {max_chars} characters). "
            "Below is the full text of that answer. Rewrite the same guidance in much fewer words "
            "(same meaning; terrain and directions only; stay within the channel limit):\n\n"
            "---\n"
            f"{embedded}\n"
            "---"
        )

    def _length_repair_analysis_note(self, initial_answer: str, max_chars: int) -> str:
        """Second repair path: prepend analysis + full prior answer into the main expert prompt."""
        embedded = self._embed_previous_answer_for_repair(initial_answer)
        return (
            f"Analysis of previous answer: it exceeds the maximum length the agent can receive "
            f"(about {max_chars} characters). The agent cannot take such a long reply through this channel. "
            "Full previous answer to rewrite briefly:\n\n"
            f"---\n{embedded}\n---\n\n"
            "Provide the same guidance again in far fewer words under the limit."
        )

    def _repair_answer_if_too_long(
        self,
        *,
        question: str,
        initial_answer: str,
        rendered_map: str,
        agent_position: str,
        target_location: str,
        colorful_waypoint: str,
        max_chars: int,
    ) -> str:
        """
        Same repair pattern as arrows: embed the current candidate and ask for a shorter rewrite.
        Repeat for MAX_LENGTH_REPAIR_ROUNDS (same-dialog attempt, then prompt-resend each round).
        Only if the model still exceeds the budget after all rounds, truncate deterministically so the
        agent always receives in-channel guidance—not a meta “ask again” stub.
        """
        if not self._answer_exceeds_length_budget(initial_answer, max_chars):
            return initial_answer

        prompt = self.path_expert.build_prompt(
            question,
            rendered_map=rendered_map,
            agent_position=agent_position,
            target_location=target_location,
            colorful_waypoint=colorful_waypoint,
        )

        candidate = initial_answer
        for round_idx in range(MAX_LENGTH_REPAIR_ROUNDS):
            if not self._answer_exceeds_length_budget(candidate, max_chars):
                logger.info("Path expert answer within length budget after round %s.", round_idx)
                return candidate

            correction_message = self._length_repair_same_dialog_message(candidate, max_chars)

            try:
                if self._agent_chat_accepts_history(self.path_expert.agent):
                    repaired_same_dialog = self.path_expert.agent.chat(
                        correction_message,
                        gen=self.path_expert.gen_config,
                        history=[{"role": "assistant", "content": candidate}],
                    )
                    if repaired_same_dialog and not self._answer_exceeds_length_budget(
                        repaired_same_dialog, max_chars
                    ):
                        logger.info(
                            "Path expert answer repaired in same dialog (length repair), round %s.",
                            round_idx,
                        )
                        return repaired_same_dialog
                    if repaired_same_dialog:
                        candidate = repaired_same_dialog
            except Exception:
                logger.exception(
                    "Same-dialog length repair failed (round %s).", round_idx,
                )

            if not self._answer_exceeds_length_budget(candidate, max_chars):
                return candidate

            # Prompt-resend must embed the latest candidate (e.g. after same-dialog shortened but still over).
            analysis_note = self._length_repair_analysis_note(candidate, max_chars)

            try:
                insertion_anchor = "Your answer:"
                if insertion_anchor in prompt:
                    repaired_prompt = prompt.replace(
                        insertion_anchor,
                        f"{analysis_note}\n\n{insertion_anchor}",
                        1,
                    )
                else:
                    repaired_prompt = f"{prompt}\n\n{analysis_note}"
                repaired_answer = self.path_expert.agent.chat(
                    repaired_prompt,
                    gen=self.path_expert.gen_config,
                )
                if repaired_answer and not self._answer_exceeds_length_budget(
                    repaired_answer, max_chars
                ):
                    logger.info(
                        "Path expert answer repaired via prompt resend (length), round %s.",
                        round_idx,
                    )
                    return repaired_answer
                if repaired_answer:
                    candidate = repaired_answer
            except Exception:
                logger.exception(
                    "Prompt-resend length repair failed (round %s).", round_idx,
                )

        if self._answer_exceeds_length_budget(candidate, max_chars):
            logger.warning(
                "Path expert answer still over budget after %s rounds; truncating to %s chars.",
                MAX_LENGTH_REPAIR_ROUNDS,
                max_chars,
            )
            return self._truncate_answer_to_char_budget(candidate, max_chars)
        return candidate

    def answer(
        self,
        question: str,
        env_state,
        goal_hint: Optional[str] = None,
        operator_max_answer_chars_override: Optional[int] = None,
        target_location: Optional[Sequence[int]] = None,
    ) -> str:
        agent_position = self._extract_agent_position(env_state)
        normalized_target = self._position_from_string(
            target_location if target_location is not None else ""
        )
        if normalized_target is None and env_state is not None:
            env_tgt = self._extract_target_location(env_state)
            if env_tgt != "not provided":
                normalized_target = self._position_from_string(env_tgt)
        if normalized_target is None and goal_hint:
            hint_text = str(goal_hint).strip()
            if hint_text:
                extracted = self._extract_target_location_from_text(hint_text)
                if extracted != "not provided":
                    normalized_target = self._position_from_string(extracted)
        if normalized_target is None and question:
            extracted_q = self._extract_target_location_from_text(question)
            if extracted_q != "not provided":
                normalized_target = self._position_from_string(extracted_q)
        if normalized_target is None:
            logger.info(
                "Path expert: no target from env/caller, goal_hint, or question text; "
                "inferring goal via path_expert_helper from agent question.",
            )
            normalized_target = self._infer_goal_via_path_helper(question, agent_position)
        if normalized_target is None:
            logger.error(
                "Path expert: no valid target_location after caller hint, text parsing, "
                "and path helper inference (caller arg was %r).",
                target_location,
            )
            return (
                "Path expert error: missing navigation target_location. "
                "CrafText did not provide goal coordinates, none were found in the goal hint "
                "or question text, and the path helper could not infer a goal from the question."
            )
        target_location = str(normalized_target)

        logger.info(
            "Path pipeline inputs: agent_position=%r target_location=%r goal_hint=%r question_prefix=%r",
            agent_position,
            target_location,
            str(goal_hint or "")[:120],
            question[:200],
        )
        helper_answer = self.helper_expert.chat_with_retry(
            question,
            agent_position=agent_position,
            target_location=target_location,
        )
        logger.warning("Path helper answer: %s", helper_answer)

        rendered_map, predicted_path = self._render_path_observation(
            helper_answer=helper_answer,
            env_state=env_state,
            agent_position=agent_position,
            target_location=target_location,
        )
        colorful_waypoint = ""
        if env_state is not None and predicted_path:
            colorful_waypoint = self._format_colorful_waypoint_section(
                env_state,
                predicted_path,
                near_radius=2,
            )
        logger.info("Rendered map passed to path expert:\n%s", rendered_map)
        if colorful_waypoint:
            logger.info("Colorful waypoint section:\n%s", colorful_waypoint)
        path_expert_answer = self.path_expert.chat_with_retry(
            question,
            rendered_map=rendered_map,
            agent_position=agent_position,
            target_location=target_location,
            colorful_waypoint=colorful_waypoint,
        )
        path_expert_answer = self._repair_path_expert_answer(
            question=question,
            initial_answer=path_expert_answer,
            rendered_map=rendered_map,
            agent_position=agent_position,
            target_location=target_location,
            colorful_waypoint=colorful_waypoint,
        )
        effective_cap = (
            operator_max_answer_chars_override
            if operator_max_answer_chars_override is not None
            else self.operator_max_answer_chars
        )
        if effective_cap is not None and effective_cap > 0:
            path_expert_answer = self._repair_answer_if_too_long(
                question=question,
                initial_answer=path_expert_answer,
                rendered_map=rendered_map,
                agent_position=agent_position,
                target_location=target_location,
                colorful_waypoint=colorful_waypoint,
                max_chars=effective_cap,
            )
        logger.info("Path expert answer: %s", path_expert_answer)
        return path_expert_answer


def stringify_agent_position(env_state) -> str:
    if env_state is not None and hasattr(env_state, "player_position"):
        return PathExpertPipeline._stringify_position(env_state.player_position)
    return "not provided"
