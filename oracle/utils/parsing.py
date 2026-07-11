import ast
import re
from typing import Dict


def parse_question_expert_response(text: str) -> Dict[str, str]:
    result = {
        "map_expert_question": "",
        "mechanics_expert_question": "",
        "action_expert_question": "",
    }
    if not text or not text.strip():
        return result

    for marker in ("```python", "```"):
        if marker in text:
            try:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                block = text[start:end].strip()
                parsed = ast.literal_eval(block)
                if isinstance(parsed, dict):
                    result["map_expert_question"] = str(
                        parsed.get("map_expert_question", "")
                    ).strip()
                    result["mechanics_expert_question"] = str(
                        parsed.get("mechanics_expert_question", "")
                    ).strip()
                    result["action_expert_question"] = str(
                        parsed.get("action_expert_question", "")
                    ).strip()
                    return result
            except (ValueError, SyntaxError):
                pass

    for key in (
        "map_expert_question",
        "mechanics_expert_question",
        "action_expert_question",
    ):
        m = re.search(
            rf'["\']?{key}["\']?\s*:\s*["\']([^"\']*)["\']', text, re.IGNORECASE
        )
        if m:
            result[key] = m.group(1).strip()
    return result


def default_expert_questions_for_goal(goal: str) -> Dict[str, str]:
    """Fallback sub-questions when the question expert output is empty or unparseable."""
    topic = str(goal or "the current objective").strip() or "the current objective"
    return {
        "map_expert_question": f"On the map, where are objects or terrain relevant to: {topic}?",
        "mechanics_expert_question": f"What achievements and prerequisites apply to: {topic}?",
        "action_expert_question": f"What is the next concrete action toward: {topic}?",
    }


def ensure_expert_questions(questions: Dict[str, str], goal: str) -> Dict[str, str]:
    """Fill missing sub-questions with deterministic fallbacks."""
    defaults = default_expert_questions_for_goal(goal)
    merged = {key: str(questions.get(key, "") or "").strip() for key in defaults}
    if any(merged.values()):
        for key, fallback in defaults.items():
            if not merged[key]:
                merged[key] = fallback
        return merged
    return defaults
