import importlib.util
from enum import Enum
from pathlib import Path
import sys
import types


class _FakeAction(Enum):
    NOOP = 0


def _install_parser_import_stubs():
    craftax = types.ModuleType("craftax")
    craftax_classic = types.ModuleType("craftax.craftax_classic")
    constants = types.ModuleType("craftax.craftax_classic.constants")
    constants.Action = _FakeAction
    sys.modules.setdefault("craftax", craftax)
    sys.modules.setdefault("craftax.craftax_classic", craftax_classic)
    sys.modules.setdefault("craftax.craftax_classic.constants", constants)

    oracle = types.ModuleType("oracle")
    knowledge = types.ModuleType("oracle.knowledge")
    knowledge.apply_knowledge_from_response = lambda text: (text, False, "")
    utils = types.ModuleType("oracle.utils")
    observation = types.ModuleType("oracle.utils.observation_formatting")
    observation.format_observation_from_env_state = lambda *args, **kwargs: ""
    observation.render_symbolic_map_from_env_state = lambda *args, **kwargs: ""
    sys.modules.setdefault("oracle", oracle)
    sys.modules.setdefault("oracle.knowledge", knowledge)
    sys.modules.setdefault("oracle.utils", utils)
    sys.modules.setdefault("oracle.utils.observation_formatting", observation)


_install_parser_import_stubs()
_MOD_PATH = Path(__file__).resolve().with_name("active_agent_helpers.py")
_SPEC = importlib.util.spec_from_file_location("active_agent_helpers_for_test", _MOD_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)
parse_agent_answer = _MOD.parse_agent_answer


def test_parse_arc_action1_block():
    parsed = parse_agent_answer("--- Act ---\nACTION1\n--- Act ---")
    assert parsed["action"] == "ACTION1"


def test_parse_arc_action6_plain_coords_block():
    parsed = parse_agent_answer("--- Act ---\nACTION6 12 40\n--- Act ---")
    assert parsed["action"] == "ACTION6 12 40"


def test_parse_arc_action6_bracket_coords_block():
    parsed = parse_agent_answer("--- Act ---\nACTION6 [12, 40]\n--- Act ---")
    assert parsed["action"] == "ACTION6 [12, 40]"


def test_parse_arc_action_block_case_insensitive():
    parsed = parse_agent_answer("--- ACT ---\nACTION1\n--- ACT ---")
    assert parsed["action"] == "ACTION1"


def test_parse_plain_arc_action():
    parsed = parse_agent_answer("ACTION1")
    assert parsed["action"] == "ACTION1"


def test_parse_plain_arc_action_with_coords():
    parsed = parse_agent_answer("ACTION6 12 40")
    assert parsed["action"] == "ACTION6 12 40"


def test_parse_question_block_stays_question():
    parsed = parse_agent_answer("--- Q ---\nWhat changed after ACTION1?\n--- Q ---")
    assert parsed == {"question": "What changed after ACTION1?"}


if __name__ == "__main__":
    test_parse_arc_action1_block()
    test_parse_arc_action6_plain_coords_block()
    test_parse_arc_action6_bracket_coords_block()
    test_parse_arc_action_block_case_insensitive()
    test_parse_plain_arc_action()
    test_parse_plain_arc_action_with_coords()
    test_parse_question_block_stays_question()
