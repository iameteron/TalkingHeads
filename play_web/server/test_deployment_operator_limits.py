import importlib.util
import sys
import types
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_server_pkg = types.ModuleType("server")
_server_pkg.__path__ = [str(_SERVER_DIR)]
sys.modules.setdefault("server", _server_pkg)

for name, fname in [
    ("server.model_names", "model_names.py"),
    ("server.campaign_mode", "campaign_mode.py"),
]:
    spec = importlib.util.spec_from_file_location(
        name, _SERVER_DIR / fname, submodule_search_locations=[str(_SERVER_DIR)]
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    assert spec.loader is not None
    spec.loader.exec_module(m)

spec = importlib.util.spec_from_file_location(
    "server.deployment_operator_limits",
    _SERVER_DIR / "deployment_operator_limits.py",
    submodule_search_locations=[str(_SERVER_DIR)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["server.deployment_operator_limits"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_operator_call_limits_by_task_tier():
    assert mod.operator_call_limit_for_task("collect_wood") == 1
    assert mod.operator_call_limit_for_task("make_wood_pickaxe") == 1
    assert mod.operator_call_limit_for_task("collect_stone") == 2
    assert mod.operator_call_limit_for_task("make_stone_pickaxe") == 2
    assert mod.operator_call_limit_for_task("collect_coal") == 2
    assert mod.operator_call_limit_for_task("collect_iron") == 2
    assert mod.operator_call_limit_for_task("make_furnace") == 4
    assert mod.operator_call_limit_for_task("make_iron_pickaxe") == 4
    assert mod.operator_call_limit_for_task("collect_diamond") == 4


def test_operator_call_limit_violated_only_after_limit():
    assert not mod.operator_call_limit_violated(questions_count=1, limit=1)
    assert not mod.operator_call_limit_violated(questions_count=2, limit=2)
    assert mod.operator_call_limit_violated(questions_count=2, limit=1)
    assert mod.operator_call_limit_violated(questions_count=5, limit=4)


def test_deployment_megaprompt_name_by_world_mode():
    assert mod.deployment_megaprompt_config_name("craftax") == "database_formulation_deployment"
    assert mod.deployment_megaprompt_config_name("exo-planet") == "exo_database_formulation_deployment"


def test_format_operator_call_budget_text():
    text = mod.format_operator_call_budget_text(used=2, limit=3)
    assert "2" in text and "3" in text
    assert "penalized" in text.lower()


if __name__ == "__main__":
    test_operator_call_limits_by_task_tier()
    test_operator_call_limit_violated_only_after_limit()
    test_deployment_megaprompt_name_by_world_mode()
    test_format_operator_call_budget_text()
    print("Deployment operator limits tests passed.")
