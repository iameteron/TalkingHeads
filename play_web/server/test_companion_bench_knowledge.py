from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SERVER_DIR = Path(__file__).resolve().parent
_server_pkg = types.ModuleType("server")
_server_pkg.__path__ = [str(_SERVER_DIR)]
sys.modules.setdefault("server", _server_pkg)

for name, fname in [
    ("server.model_names", "model_names.py"),
    ("server.campaign_mode", "campaign_mode.py"),
    ("server.deployment_operator_limits", "deployment_operator_limits.py"),
    ("server.leaderboard", "leaderboard.py"),
]:
    spec = importlib.util.spec_from_file_location(
        name, _SERVER_DIR / fname, submodule_search_locations=[str(_SERVER_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

_runtime = types.ModuleType("server.runtime")
_runtime.create_isolated_session = MagicMock()
sys.modules["server.runtime"] = _runtime

spec = importlib.util.spec_from_file_location(
    "server.companion_bench",
    _SERVER_DIR / "companion_bench.py",
    submodule_search_locations=[str(_SERVER_DIR)],
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "server"
mod.__name__ = "server.companion_bench"
sys.modules["server.companion_bench"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

from oracle.knowledge import (
    load_durable_knowledge_records,
    read_starter_revision,
    save_knowledge_records,
    use_knowledge_paths,
    write_starter_revision,
)


class CompanionBenchKnowledgeTests(unittest.TestCase):
    def test_model_knowledge_paths_include_world_slug(self) -> None:
        craftax = mod.model_knowledge_paths("gpt-test", "craftax")
        exo = mod.model_knowledge_paths("gpt-test", "exo-planet")
        self.assertIn("__craftax", craftax[0].name)
        self.assertIn("__exo_planet", exo[0].name)
        self.assertNotEqual(craftax[0], exo[0])

    def test_base_knowledge_paths_by_world(self) -> None:
        craftax_json, _ = mod._base_knowledge_paths("craftax")
        exo_json, _ = mod._base_knowledge_paths("exo-planet")
        self.assertIn("craftext_prompt", str(craftax_json))
        self.assertIn("exo-planet_prompt", str(exo_json))

    def test_prepare_test_agent_knowledge_uses_world_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            craftax_root = root / "craftext"
            exo_root = root / "exo"
            craftax_root.mkdir()
            exo_root.mkdir()
            craftax_json = craftax_root / "knowledge_data.json"
            craftax_txt = craftax_root / "knowledge_data.txt"
            exo_json = exo_root / "knowledge_data.json"
            exo_txt = exo_root / "knowledge_data.txt"
            agent_json = root / "agent.json"
            agent_txt = root / "agent.txt"

            with use_knowledge_paths(json_path=craftax_json, txt_path=craftax_txt):
                save_knowledge_records(
                    [{"id": 1, "type": "MECHANICS", "skill": "craftax_only", "rules": "from craftax"}]
                )
            with use_knowledge_paths(json_path=exo_json, txt_path=exo_txt):
                save_knowledge_records(
                    [{"id": 1, "type": "MECHANICS", "skill": "exo_only", "rules": "from exo"}]
                )

            with patch.object(
                mod,
                "_base_knowledge_paths",
                side_effect=lambda world_mode: (
                    (exo_json, exo_txt)
                    if str(world_mode).startswith("exo")
                    else (craftax_json, craftax_txt)
                ),
            ):
                mod._prepare_test_agent_knowledge(
                    knowledge_source="base",
                    world_mode="exo-planet",
                    model="gpt-test",
                    agent_json=agent_json,
                    agent_txt=agent_txt,
                )

            with use_knowledge_paths(json_path=agent_json, txt_path=agent_txt):
                durable = load_durable_knowledge_records()
            self.assertEqual(len(durable), 1)
            self.assertEqual(durable[0]["skill"], "exo_only")

    def test_prepare_test_agent_knowledge_excludes_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_json = root / "model.json"
            model_txt = root / "model.txt"
            agent_json = root / "agent.json"
            agent_txt = root / "agent.txt"

            with use_knowledge_paths(json_path=model_json, txt_path=model_txt):
                save_knowledge_records(
                    [
                        {"id": 1, "type": "MECHANICS", "skill": "keep", "rules": "durable"},
                        {"id": 2, "type": "NOTE", "skill": "drop", "rules": "ephemeral"},
                    ]
                )

            with patch.object(
                mod,
                "resolve_model_knowledge_paths",
                return_value=(model_json, model_txt),
            ):
                mod._prepare_test_agent_knowledge(
                    knowledge_source="own",
                    world_mode="craftax",
                    model="gpt-test",
                    agent_json=agent_json,
                    agent_txt=agent_txt,
                )

            with use_knowledge_paths(json_path=agent_json, txt_path=agent_txt):
                durable = load_durable_knowledge_records()
            self.assertEqual(len(durable), 1)
            self.assertEqual(durable[0]["skill"], "keep")

    def test_resolve_model_knowledge_paths_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp)
            legacy_json = bench / "knowlage_data_legacy_model.json"
            legacy_txt = bench / "knowlage_data_legacy_model.txt"
            legacy_json.write_text("[]", encoding="utf-8")
            legacy_txt.write_text("", encoding="utf-8")

            with patch.object(mod, "_bench_dir", return_value=bench):
                with patch.object(
                    mod,
                    "_legacy_model_knowledge_paths",
                    return_value=(legacy_json, legacy_txt),
                ):
                    resolved = mod.resolve_model_knowledge_paths("legacy-model", "craftax")
            self.assertEqual(resolved[0], legacy_json)

    def test_ensure_model_knowledge_current_refreshes_stale_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            craftax_root = root / "craftext"
            exo_root = root / "exo"
            bench = craftax_root / "companion_bench"
            craftax_root.mkdir()
            exo_root.mkdir()
            bench.mkdir()
            base_json = craftax_root / "knowledge_data.json"
            base_txt = craftax_root / "knowledge_data.txt"
            model_json = bench / "knowlage_data_test-model__craftax.json"
            model_txt = bench / "knowlage_data_test-model__craftax.txt"

            with use_knowledge_paths(json_path=base_json, txt_path=base_txt):
                save_knowledge_records(
                    [
                        {
                            "id": 1,
                            "type": "ACTION",
                            "skill": "PLACE_CRAFT_BOARD",
                            "rules": "starter",
                        }
                    ]
                )
                write_starter_revision(base_json, 4)

            with use_knowledge_paths(json_path=model_json, txt_path=model_txt):
                save_knowledge_records(
                    [
                        {
                            "id": 1,
                            "type": "ACTION",
                            "skill": "old_row",
                            "rules": "stale",
                        }
                    ]
                )

            with patch.object(mod, "_KNOWLEDGE_ROOT", craftax_root), patch.object(
                mod, "_EXO_KNOWLEDGE_ROOT", exo_root
            ):
                mod.ensure_model_knowledge_current("test-model", "craftax")

            with use_knowledge_paths(json_path=model_json, txt_path=model_txt):
                durable = load_durable_knowledge_records()
            self.assertEqual(durable[0]["skill"], "PLACE_CRAFT_BOARD")
            self.assertEqual(read_starter_revision(model_json), 4)

    def test_knowledge_file_info_for_model_world_mode(self) -> None:
        info = mod.knowledge_file_info_for_model("my-model", "exo-planet")
        self.assertEqual(info["world_mode"], "exo_planet")
        self.assertIn("__exo_planet", info["model_knowledge_file"])


if __name__ == "__main__":
    unittest.main()
