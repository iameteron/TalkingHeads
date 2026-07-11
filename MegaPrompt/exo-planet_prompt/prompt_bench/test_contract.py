from __future__ import annotations

import sys
import unittest
from pathlib import Path

MEGAPROMPT_ROOT = Path(__file__).resolve().parents[2]
EXO_ROOT = MEGAPROMPT_ROOT / "exo-planet_prompt"
for p in (MEGAPROMPT_ROOT, EXO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from action_bridge import to_display_action, to_engine_action
from megaprompt.renders import Renderer

try:
    from .contract import EXO_ALLOWED_ACTIONS_META, validate_response_contract
    from .exo_fixture import extract_state_from_entry, load_exo_trajectory_entry
except ImportError:
    from contract import EXO_ALLOWED_ACTIONS_META, validate_response_contract
    from exo_fixture import extract_state_from_entry, load_exo_trajectory_entry

CONFIGS = [
    EXO_ROOT / "templates/database_formulation/exo-planet.yaml",
    EXO_ROOT / "templates/reasoning_or_ask_path/exo-planet.yaml",
    EXO_ROOT / "templates/reasoning_or_ask_help/exo-planet.yaml",
    EXO_ROOT / "templates/no_dialog/exo-planet.yaml",
]


class ExoPromptContractTests(unittest.TestCase):
    def test_action_bridge(self) -> None:
        self.assertEqual(to_engine_action("PLACE_REPLICATOR"), "PLACE_TABLE")
        self.assertEqual(to_engine_action("EXTRACT"), "DO (TO GATHER SOMETHING)")
        self.assertEqual(to_engine_action("MAKE_BONE_DRILL"), "MAKE_WOOD_PICKAXE")
        self.assertEqual(to_display_action("EXTRACT", "DO"), "EXTRACT")
        self.assertEqual(to_display_action("PLACE_REPLICATOR", "PLACE_TABLE"), "PLACE_REPLICATOR")
        self.assertEqual(to_display_action("", "PLACE_TABLE"), "PLACE_REPLICATOR")
        self.assertEqual(to_display_action("", "DO"), "EXTRACT")

    def test_validate_accepts_exo_action(self) -> None:
        raw = "<reasoning>ok</reasoning>\n<action>EXTRACT</action>"
        action, _ = validate_response_contract(
            raw,
            allowed_actions=EXO_ALLOWED_ACTIONS_META,
            allow_ask_operator=True,
            allow_update_database=True,
        )
        self.assertEqual(action, "EXTRACT")

    def test_validate_rejects_legacy_term(self) -> None:
        raw = "<reasoning>need crafting table</reasoning>\n<action>LEFT</action>"
        with self.assertRaises(ValueError):
            validate_response_contract(
                raw,
                allowed_actions=EXO_ALLOWED_ACTIONS_META,
                allow_ask_operator=True,
                allow_update_database=True,
            )

    def test_validate_rejects_mixed_question(self) -> None:
        raw = (
            "<reasoning>x</reasoning>"
            "<action>LEFT</action><question>where?</question>"
        )
        with self.assertRaises(ValueError):
            validate_response_contract(
                raw,
                allowed_actions=EXO_ALLOWED_ACTIONS_META,
                allow_ask_operator=True,
                allow_update_database=True,
            )

    def test_trajectory_dialog_uses_exo_terms(self) -> None:
        entry = load_exo_trajectory_entry(-1)
        dialog = entry.get("oracle_dialog") or []
        self.assertTrue(dialog)
        blob = " ".join(str(turn.get(k, "")) for turn in dialog for k in ("question", "answer"))
        self.assertIn("PLACE_REPLICATOR", blob)
        self.assertNotIn("PLACE_TABLE", blob)
        self.assertNotIn("place a table", blob.lower())
        self.assertNotIn(" two wood", blob.lower())

    def test_rendered_prompt_has_no_world_block(self) -> None:
        entry = load_exo_trajectory_entry(-1)
        state = extract_state_from_entry(entry)
        cfg = EXO_ROOT / "templates/database_formulation/exo-planet.yaml"
        text = Renderer(config_path=cfg).render(
            {
                "goal": "goal: test",
                "obs": state,
                "act": ["LEFT"],
                "dialog": [],
                "action_history": [],
                "state_history": [],
                "knowledge": "(empty)",
            }
        )
        self.assertNotIn("## World", text)
        for leaked in (
            "Craftax",
            "craftax",
            "Craftax term",
            "`grass`",
            "`wood`",
            "PLACE_TABLE",
            "texture_mapping",
            "telemetry",
            "as exo term",
            "World →",
            "World tables",
            "map telemetry",
            "Progression hierarchy",
        ):
            self.assertNotIn(leaked, text, msg=f"Prompt leaks {leaked!r}")

    def test_exo_observation_uses_exo_terms(self) -> None:
        entry = load_exo_trajectory_entry(-1)
        state = extract_state_from_entry(entry)
        cfg = EXO_ROOT / "templates/no_dialog/exo-planet.yaml"
        text = Renderer(config_path=cfg).render(
            {
                "goal": "goal: test",
                "obs": state,
                "act": ["LEFT"],
                "dialog": [],
            }
        )
        self.assertIn("Regolith Turf", text)
        self.assertNotIn("standing on Grass", text)
        self.assertIn("Biomass=2", text)

    def test_all_configs_render(self) -> None:
        entry = load_exo_trajectory_entry(-1)
        state = extract_state_from_entry(entry)
        for cfg in CONFIGS:
            with self.subTest(config=str(cfg)):
                text = Renderer(config_path=cfg).render(
                    {
                        "goal": "goal: test",
                        "obs": state,
                        "act": ["LEFT", "EXTRACT", "ASK_OPERATOR"],
                        "dialog": [],
                        "action_history": [],
                        "state_history": [],
                        "knowledge": "(empty)",
                    }
                )
                self.assertNotIn("{{", text)
                self.assertIn("Survey Unit MC-3", text)
                self.assertIn("Regolith Turf", text)


if __name__ == "__main__":
    unittest.main()
