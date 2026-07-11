from __future__ import annotations

import unittest

from oracle.prompts.prompt_generation import (
    WORLD_MODE_CRAFTAX,
    WORLD_MODE_EXO,
    coerce_megaprompt_config_for_world_mode,
    create_action_prompt,
    create_goal_prompt,
    create_map_prompt,
    create_mechanics_promt,
    create_path_expert_prompt,
    create_path_prompt,
    create_question_prompt,
    generate_agent_prompt,
    list_megaprompt_configs_for_world_mode,
    normalize_world_mode,
)

EXO_LEGACY_TERMS = (
    "craftax",
    "minecraft",
    "crafting table",
    "PLACE_TABLE",
    "MAKE_WOOD_PICKAXE",
    "DO (TO GATHER SOMETHING)",
    "collect_wood",
    "wood pickaxe",
    "GRASS:",
    '"GRASS"',
)


class ExoExpertPromptTests(unittest.TestCase):
    def test_normalize_world_mode(self) -> None:
        self.assertEqual(normalize_world_mode("exo"), WORLD_MODE_EXO)
        self.assertEqual(normalize_world_mode("exo-planet"), WORLD_MODE_EXO)
        self.assertEqual(normalize_world_mode("craftax"), WORLD_MODE_CRAFTAX)

    def _assert_exo_prompt(self, text: str) -> None:
        lowered = text.lower()
        for term in EXO_LEGACY_TERMS:
            self.assertNotIn(term.lower(), lowered, msg=f"legacy term {term!r} in prompt")
        self.assertNotIn("{{", text)

    def test_exo_map_prompt_uses_exo_tile_names(self) -> None:
        text = create_map_prompt("Where is the nearest Energy Ore?", world_mode=WORLD_MODE_EXO)
        self._assert_exo_prompt(text)
        self.assertIn("Regolith Turf", text)
        self.assertIn("Energy Ore", text)

    def test_exo_mechanics_prompt_uses_exo_achievements(self) -> None:
        text = create_mechanics_promt("What do I need for PLACE_REPLICATOR?", world_mode=WORLD_MODE_EXO)
        self._assert_exo_prompt(text)
        self.assertIn("place_replicator", text)
        self.assertIn("exo_achievement_dependencies", text)

    def test_exo_action_prompt_lists_exo_actions(self) -> None:
        text = create_action_prompt(
            "What do I need for PLACE_REPLICATOR?",
            inventory="Biomass: 2",
            world_mode=WORLD_MODE_EXO,
        )
        self._assert_exo_prompt(text)
        self.assertIn("PLACE_REPLICATOR", text)
        self.assertIn("EXTRACT", text)

    def test_exo_goal_prompt(self) -> None:
        text = create_goal_prompt(
            goal="How do I deploy the Replicator?",
            question_1="map q",
            answer_1="map a",
            question_2="mech q",
            answer_2="mech a",
            action_answer="action a",
            world_mode=WORLD_MODE_EXO,
        )
        self._assert_exo_prompt(text)
        self.assertIn("Survey Unit MC-3", text)
        self.assertIn("Regolith Turf", text)
        self.assertIn("Survey Trail", text)
        self.assertIn("Dune Silts", text)

    def test_exo_question_prompt(self) -> None:
        text = create_question_prompt("Deploy Replicator", world_mode=WORLD_MODE_EXO)
        self._assert_exo_prompt(text)
        self.assertIn("map_expert_question", text)

    def test_exo_path_prompts(self) -> None:
        helper = create_path_prompt(
            "How do I reach the Brine Pool?",
            agent_position="[32, 32]",
            target_location="[30, 32]",
            world_mode=WORLD_MODE_EXO,
        )
        self._assert_exo_prompt(helper)
        infer = create_path_prompt(
            "Navigate to Basalt Crust ridge",
            agent_position="[32, 32]",
            infer_goal_from_question=True,
            world_mode=WORLD_MODE_EXO,
        )
        self._assert_exo_prompt(infer)
        expert = create_path_expert_prompt(
            "Which way from here?",
            rendered_map="PATH MAP",
            agent_position="[32, 32]",
            target_location="[30, 32]",
            colorful_waypoint="Brine Pool to the left",
            world_mode=WORLD_MODE_EXO,
        )
        self._assert_exo_prompt(expert)
        self.assertIn("Brine Pool", expert)

    def test_craftax_prompts_still_load(self) -> None:
        text = create_map_prompt("Where is coal?", world_mode=WORLD_MODE_CRAFTAX)
        self.assertIn("GRASS", text)

    def test_megaprompt_config_coercion(self) -> None:
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode("dialog", WORLD_MODE_EXO),
            "exo_reasoning_or_ask_help",
        )
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode("database_formulation", WORLD_MODE_EXO),
            "exo_database_formulation",
        )
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode("exo_no_dialog", WORLD_MODE_CRAFTAX),
            "no_dialog",
        )
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode("exo_database_formulation", WORLD_MODE_CRAFTAX),
            "database_formulation",
        )
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode(
                "arc_grid",
                WORLD_MODE_CRAFTAX,
                game_kind="craftax",
            ),
            "database_formulation",
        )
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode(
                "arc_image",
                WORLD_MODE_EXO,
                game_kind="craftax",
            ),
            "exo_database_formulation",
        )
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode(
                "arc_grid_image",
                WORLD_MODE_CRAFTAX,
                game_kind="craftax",
            ),
            "database_formulation",
        )
        self.assertEqual(
            coerce_megaprompt_config_for_world_mode(
                "arc_2_image",
                WORLD_MODE_EXO,
                game_kind="craftax",
            ),
            "exo_database_formulation",
        )

    def test_megaprompt_options_filtered_by_world_mode(self) -> None:
        craftax_options = list_megaprompt_configs_for_world_mode(WORLD_MODE_CRAFTAX)
        exo_options = list_megaprompt_configs_for_world_mode(WORLD_MODE_EXO)
        self.assertIn("dialog", craftax_options)
        self.assertNotIn("exo_database_formulation", craftax_options)
        self.assertIn("exo_database_formulation", exo_options)
        self.assertNotIn("dialog", exo_options)


    def test_generate_agent_prompt_fallback_respects_world_mode(self) -> None:
        craftax_prompt = generate_agent_prompt(
            goal="Collect stone",
            observation="map view",
            message_from_operator="",
            megaprompt_config_name="database_formulation",
            world_mode=WORLD_MODE_CRAFTAX,
        )
        exo_prompt = generate_agent_prompt(
            goal="Collect Biomass",
            observation="map view",
            message_from_operator="",
            megaprompt_config_name="database_formulation",
            world_mode=WORLD_MODE_EXO,
        )
        self.assertIn("Craftax grid world", craftax_prompt)
        self.assertIn("exoplanet", exo_prompt.lower())


if __name__ == "__main__":
    unittest.main()
