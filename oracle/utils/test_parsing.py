from __future__ import annotations

import unittest

from oracle.utils.parsing import (
    default_expert_questions_for_goal,
    ensure_expert_questions,
    parse_question_expert_response,
)


class ParsingTests(unittest.TestCase):
    def test_parse_question_expert_response_from_codeblock(self) -> None:
        text = """Here:
```python
{
  "map_expert_question": "Where is coal?",
  "mechanics_expert_question": "How to mine coal?",
  "action_expert_question": "What action next?"
}
```"""
        parsed = parse_question_expert_response(text)
        self.assertEqual(parsed["map_expert_question"], "Where is coal?")
        self.assertEqual(parsed["mechanics_expert_question"], "How to mine coal?")

    def test_ensure_expert_questions_uses_defaults_when_empty(self) -> None:
        result = ensure_expert_questions({}, "Deploy Replicator")
        self.assertIn("Deploy Replicator", result["map_expert_question"])
        self.assertIn("Deploy Replicator", result["mechanics_expert_question"])
        self.assertIn("Deploy Replicator", result["action_expert_question"])

    def test_ensure_expert_questions_fills_missing_only(self) -> None:
        partial = {
            "map_expert_question": "Where is Basalt Crust?",
            "mechanics_expert_question": "",
            "action_expert_question": "",
        }
        result = ensure_expert_questions(partial, "Collect Basalt Shard")
        self.assertEqual(result["map_expert_question"], "Where is Basalt Crust?")
        self.assertTrue(result["mechanics_expert_question"])
        self.assertTrue(result["action_expert_question"])

    def test_default_expert_questions_for_goal(self) -> None:
        defaults = default_expert_questions_for_goal("Collect Biomass")
        self.assertIn("Collect Biomass", defaults["action_expert_question"])


if __name__ == "__main__":
    unittest.main()
