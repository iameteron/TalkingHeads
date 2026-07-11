from __future__ import annotations

import unittest

from megaprompt.emergency_message.from_dialog import render


class EmergencyMessageRendererTests(unittest.TestCase):
    def test_no_dialog(self) -> None:
        self.assertEqual(render({"dialog": [], "action_history": []}), "No emergency messages.")

    def test_operator_reply_without_database_update(self) -> None:
        payload = {
            "dialog": [{"question": "Where is biomass?", "answer": "North of the ridge."}],
            "action_history": ["DO"],
        }
        text = render(payload)
        self.assertIn("UPDATE_DATABASE", text)
        self.assertIn("operator link may drop soon", text)

    def test_operator_reply_after_database_update(self) -> None:
        payload = {
            "dialog": [{"question": "Where is biomass?", "answer": "North of the ridge."}],
            "action_history": ["ASK_OPERATOR", "UPDATE_DATABASE", "DO"],
        }
        self.assertEqual(render(payload), "No emergency messages.")

    def test_no_operator_answer(self) -> None:
        payload = {
            "dialog": [{"question": "Where is biomass?", "answer": ""}],
            "action_history": [],
        }
        self.assertEqual(render(payload), "No emergency messages.")


if __name__ == "__main__":
    unittest.main()
