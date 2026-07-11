import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle.knowledge import store as knowledge_store


class KnowledgeStoreTests(unittest.TestCase):
    def _patch_paths(self, tmp_path: Path):
        return (
            mock.patch.object(knowledge_store, "KNOWLEDGE_JSON_PATH", tmp_path / "knowledge_data.json"),
            mock.patch.object(knowledge_store, "KNOWLEDGE_DATA_PATH", tmp_path / "knowledge_data.txt"),
            mock.patch.object(knowledge_store, "DATABASE_FORMULATION_DIR", tmp_path),
            mock.patch.object(knowledge_store, "REASONING_TEMPLATE_PATH", tmp_path / "reasoning.txt"),
        )

    def test_render_table(self) -> None:
        entries = [
            {
                "id": 1,
                "type": "RECIPE",
                "skill": "wood_pickaxe",
                "recipe": "1 wood",
                "rules": "",
            }
        ]
        rendered = knowledge_store.render_knowledge_table(entries)
        self.assertIn("wood_pickaxe", rendered)
        self.assertIn("| ID |", rendered)

    def test_merge_upserts_by_type_and_skill(self) -> None:
        existing = [
            {
                "id": 1,
                "type": "RECIPE",
                "skill": "pickaxe",
                "recipe": "old",
                "rules": "",
            }
        ]
        merged = knowledge_store.merge_knowledge_entries(
            existing,
            "TYPE=RECIPE\nSKILL=pickaxe\nRECIPE=3 wood",
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["recipe"], "3 wood")

    def test_merge_legacy_lines(self) -> None:
        merged = knowledge_store.merge_knowledge(
            "",
            "RECIPE: pickaxe needs wood\nMECHANICS: coal needs pickaxe",
        )
        self.assertIn("pickaxe", merged)
        self.assertIn("coal", merged)

    def test_save_writes_json_and_txt(self) -> None:
        entries = [
            {
                "id": 1,
                "type": "MECHANICS",
                "skill": "water_movement",
                "recipe": "",
                "rules": "no water walk",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            patches = self._patch_paths(tmp_path)
            for p in patches:
                p.start()
            try:
                knowledge_store.save_knowledge_records(entries)
                data = json.loads((tmp_path / "knowledge_data.json").read_text(encoding="utf-8"))
                self.assertEqual(len(data["entries"]), 1)
                txt = (tmp_path / "knowledge_data.txt").read_text(encoding="utf-8")
                self.assertIn("water_movement", txt)
            finally:
                for p in patches:
                    p.stop()

    def test_extract_and_strip_to_database(self) -> None:
        raw = (
            "<reasoning>learned recipe</reasoning>\n"
            "<to_database>\nTYPE=RECIPE\nSKILL=crafting_table\nRECIPE=4 wood\n</to_database>\n"
            "<action>PLACE_TABLE</action>"
        )
        blocks = knowledge_store.extract_to_database_blocks(raw)
        self.assertIn("TYPE=RECIPE", blocks[0])
        stripped = knowledge_store.strip_to_database_tags(raw)
        self.assertNotIn("to_database", stripped.lower())
        self.assertIn("<action>PLACE_TABLE</action>", stripped)

    def test_merge_multiple_records_in_one_block(self) -> None:
        block = (
            "OP=UPSERT\nTYPE=RECIPE\nSKILL=a\nRECIPE=x\n\n"
            "OP=UPSERT\nTYPE=MECHANICS\nSKILL=b\nRULES=y"
        )
        merged = knowledge_store.merge_knowledge_entries([], block)
        self.assertEqual(len(merged), 2)
        skills = {e["skill"] for e in merged}
        self.assertEqual(skills, {"a", "b"})

    def test_note_not_persisted_to_disk(self) -> None:
        raw = (
            "<to_database>\n"
            "OP=UPSERT\nTYPE=NOTE\nSKILL=crafting_table\n"
            "RULES=placed at [1, 2]\n"
            "</to_database>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            patches = self._patch_paths(tmp_path)
            for p in patches:
                p.start()
            try:
                (tmp_path / "reasoning.txt").write_text("template", encoding="utf-8")
                knowledge_store.clear_episode_notes()
                _, updated, _ = knowledge_store.apply_knowledge_from_response(raw)
                self.assertTrue(updated)
                saved = json.loads((tmp_path / "knowledge_data.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["entries"], [])
                notes = knowledge_store.load_episode_note_records()
                self.assertEqual(len(notes), 1)
                self.assertEqual(notes[0]["skill"], "crafting_table")
            finally:
                for p in patches:
                    p.stop()

    def test_clear_episode_notes_at_run_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            patches = self._patch_paths(tmp_path)
            for p in patches:
                p.start()
            try:
                knowledge_store.save_knowledge_records(
                    [
                        {
                            "id": 1,
                            "type": "MECHANICS",
                            "skill": "water",
                            "recipe": "",
                            "rules": "no walk",
                        },
                        {
                            "id": 2,
                            "type": "NOTE",
                            "skill": "table",
                            "recipe": "",
                            "rules": "old note",
                        },
                    ]
                )
                knowledge_store.set_episode_note_records(
                    [
                        {
                            "id": 3,
                            "type": "NOTE",
                            "skill": "coal",
                            "recipe": "",
                            "rules": "ephemeral",
                        }
                    ]
                )
                knowledge_store.clear_episode_notes()
                self.assertEqual(knowledge_store.load_episode_note_records(), [])
                durable = knowledge_store.load_durable_knowledge_records()
                self.assertEqual(len(durable), 1)
                self.assertEqual(durable[0]["type"], "MECHANICS")
            finally:
                for p in patches:
                    p.stop()

    def test_merge_note_same_format_as_other_types(self) -> None:
        block = (
            "OP=UPSERT\nTYPE=NOTE\nSKILL=crafting_table\n"
            "RULES=placed crafting table at coord [10, 20]"
        )
        merged = knowledge_store.merge_knowledge_entries([], block)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["type"], "NOTE")
        self.assertEqual(merged[0]["skill"], "crafting_table")
        self.assertIn("[10, 20]", merged[0]["rules"])

        updated = knowledge_store.merge_knowledge_entries(
            merged,
            "OP=UPSERT\nTYPE=NOTE\nSKILL=crafting_table\nRULES=table at [10, 20]; agent adjacent",
        )
        self.assertEqual(len(updated), 1)
        self.assertIn("adjacent", updated[0]["rules"])

    def test_apply_knowledge_from_response(self) -> None:
        raw = (
            "<to_database>\n"
            "TYPE=MECHANICS\nSKILL=test_rule\nRULES=test value\n"
            "</to_database><action>X</action>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            patches = self._patch_paths(tmp_path)
            for p in patches:
                p.start()
            try:
                (tmp_path / "reasoning.txt").write_text("template", encoding="utf-8")
                cleaned, updated, block = knowledge_store.apply_knowledge_from_response(raw)
                self.assertTrue(updated)
                self.assertIn("TYPE=MECHANICS", block)
                self.assertNotIn("to_database", cleaned.lower())
                saved = json.loads((tmp_path / "knowledge_data.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["entries"][0]["skill"], "test_rule")
                txt = (tmp_path / "knowledge_data.txt").read_text(encoding="utf-8")
                self.assertIn("test_rule", txt)
            finally:
                for p in patches:
                    p.stop()

    def test_copy_durable_knowledge_excludes_notes(self) -> None:
        source_entries = [
            {
                "id": 1,
                "type": "RECIPE",
                "skill": "wood_pickaxe",
                "recipe": "1 wood",
                "rules": "",
            },
            {
                "id": 2,
                "type": "NOTE",
                "skill": "crafting_table",
                "recipe": "",
                "rules": "placed at [1, 2]",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_json = tmp_path / "source.json"
            source_txt = tmp_path / "source.txt"
            dest_json = tmp_path / "dest.json"
            dest_txt = tmp_path / "dest.txt"
            with knowledge_store.use_knowledge_paths(json_path=source_json, txt_path=source_txt):
                knowledge_store.save_knowledge_records(source_entries)
            knowledge_store.copy_durable_knowledge(
                source_json=source_json,
                source_txt=source_txt,
                dest_json=dest_json,
                dest_txt=dest_txt,
            )
            copied = json.loads(dest_json.read_text(encoding="utf-8"))
            self.assertEqual(len(copied["entries"]), 1)
            self.assertEqual(copied["entries"][0]["type"], "RECIPE")


if __name__ == "__main__":
    unittest.main()
