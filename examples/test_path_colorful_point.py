import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from oracle.utils.path_expert_pipeline import PathExpertPipeline


class TestFindNearestMostColorfulPathPoint(unittest.TestCase):
    def test_selects_nearest_among_top3_colorful_candidates(self):
        with patch.object(
            PathExpertPipeline,
            "_tile_name_from_id",
            side_effect=lambda block_id: {
                0: "Grass",
                1: "Tree",
                2: "Water",
                3: "Stone",
            }[int(block_id)],
        ):
            grid = np.zeros((7, 7), dtype=np.int32)
            # p1 -> colorfulness 3 (Tree, Water, Stone)
            grid[0, 1] = 1
            grid[2, 1] = 2
            grid[1, 0] = 3
            # p0 -> colorfulness 2 (Tree, Water) and nearest to agent
            grid[3, 0] = 1
            grid[4, 1] = 2
            # p2 -> colorfulness 2 (Tree, Water)
            grid[0, 3] = 1
            grid[2, 3] = 2
            # p3 -> colorfulness 1 (Tree), should be outside top-3
            grid[3, 4] = 1

            state = SimpleNamespace(map=grid, player_position=np.array([3, 0]))
            predicted_path = [[3, 1], [1, 1], [1, 3], [3, 5]]

            result = PathExpertPipeline.find_nearest_most_colorful_path_point(
                state=state,
                predicted_path=predicted_path,
                near_radius=1,
            )

            self.assertEqual(result["point"], [3, 1])
            self.assertEqual(result["point_index"], 0)
            self.assertFalse(result["used_secondary_fallback"])

    def test_selects_nearest_point_when_colorfulness_ties(self):
        # 0,1,2 map directly to three names (grass excluded from colorfulness).
        with patch.object(
            PathExpertPipeline,
            "_tile_name_from_id",
            side_effect=lambda block_id: {0: "Grass", 1: "Tree", 2: "Water", 3: "Stone"}[int(block_id)],
        ):
            grid = np.array(
                [
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 2, 0, 3],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
            state = SimpleNamespace(map=grid)
            predicted_path = [[1, 1], [2, 2], [2, 3]]

            result = PathExpertPipeline.find_nearest_most_colorful_path_point(
                state=state,
                predicted_path=predicted_path,
                near_radius=1,
            )

            # First two points both see 2 unique non-grass names; nearest on path must win.
            self.assertEqual(result["point"], [1, 1])
            self.assertEqual(result["point_index"], 0)
            self.assertEqual(result["colorfulness"], 2)
            self.assertEqual(result["used_secondary_fallback"], False)

            blocks = result["blocks"]
            block_names = {b["name"] for b in blocks}
            self.assertSetEqual(block_names, {"Tree", "Water"})
            for block in blocks:
                self.assertEqual(len(block["coord"]), 2)

    def test_uses_secondary_fallback_when_no_diverse_point(self):
        with patch.object(
            PathExpertPipeline,
            "_tile_name_from_id",
            side_effect=lambda block_id: {0: "Grass", 1: "Tree"}[int(block_id)],
        ):
            # Every neighborhood has at most one non-grass type (Tree),
            # so fallback should choose by repeated non-grass count.
            grid = np.array(
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
            state = SimpleNamespace(map=grid)
            predicted_path = [[0, 0], [1, 1], [1, 2]]

            result = PathExpertPipeline.find_nearest_most_colorful_path_point(
                state=state,
                predicted_path=predicted_path,
                near_radius=1,
            )

            self.assertEqual(result["point"], [1, 1])
            self.assertEqual(result["point_index"], 1)
            self.assertEqual(result["colorfulness"], 1)
            self.assertTrue(result["used_secondary_fallback"])
            self.assertGreaterEqual(result["secondary_colorfulness"], 2)
            self.assertEqual({b["name"] for b in result["blocks"]}, {"Tree"})

    def test_excludes_agent_position_from_candidates(self):
        with patch.object(
            PathExpertPipeline,
            "_tile_name_from_id",
            side_effect=lambda block_id: {0: "Grass", 1: "Tree"}[int(block_id)],
        ):
            grid = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
            state = SimpleNamespace(map=grid, player_position=np.array([1, 1]))
            predicted_path = [[1, 1], [1, 2], [2, 2]]

            result = PathExpertPipeline.find_nearest_most_colorful_path_point(
                state=state,
                predicted_path=predicted_path,
                near_radius=1,
            )

            self.assertNotEqual(result["point"], [1, 1])
            self.assertEqual(result["point_index"], 1)

    def test_returns_empty_result_for_invalid_input(self):
        result = PathExpertPipeline.find_nearest_most_colorful_path_point(
            state=None,
            predicted_path=[[0, 0]],
            near_radius=1,
        )
        self.assertEqual(result["point"], None)
        self.assertEqual(result["point_index"], -1)
        self.assertEqual(result["colorfulness"], 0)
        self.assertEqual(result["secondary_colorfulness"], 0)
        self.assertEqual(result["used_secondary_fallback"], False)
        self.assertEqual(result["blocks"], [])


class TestFormatColorfulWaypointSection(unittest.TestCase):
    def test_empty_predicted_path_returns_empty_string(self):
        state = SimpleNamespace(map=np.zeros((2, 2), dtype=np.int32))
        self.assertEqual(
            PathExpertPipeline._format_colorful_waypoint_section(state, [], near_radius=1),
            "",
        )

    def test_returns_empty_when_no_valid_path_points(self):
        state = SimpleNamespace(map=np.zeros((3, 3), dtype=np.int32))
        section = PathExpertPipeline._format_colorful_waypoint_section(
            state, [[99, 99]], near_radius=1
        )
        self.assertEqual(section, "")

    def test_section_contains_coords_landmarks_and_reminder(self):
        with patch.object(
            PathExpertPipeline,
            "_tile_name_from_id",
            side_effect=lambda block_id: {0: "Grass", 1: "Tree", 2: "Water", 3: "Stone"}[
                int(block_id)
            ],
        ):
            grid = np.array(
                [
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 2, 0, 3],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
            state = SimpleNamespace(map=grid)
            predicted_path = [[1, 1], [2, 2], [2, 3]]
            section = PathExpertPipeline._format_colorful_waypoint_section(
                state, predicted_path, near_radius=1
            )

        self.assertNotIn("### Next landmark waypoint", section)
        self.assertIn("(1, 1)", section)
        self.assertIn("Block that is", section)
        self.assertIn("Tree", section)
        self.assertIn("Water", section)


if __name__ == "__main__":
    unittest.main()
