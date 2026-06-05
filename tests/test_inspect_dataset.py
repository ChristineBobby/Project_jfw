import unittest

import pandas as pd

from vla_coreset.data.inspect_dataset import (
    action_range_rows,
    build_summary_rows,
    episode_length_stats,
)


class InspectDatasetTest(unittest.TestCase):
    def test_episode_length_stats_reports_count_and_range(self):
        episodes = pd.DataFrame(
            {
                "episode_index": [0, 1, 2],
                "length": [400, 400, 399],
            }
        )

        stats = episode_length_stats(episodes)

        self.assertEqual(
            stats,
            {
                "episode_count": 3,
                "min_episode_index": 0,
                "max_episode_index": 2,
                "min_frames_per_episode": 399,
                "max_frames_per_episode": 400,
            },
        )

    def test_build_summary_rows_uses_metadata_and_episode_stats(self):
        info = {
            "codebase_version": "v3.0",
            "robot_type": "aloha",
            "total_episodes": 50,
            "total_frames": 20000,
            "total_tasks": 1,
            "fps": 50,
            "features": {
                "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
                "observation.state": {"dtype": "float32", "shape": [14]},
                "action": {"dtype": "float32", "shape": [14]},
            },
        }
        episode_stats = {
            "episode_count": 50,
            "min_episode_index": 0,
            "max_episode_index": 49,
            "min_frames_per_episode": 400,
            "max_frames_per_episode": 400,
        }

        rows = build_summary_rows(info, episode_stats, ["transfer cube"])

        self.assertIn({"metric": "total_episodes", "value": "50"}, rows)
        self.assertIn({"metric": "total_frames", "value": "20000"}, rows)
        self.assertIn({"metric": "image_key", "value": "observation.images.top"}, rows)
        self.assertIn({"metric": "image_shape", "value": "480x640x3"}, rows)
        self.assertIn({"metric": "action_dim", "value": "14"}, rows)
        self.assertIn({"metric": "task_0", "value": "transfer cube"}, rows)

    def test_action_range_rows_formats_left_and_right_arm_ranges(self):
        stats = {
            "stats/action/min": [-1.0, -2.0, -3.0, -4.0],
            "stats/action/max": [1.0, 2.0, 3.0, 4.0],
        }
        motor_names = ["left_a", "left_b", "right_a", "right_b"]

        rows = action_range_rows(stats, motor_names)

        self.assertEqual(
            rows,
            [
                {"motor": "left_a", "min": "-1.000000", "max": "1.000000"},
                {"motor": "left_b", "min": "-2.000000", "max": "2.000000"},
                {"motor": "right_a", "min": "-3.000000", "max": "3.000000"},
                {"motor": "right_b", "min": "-4.000000", "max": "4.000000"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
