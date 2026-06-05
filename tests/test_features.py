import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vla_coreset.features.extract_clip import (
    FeatureRecord,
    build_index_frame,
    episode_to_split,
    left7_action,
    save_feature_artifacts,
)


class FeatureExtractionTest(unittest.TestCase):
    def test_episode_to_split_maps_all_protocol_groups(self):
        split = {
            "candidate_train": [0, 1],
            "val": [2],
            "test": [3],
        }

        self.assertEqual(episode_to_split(0, split), "candidate_train")
        self.assertEqual(episode_to_split(2, split), "val")
        self.assertEqual(episode_to_split(3, split), "test")
        with self.assertRaises(ValueError):
            episode_to_split(4, split)

    def test_left7_action_extracts_single_arm_label(self):
        action = np.arange(14, dtype=np.float32)

        left = left7_action(action)

        np.testing.assert_array_equal(left, np.arange(7, dtype=np.float32))

    def test_build_index_frame_has_required_columns(self):
        records = [
            FeatureRecord(
                row_index=0,
                episode_index=0,
                frame_index=0,
                timestamp=0.0,
                split="candidate_train",
                action_left=np.arange(7, dtype=np.float32),
            ),
            FeatureRecord(
                row_index=1,
                episode_index=40,
                frame_index=3,
                timestamp=0.06,
                split="val",
                action_left=np.arange(7, dtype=np.float32) + 1,
            ),
        ]

        index = build_index_frame(records)

        self.assertEqual(
            list(index.columns),
            [
                "row_index",
                "episode_index",
                "frame_index",
                "timestamp",
                "split",
                "action_left_0",
                "action_left_1",
                "action_left_2",
                "action_left_3",
                "action_left_4",
                "action_left_5",
                "action_left_6",
            ],
        )
        self.assertEqual(index.loc[1, "split"], "val")
        self.assertEqual(index.loc[1, "action_left_6"], 7.0)

    def test_save_feature_artifacts_writes_expected_files(self):
        features = np.zeros((2, 512), dtype=np.float32)
        actions = np.ones((2, 7), dtype=np.float32)
        text_features = np.full((1, 512), 2.0, dtype=np.float32)
        index = pd.DataFrame(
            {
                "row_index": [0, 1],
                "episode_index": [0, 1],
                "frame_index": [0, 0],
                "timestamp": [0.0, 0.0],
                "split": ["candidate_train", "candidate_train"],
                **{f"action_left_{i}": [0.0, 1.0] for i in range(7)},
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            save_feature_artifacts(output_dir, features, actions, text_features, index)

            self.assertTrue((output_dir / "features.npy").exists())
            self.assertTrue((output_dir / "actions_left7.npy").exists())
            self.assertTrue((output_dir / "text_features.npy").exists())
            self.assertTrue((output_dir / "index.parquet").exists())
            self.assertEqual(np.load(output_dir / "features.npy").shape, (2, 512))
