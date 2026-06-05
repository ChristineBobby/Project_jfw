import unittest

import numpy as np
import pandas as pd

from vla_coreset.selection.scores import (
    aggregate_episode_scores,
    compute_frame_scores,
    robust_normalize,
)


class PCRASScoresTest(unittest.TestCase):
    def test_robust_normalize_maps_percentile_range_and_clips_outliers(self):
        values = np.array([-100.0, 0.0, 0.0, 5.0, 10.0, 10.0, 100.0], dtype=np.float32)

        normalized = robust_normalize(values, lower=25.0, upper=75.0)

        np.testing.assert_allclose(
            normalized,
            np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_compute_frame_scores_reset_temporal_deltas_at_episode_boundaries(self):
        features = np.array(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [100.0, 100.0],
                [101.0, 100.0],
            ],
            dtype=np.float32,
        )
        actions = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [13.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        index = pd.DataFrame(
            {
                "episode_index": [0, 0, 1, 1],
                "frame_index": [0, 1, 0, 1],
                "split": ["candidate_train"] * 4,
            }
        )

        scores = compute_frame_scores(features, actions, index)

        self.assertEqual(scores.loc[0, "visual_delta"], 0.0)
        self.assertEqual(scores.loc[2, "visual_delta"], 0.0)
        self.assertAlmostEqual(scores.loc[1, "visual_delta"], 5.0)
        self.assertAlmostEqual(scores.loc[3, "visual_delta"], 1.0)
        self.assertEqual(scores.loc[0, "action_delta"], 0.0)
        self.assertEqual(scores.loc[2, "action_delta"], 0.0)
        self.assertAlmostEqual(scores.loc[3, "action_delta"], np.sqrt(26.0))
        self.assertTrue((scores["frame_score"] >= 0.0).all())

    def test_aggregate_episode_scores_uses_top_fraction_frames(self):
        frame_scores = pd.DataFrame(
            {
                "episode_index": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                "frame_score": [0.0, 0.2, 0.4, 0.8, 1.0, 0.1, 0.1, 0.1, 0.9, 0.9],
                "pc_score": [0.0, 0.2, 0.4, 0.8, 1.0, 0.1, 0.1, 0.1, 0.9, 0.9],
                "ras_score": [0.0, 0.2, 0.4, 0.8, 1.0, 0.1, 0.1, 0.1, 0.9, 0.9],
                "action_delta": [0.0, 0.2, 0.4, 0.8, 1.0, 0.1, 0.1, 0.1, 0.9, 0.9],
                "visual_delta": [0.0, 0.2, 0.4, 0.8, 1.0, 0.1, 0.1, 0.1, 0.9, 0.9],
                "phase_bin": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            }
        )

        episode_scores = aggregate_episode_scores(frame_scores, top_fraction=0.2)

        self.assertEqual(episode_scores["episode_index"].tolist(), [0, 1])
        self.assertAlmostEqual(episode_scores.loc[0, "episode_score"], 1.0)
        self.assertAlmostEqual(episode_scores.loc[1, "episode_score"], 0.9)
        self.assertEqual(episode_scores.loc[0, "frames"], 5)
        self.assertEqual(episode_scores.loc[0, "top_frames"], 1)
        self.assertIn("phase_coverage", episode_scores.columns)


if __name__ == "__main__":
    unittest.main()
