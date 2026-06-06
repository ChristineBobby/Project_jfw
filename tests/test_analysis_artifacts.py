import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vla_coreset.analysis.artifacts import (
    build_selected_episode_explanations,
    compute_phase_mse_summary,
    make_action_curve_figure,
    make_budget_curve_figure,
    make_phase_mse_figure,
    make_score_timeline_figure,
)


class AnalysisArtifactsTest(unittest.TestCase):
    def test_build_selected_episode_explanations_includes_top_frames_and_reason(self):
        frame_scores = pd.DataFrame(
            {
                "episode_index": [1, 1, 1, 2, 2],
                "frame_index": [0, 1, 2, 0, 1],
                "phase_bin": [0, 1, 1, 0, 1],
                "frame_score": [0.2, 0.9, 0.7, 0.1, 0.3],
                "visual_delta": [1.0, 2.0, 3.0, 1.0, 1.0],
                "action_delta": [0.1, 0.2, 0.3, 0.1, 0.1],
                "gripper_delta": [0.0, 0.5, 0.0, 0.0, 0.0],
            }
        )
        episode_scores = pd.DataFrame(
            {
                "episode_index": [1, 2],
                "episode_score": [0.8, 0.2],
                "episode_pc_score": [1.1, 0.4],
                "episode_ras_score": [0.6, 0.2],
                "phase_coverage": [2, 2],
                "mean_visual_delta": [2.0, 1.0],
                "mean_action_delta": [0.2, 0.1],
            }
        )
        trace = pd.DataFrame(
            {
                "selection_order": [1],
                "episode_index": [1],
                "final_score": [1.0],
                "diversity_score": [0.8],
                "raw_diversity_distance": [0.4],
            }
        )

        table = build_selected_episode_explanations(frame_scores, episode_scores, trace, [1])

        self.assertEqual(table.loc[0, "episode_index"], 1)
        self.assertEqual(table.loc[0, "top_frame_indices"], "1 2 0")
        self.assertEqual(table.loc[0, "phase_coverage"], 2)
        self.assertIn("high PC-RAS score", table.loc[0, "selected_reason"])

    def test_compute_phase_mse_summary_bins_test_predictions(self):
        index = pd.DataFrame(
            {
                "split": ["test", "test", "test", "test"],
                "episode_index": [4, 4, 4, 4],
                "frame_index": [0, 1, 2, 3],
            }
        )
        predictions = {
            "A": [
                {
                    "seed": 0,
                    "test_pred": np.zeros((4, 7), dtype=np.float32),
                    "test_target": np.ones((4, 7), dtype=np.float32),
                }
            ],
            "B": [
                {
                    "seed": 0,
                    "test_pred": np.ones((4, 7), dtype=np.float32),
                    "test_target": np.ones((4, 7), dtype=np.float32),
                }
            ],
        }

        summary = compute_phase_mse_summary(index, predictions, phase_bins=2)

        self.assertEqual(set(summary["method"]), {"A", "B"})
        self.assertEqual(summary.loc[(summary["method"] == "A") & (summary["phase_bin"] == 0), "mse_mean"].iloc[0], 1.0)
        self.assertEqual(summary.loc[(summary["method"] == "B") & (summary["phase_bin"] == 1), "mse_mean"].iloc[0], 0.0)

    def test_figures_write_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_scores = pd.DataFrame(
                {
                    "episode_index": [1, 1, 1, 1],
                    "frame_index": [0, 1, 2, 3],
                    "phase": [0.0, 0.33, 0.66, 1.0],
                    "phase_bin": [0, 1, 2, 2],
                    "frame_score": [0.2, 0.6, 0.9, 0.4],
                    "pc_score": [0.3, 0.7, 1.0, 0.5],
                    "ras_score": [0.1, 0.4, 0.8, 0.3],
                }
            )
            actions = pd.DataFrame(
                {
                    "episode_index": [1, 1, 2, 2],
                    "frame_index": [0, 1, 0, 1],
                    "group": ["PC-RAS", "PC-RAS", "Random", "Random"],
                    "action_0": [0.0, 1.0, 0.5, 0.8],
                    "action_1": [0.0, 0.2, 0.1, 0.3],
                    "action_2": [0.0, 0.4, 0.2, 0.5],
                    "action_6": [0.1, 0.9, 0.2, 0.7],
                }
            )
            phase = pd.DataFrame(
                {
                    "method": ["Random-10%", "Random-10%", "PC-RAS", "PC-RAS"],
                    "phase_bin": [0, 1, 0, 1],
                    "phase_label": ["0-50%", "50-100%", "0-50%", "50-100%"],
                    "mse_mean": [0.2, 0.3, 0.1, 0.2],
                    "mse_std": [0.01, 0.02, 0.01, 0.02],
                }
            )
            budget = pd.DataFrame(
                {
                    "method": ["Random", "Random", "PC-RAS", "PC-RAS"],
                    "budget_percent": [5, 10, 5, 10],
                    "test_original_mse_mean": [0.01, 0.008, 0.009, 0.006],
                    "test_original_mse_std": [0.001, 0.001, 0.001, 0.001],
                }
            )

            written = []
            written.extend(make_score_timeline_figure(frame_scores, 1, root))
            written.extend(make_action_curve_figure(actions, root))
            written.extend(make_phase_mse_figure(phase, root))
            written.extend(make_budget_curve_figure(budget, root))

            self.assertEqual(len(written), 8)
            for path in written:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 1000, path)


if __name__ == "__main__":
    unittest.main()
