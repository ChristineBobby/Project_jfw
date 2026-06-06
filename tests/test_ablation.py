import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vla_coreset.selection.ablation import (
    ABLATION_VARIANTS,
    aggregate_ablation_summary,
    build_ablation_episode_scores,
    select_ablation_coreset,
    write_ablation_coreset_json,
)


class AblationSelectionTest(unittest.TestCase):
    def test_build_ablation_episode_scores_uses_variant_specific_columns(self):
        frame_scores = pd.DataFrame(
            {
                "episode_index": [0, 0, 1, 1],
                "visual_delta": [1.0, 3.0, 10.0, 10.0],
                "action_delta": [10.0, 10.0, 1.0, 3.0],
                "pc_score": [2.0, 2.0, 9.0, 9.0],
                "ras_score": [8.0, 8.0, 1.0, 1.0],
            }
        )

        visual = build_ablation_episode_scores(frame_scores, "visual_delta_only")
        action = build_ablation_episode_scores(frame_scores, "action_delta_only")
        pc_ras = build_ablation_episode_scores(frame_scores, "pc_ras_no_coverage")

        self.assertGreater(
            visual.loc[visual["episode_index"] == 1, "episode_score"].iloc[0],
            visual.loc[visual["episode_index"] == 0, "episode_score"].iloc[0],
        )
        self.assertGreater(
            action.loc[action["episode_index"] == 0, "episode_score"].iloc[0],
            action.loc[action["episode_index"] == 1, "episode_score"].iloc[0],
        )
        self.assertIn("method", pc_ras.columns)
        self.assertEqual(pc_ras["method"].iloc[0], "pc_ras_no_coverage")

    def test_select_ablation_coreset_supports_coverage_only(self):
        episode_scores = pd.DataFrame(
            {
                "episode_index": [0, 1, 2],
                "episode_score": [1.0, 0.9, 0.1],
                "method": ["coverage_only"] * 3,
            }
        )
        embeddings = pd.DataFrame(
            [[1.0, 0.0], [0.98, 0.0], [0.0, 1.0]],
            index=[0, 1, 2],
            dtype=np.float32,
        )

        selected, trace = select_ablation_coreset("coverage_only", episode_scores, embeddings, budget=2)

        self.assertEqual(selected, [0, 2])
        self.assertEqual(trace["method"].tolist(), ["coverage_only", "coverage_only"])
        self.assertEqual(trace["episode_index"].tolist(), [0, 2])

    def test_select_ablation_coreset_rejects_unknown_variant(self):
        episode_scores = pd.DataFrame({"episode_index": [0], "episode_score": [1.0]})
        embeddings = pd.DataFrame([[1.0, 0.0]], index=[0], dtype=np.float32)

        with self.assertRaises(ValueError):
            select_ablation_coreset("unknown", episode_scores, embeddings, budget=1)

    def test_write_ablation_coreset_json_has_traceable_fields(self):
        trace = pd.DataFrame(
            {
                "method": ["pc_only", "pc_only"],
                "selection_order": [1, 2],
                "episode_index": [3, 5],
                "final_score": [1.0, 0.8],
                "saliency_score": [1.0, 0.7],
                "diversity_score": [1.0, 0.9],
                "source_episode_score": [2.0, 1.5],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ablation_pc_only_top2.json"

            write_ablation_coreset_json(path, "pc_only", [3, 5], trace, "data/cache/selection/pc_ras_frame_scores.parquet")

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["method"], "ablation_pc_only")
            self.assertEqual(payload["variant"], "pc_only")
            self.assertEqual(payload["selected_episodes"], [3, 5])
            self.assertEqual(payload["source_frame_score_file"], "data/cache/selection/pc_ras_frame_scores.parquet")
            self.assertEqual(len(payload["selection_trace"]), 2)

    def test_aggregate_ablation_summary_reports_per_variant_statistics(self):
        metrics = pd.DataFrame(
            {
                "variant": ["a", "a", "b"],
                "seed": [0, 1, 0],
                "test_original_mse": [0.1, 0.3, 0.2],
                "test_normalized_mse": [1.0, 3.0, 2.0],
                "selected_episodes": ["1 2", "1 2", "3 4"],
            }
        )

        summary = aggregate_ablation_summary(metrics)

        self.assertEqual(summary["variant"].tolist(), ["a", "b"])
        self.assertAlmostEqual(summary.loc[0, "test_original_mse_mean"], 0.2)
        self.assertAlmostEqual(summary.loc[0, "test_original_mse_std"], np.sqrt(0.02))
        self.assertEqual(summary.loc[1, "num_seeds"], 1)

    def test_expected_variants_are_declared(self):
        self.assertEqual(
            ABLATION_VARIANTS,
            [
                "action_delta_only",
                "visual_delta_only",
                "pc_only",
                "pc_ras_no_coverage",
                "coverage_only",
                "pc_ras_full",
            ],
        )


if __name__ == "__main__":
    unittest.main()
