import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vla_coreset.selection.episode_selector import (
    build_episode_embeddings,
    select_diverse_episodes,
    write_pc_ras_coreset_json,
)


class EpisodeSelectorTest(unittest.TestCase):
    def test_build_episode_embeddings_returns_normalized_episode_rows(self):
        features = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )
        actions = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        index = pd.DataFrame({"episode_index": [0, 0, 1, 1], "split": ["candidate_train"] * 4})

        embeddings = build_episode_embeddings(features, actions, index)

        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.index.tolist(), [0, 1])
        norms = np.linalg.norm(embeddings.to_numpy(dtype=np.float32), axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)

    def test_select_diverse_episodes_starts_with_highest_saliency(self):
        episode_scores = pd.DataFrame(
            {
                "episode_index": [0, 1, 2],
                "episode_score": [0.1, 0.9, 0.8],
            }
        )
        embeddings = pd.DataFrame(
            [[1.0, 0.0], [0.9, 0.0], [0.0, 1.0]],
            index=[0, 1, 2],
            dtype=np.float32,
        )

        selected, trace = select_diverse_episodes(episode_scores, embeddings, budget=2)

        self.assertEqual(selected[0], 1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(trace.loc[0, "selection_order"], 1)
        self.assertEqual(trace.loc[0, "episode_index"], 1)

    def test_select_diverse_episodes_can_prefer_coverage_over_pure_top_two(self):
        episode_scores = pd.DataFrame(
            {
                "episode_index": [0, 1, 2],
                "episode_score": [1.0, 0.95, 0.8],
            }
        )
        embeddings = pd.DataFrame(
            [[1.0, 0.0], [0.98, 0.0], [0.0, 1.0]],
            index=[0, 1, 2],
            dtype=np.float32,
        )

        selected, trace = select_diverse_episodes(
            episode_scores,
            embeddings,
            budget=2,
            saliency_weight=0.4,
            diversity_weight=0.6,
        )

        self.assertEqual(selected, [0, 2])
        self.assertEqual(trace["episode_index"].tolist(), [0, 2])
        self.assertGreater(trace.loc[1, "diversity_score"], trace.loc[1, "saliency_score"])

    def test_select_diverse_episodes_rejects_invalid_budget(self):
        episode_scores = pd.DataFrame({"episode_index": [0], "episode_score": [1.0]})
        embeddings = pd.DataFrame([[1.0, 0.0]], index=[0], dtype=np.float32)

        with self.assertRaises(ValueError):
            select_diverse_episodes(episode_scores, embeddings, budget=2)

    def test_write_pc_ras_coreset_json_has_traceable_fields(self):
        trace = pd.DataFrame(
            {
                "selection_order": [1, 2],
                "episode_index": [3, 5],
                "final_score": [1.0, 0.8],
                "saliency_score": [1.0, 0.7],
                "diversity_score": [1.0, 0.9],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pc_ras_episode_top2.json"

            write_pc_ras_coreset_json(
                path=path,
                selected_episodes=[3, 5],
                trace=trace,
                saliency_weight=0.65,
                diversity_weight=0.35,
                score_file="results/tables/pc_ras_episode_scores.csv",
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["method"], "pc_ras_diversity_greedy")
            self.assertEqual(payload["unit"], "episode")
            self.assertEqual(payload["budget_episodes"], 2)
            self.assertEqual(payload["selected_episodes"], [3, 5])
            self.assertEqual(payload["weights"]["saliency"], 0.65)
            self.assertEqual(payload["source_score_file"], "results/tables/pc_ras_episode_scores.csv")
            self.assertEqual(len(payload["selection_trace"]), 2)


if __name__ == "__main__":
    unittest.main()
