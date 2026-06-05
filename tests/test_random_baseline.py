import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vla_coreset.selection.random_baseline import (
    build_random_episode_coreset,
    write_coreset_json,
)
from vla_coreset.training.mlp_baseline import (
    ActionNormalizer,
    MLPRegressor,
    compute_mse_metrics,
    make_feature_matrix,
)


class RandomBaselineTest(unittest.TestCase):
    def test_random_episode_coreset_is_reproducible_and_train_only(self):
        candidate_train = list(range(40))

        first = build_random_episode_coreset(candidate_train, budget=5, seed=3)
        second = build_random_episode_coreset(candidate_train, budget=5, seed=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(set(first).issubset(candidate_train))
        self.assertEqual(first, sorted(first))

    def test_write_coreset_json_has_traceable_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random_episode_seed0.json"

            write_coreset_json(path, seed=0, selected_episodes=[1, 2, 3, 4, 5])

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["method"], "random_episode")
            self.assertEqual(payload["seed"], 0)
            self.assertEqual(payload["unit"], "episode")
            self.assertEqual(payload["budget_episodes"], 5)
            self.assertEqual(payload["selected_episodes"], [1, 2, 3, 4, 5])

    def test_make_feature_matrix_concatenates_text_feature_per_row(self):
        image_features = np.ones((3, 2), dtype=np.float32)
        text_features = np.array([[2.0, 3.0]], dtype=np.float32)

        matrix = make_feature_matrix(image_features, text_features)

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1.0, 1.0, 2.0, 3.0],
                    [1.0, 1.0, 2.0, 3.0],
                    [1.0, 1.0, 2.0, 3.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_action_normalizer_round_trips_values(self):
        actions = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)

        normalizer = ActionNormalizer.fit(actions)
        restored = normalizer.inverse_transform(normalizer.transform(actions))

        np.testing.assert_allclose(restored, actions, atol=1e-6)

    def test_mlp_regressor_output_shape(self):
        model = MLPRegressor(input_dim=4, output_dim=7)
        batch = torch.zeros((5, 4), dtype=torch.float32)

        output = model(batch)

        self.assertEqual(tuple(output.shape), (5, 7))

    def test_compute_mse_metrics_reports_overall_and_per_dim(self):
        pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target = np.array([[0.0, 2.0], [1.0, 6.0]], dtype=np.float32)

        metrics = compute_mse_metrics(pred, target, prefix="test")

        self.assertAlmostEqual(metrics["test_mse"], 2.25)
        self.assertAlmostEqual(metrics["test_action_0_mse"], 2.5)
        self.assertAlmostEqual(metrics["test_action_1_mse"], 2.0)
