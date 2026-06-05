import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_train_pc_ras_baseline.py"
    spec = importlib.util.spec_from_file_location("train_pc_ras_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PCRASTrainingScriptTest(unittest.TestCase):
    def test_main_writes_method_metrics_and_predictions_from_pc_ras_coreset(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coreset_path = root / "pc_ras_episode_top5.json"
            coreset_path.write_text(
                json.dumps(
                    {
                        "method": "pc_ras_diversity_greedy",
                        "selected_episodes": [1, 2, 3, 4, 5],
                    }
                ),
                encoding="utf-8",
            )
            table_dir = root / "tables"
            prediction_dir = root / "predictions"
            fake_index = pd.DataFrame({"split": ["candidate_train", "val", "test"]})

            def fake_train_mlp_baseline(**kwargs):
                self.assertEqual(kwargs["selected_episodes"], [1, 2, 3, 4, 5])
                self.assertEqual(kwargs["seed"], 7)
                return {
                    "metrics": {
                        "seed": 7,
                        "train_frames": 2000,
                        "val_frames": 2000,
                        "test_frames": 2000,
                        "test_original_mse": 0.123,
                        "test_normalized_mse": 0.456,
                    },
                    "val_pred": np.zeros((2, 7), dtype=np.float32),
                    "test_pred": np.ones((2, 7), dtype=np.float32),
                    "val_target": np.zeros((2, 7), dtype=np.float32),
                    "test_target": np.ones((2, 7), dtype=np.float32),
                }

            with mock.patch.object(
                module,
                "load_feature_artifacts",
                return_value=(
                    np.zeros((3, 2), dtype=np.float32),
                    np.zeros((3, 7), dtype=np.float32),
                    np.zeros((1, 2), dtype=np.float32),
                    fake_index,
                ),
            ), mock.patch.object(module, "make_feature_matrix", return_value=np.zeros((3, 4), dtype=np.float32)), mock.patch.object(
                module,
                "train_mlp_baseline",
                side_effect=fake_train_mlp_baseline,
            ), mock.patch.object(
                module,
                "parse_args",
                return_value=module.argparse.Namespace(
                    feature_dir=root / "features",
                    coreset_file=coreset_path,
                    table_dir=table_dir,
                    prediction_dir=prediction_dir,
                    seeds=[7],
                    device="cpu",
                    batch_size=8,
                    max_epochs=5,
                    patience=2,
                    lr=1e-3,
                    weight_decay=1e-4,
                    smoke=False,
                ),
            ):
                module.main()

            metrics = pd.read_csv(table_dir / "pc_ras_clip_seed7_metrics.csv")
            summary = pd.read_csv(table_dir / "pc_ras_clip_summary.csv")
            self.assertEqual(metrics.loc[0, "method"], "pc_ras_diversity_greedy")
            self.assertEqual(metrics.loc[0, "selected_episodes"], "1 2 3 4 5")
            self.assertEqual(int(metrics.loc[0, "train_frames"]), 2000)
            self.assertEqual(summary.loc[0, "method"], "pc_ras_diversity_greedy")
            self.assertTrue((prediction_dir / "pc_ras_clip_seed7.npz").exists())


if __name__ == "__main__":
    unittest.main()
