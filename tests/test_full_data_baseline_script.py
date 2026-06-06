import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "11_train_full_clip_baseline.py"
    spec = importlib.util.spec_from_file_location("train_full_clip_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullDataBaselineScriptTest(unittest.TestCase):
    def test_main_trains_on_all_candidate_train_episodes(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_path = root / "split_v1.json"
            split_path.write_text(
                json.dumps({"candidate_train": [0, 1, 2], "val": [3], "test": [4]}),
                encoding="utf-8",
            )
            table_dir = root / "tables"
            prediction_dir = root / "predictions"
            fake_index = pd.DataFrame({"split": ["candidate_train", "val", "test"]})

            def fake_train_mlp_baseline(**kwargs):
                self.assertEqual(kwargs["selected_episodes"], [0, 1, 2])
                self.assertEqual(kwargs["seed"], 9)
                return {
                    "metrics": {
                        "seed": 9,
                        "train_frames": 1200,
                        "val_frames": 400,
                        "test_frames": 400,
                        "test_original_mse": 0.012,
                        "test_normalized_mse": 0.123,
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
                    split_file=split_path,
                    table_dir=table_dir,
                    prediction_dir=prediction_dir,
                    seeds=[9],
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

            metrics = pd.read_csv(table_dir / "full_clip_seed9_metrics.csv")
            summary = pd.read_csv(table_dir / "full_clip_summary.csv")
            self.assertEqual(metrics.loc[0, "method"], "full_candidate_train")
            self.assertEqual(metrics.loc[0, "selected_episodes"], "0 1 2")
            self.assertEqual(int(metrics.loc[0, "train_frames"]), 1200)
            self.assertEqual(summary.loc[0, "method"], "full_candidate_train")
            self.assertTrue((prediction_dir / "full_clip_seed9.npz").exists())


if __name__ == "__main__":
    unittest.main()
