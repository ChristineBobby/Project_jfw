import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "14_run_resnet18_robustness.py"
    spec = importlib.util.spec_from_file_location("run_resnet18_robustness", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResNet18RobustnessScriptTest(unittest.TestCase):
    def test_main_writes_random_and_pc_ras_summaries(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_path = root / "split_v1.json"
            split_path.write_text(
                json.dumps({"candidate_train": [0, 1, 2, 3, 4], "val": [5], "test": [6]}),
                encoding="utf-8",
            )
            coreset_path = root / "pc_ras_episode_top2.json"
            coreset_path.write_text(json.dumps({"selected_episodes": [3, 4]}), encoding="utf-8")
            fake_index = pd.DataFrame({"split": ["candidate_train", "val", "test"], "episode_index": [0, 5, 6]})

            calls = []

            def fake_train_mlp_baseline(**kwargs):
                calls.append(kwargs["selected_episodes"])
                return {
                    "metrics": {
                        "seed": kwargs["seed"],
                        "train_frames": len(kwargs["selected_episodes"]) * 400,
                        "val_frames": 400,
                        "test_frames": 400,
                        "test_original_mse": 0.01,
                        "test_normalized_mse": 0.2,
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
                    pc_ras_coreset=coreset_path,
                    coreset_dir=root / "coresets",
                    table_dir=root / "tables",
                    figure_dir=root / "figures",
                    prediction_dir=root / "predictions",
                    seeds=[0],
                    budget=2,
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

            random_summary = pd.read_csv(root / "tables" / "resnet18_random10_summary.csv")
            pc_summary = pd.read_csv(root / "tables" / "resnet18_pc_ras_summary.csv")
            self.assertEqual(random_summary.loc[0, "method"], "resnet18_random10")
            self.assertEqual(pc_summary.loc[0, "method"], "resnet18_pc_ras")
            self.assertIn([3, 4], calls)
            self.assertTrue((root / "figures" / "resnet18_mse_comparison.png").exists())
            self.assertTrue((root / "predictions" / "resnet18_pc_ras_seed0.npz").exists())


if __name__ == "__main__":
    unittest.main()
