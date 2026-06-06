import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "13_run_budget_curve.py"
    spec = importlib.util.spec_from_file_location("run_budget_curve", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BudgetCurveScriptTest(unittest.TestCase):
    def test_main_writes_summary_for_random_and_pc_ras_budgets(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_path = root / "split_v1.json"
            split_path.write_text(
                json.dumps({"candidate_train": list(range(10)), "val": [10], "test": [11]}),
                encoding="utf-8",
            )
            frame_score_path = root / "scores.parquet"
            pd.DataFrame(
                {
                    "episode_index": np.repeat(np.arange(10), 2),
                    "frame_score": np.linspace(0.1, 1.0, 20),
                    "pc_score": np.linspace(0.1, 1.0, 20),
                    "ras_score": np.linspace(0.1, 1.0, 20),
                    "action_delta": 0.1,
                    "visual_delta": 0.2,
                    "phase_bin": 0,
                }
            ).to_parquet(frame_score_path)
            fake_index = pd.DataFrame({"split": ["candidate_train", "val", "test"], "episode_index": [0, 10, 11]})

            calls = []

            def fake_train_mlp_baseline(**kwargs):
                calls.append((kwargs["selected_episodes"], kwargs["seed"]))
                return {
                    "metrics": {
                        "seed": kwargs["seed"],
                        "train_frames": len(kwargs["selected_episodes"]) * 400,
                        "val_frames": 400,
                        "test_frames": 400,
                        "test_original_mse": 0.01 / len(kwargs["selected_episodes"]),
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
                "build_episode_embeddings",
                return_value=pd.DataFrame(np.eye(10), index=list(range(10))),
            ), mock.patch.object(
                module,
                "train_mlp_baseline",
                side_effect=fake_train_mlp_baseline,
            ), mock.patch.object(
                module,
                "parse_args",
                return_value=module.argparse.Namespace(
                    feature_dir=root / "features",
                    split_file=split_path,
                    frame_score_file=frame_score_path,
                    coreset_dir=root / "coresets",
                    table_dir=root / "tables",
                    figure_dir=root / "figures",
                    prediction_dir=root / "predictions",
                    budgets=[10, 20],
                    seeds=[0],
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

            summary = pd.read_csv(root / "tables" / "budget_curve_summary.csv")
            self.assertEqual(set(summary["method"]), {"Random", "PC-RAS"})
            self.assertEqual(set(summary["budget_percent"]), {10, 20})
            self.assertEqual(len(calls), 4)
            self.assertTrue((root / "figures" / "budget_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
