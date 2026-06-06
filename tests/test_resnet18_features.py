import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vla_coreset.features.extract_resnet18 import save_resnet_feature_artifacts


class ResNet18FeaturesTest(unittest.TestCase):
    def test_save_resnet_feature_artifacts_matches_training_loader_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            features = np.zeros((4, 512), dtype=np.float32)
            actions = np.ones((4, 7), dtype=np.float32)
            text_features = np.zeros((1, 512), dtype=np.float32)
            index = pd.DataFrame(
                {
                    "row_index": [0, 1, 2, 3],
                    "episode_index": [0, 0, 1, 1],
                    "frame_index": [0, 1, 0, 1],
                    "timestamp": [0.0, 0.02, 0.0, 0.02],
                    "split": ["candidate_train", "candidate_train", "test", "test"],
                }
            )

            save_resnet_feature_artifacts(output_dir, features, actions, text_features, index)

            self.assertEqual(np.load(output_dir / "features.npy").shape, (4, 512))
            self.assertEqual(np.load(output_dir / "actions_left7.npy").shape, (4, 7))
            self.assertEqual(np.load(output_dir / "text_features.npy").shape, (1, 512))
            self.assertEqual(pd.read_parquet(output_dir / "index.parquet").shape[0], 4)


if __name__ == "__main__":
    unittest.main()
