import json
import tempfile
import unittest
from pathlib import Path

from vla_coreset.data.split import build_split_v1, validate_split, write_split


class SplitTest(unittest.TestCase):
    def test_build_split_v1_matches_project_protocol(self):
        split = build_split_v1()

        self.assertEqual(split["candidate_train"], list(range(40)))
        self.assertEqual(split["val"], list(range(40, 45)))
        self.assertEqual(split["test"], list(range(45, 50)))
        self.assertEqual(split["unit"], "episode")
        self.assertEqual(split["dataset"], "lerobot/aloha_sim_transfer_cube_human")

    def test_validate_split_rejects_overlap_or_missing_episodes(self):
        valid = build_split_v1()
        validate_split(valid, total_episodes=50)

        overlapping = dict(valid)
        overlapping["val"] = [39, 40, 41, 42, 43]
        with self.assertRaises(ValueError):
            validate_split(overlapping, total_episodes=50)

        missing = dict(valid)
        missing["test"] = [45, 46, 47, 48]
        with self.assertRaises(ValueError):
            validate_split(missing, total_episodes=50)

    def test_write_split_creates_stable_json(self):
        split = build_split_v1()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split_v1.json"

            write_split(path, split)

            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, split)
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
