import tempfile
import unittest
from pathlib import Path

import pandas as pd

from vla_coreset.visualization.report_figures import (
    FigureInputs,
    build_method_summary,
    make_report_figures,
    per_dimension_mse_table,
)


class ReportFiguresTest(unittest.TestCase):
    def test_build_method_summary_reports_mean_std_and_improvement(self):
        random_df = pd.DataFrame({"test_original_mse": [0.10, 0.20], "test_normalized_mse": [1.0, 2.0]})
        pc_df = pd.DataFrame({"test_original_mse": [0.05, 0.15], "test_normalized_mse": [0.5, 1.5]})

        summary = build_method_summary(random_df, pc_df)

        self.assertEqual(summary["method"].tolist(), ["Random-10%", "PC-RAS"])
        self.assertAlmostEqual(summary.loc[0, "test_original_mse_mean"], 0.15)
        self.assertAlmostEqual(summary.loc[1, "test_original_mse_mean"], 0.10)
        self.assertAlmostEqual(summary.attrs["relative_improvement"], (0.15 - 0.10) / 0.15)

    def test_per_dimension_mse_table_averages_action_columns(self):
        random_df = pd.DataFrame({f"test_original_action_{i}_mse": [float(i + 1), float(i + 3)] for i in range(7)})
        pc_df = pd.DataFrame({f"test_original_action_{i}_mse": [0.5 * (i + 1), 0.5 * (i + 3)] for i in range(7)})

        table = per_dimension_mse_table(random_df, pc_df)

        self.assertEqual(table.loc["Random-10%", "a0"], 2.0)
        self.assertEqual(table.loc["Random-10%", "a1"], 3.0)
        self.assertEqual(table.loc["PC-RAS", "a0"], 1.0)
        self.assertEqual(table.loc["PC-RAS", "a1"], 1.5)

    def test_make_report_figures_writes_png_and_pdf_for_main_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            random_df = pd.DataFrame(
                {
                    "seed": [0, 1],
                    "test_original_mse": [0.10, 0.20],
                    "test_normalized_mse": [1.0, 2.0],
                    **{f"test_original_action_{i}_mse": [0.01 * (i + 1), 0.02 * (i + 1)] for i in range(7)},
                }
            )
            pc_df = pd.DataFrame(
                {
                    "seed": [0, 1],
                    "test_original_mse": [0.08, 0.12],
                    "test_normalized_mse": [0.8, 1.2],
                    **{f"test_original_action_{i}_mse": [0.008 * (i + 1), 0.012 * (i + 1)] for i in range(7)},
                }
            )
            episode_scores = pd.DataFrame(
                {
                    "rank": list(range(1, 7)),
                    "episode_index": [1, 2, 3, 4, 5, 6],
                    "episode_score": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                }
            )
            selection_trace = pd.DataFrame(
                {
                    "selection_order": [1, 2, 3],
                    "episode_index": [1, 4, 6],
                    "final_score": [1.0, 0.8, 0.7],
                    "saliency_score": [1.0, 0.6, 0.5],
                    "diversity_score": [1.0, 0.9, 0.8],
                }
            )
            embeddings = pd.DataFrame(
                {
                    "episode_index": [1, 2, 3, 4, 5, 6],
                    "x": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
                    "y": [0.0, 0.2, 0.1, 1.0, 0.9, 1.1],
                    "group": ["Candidate", "Candidate", "Random", "PC-RAS", "Random", "PC-RAS"],
                }
            )
            inputs = FigureInputs(
                random_summary=random_df,
                pc_ras_summary=pc_df,
                episode_scores=episode_scores,
                selection_trace=selection_trace,
                coverage_points=embeddings,
                output_dir=root,
            )

            written = make_report_figures(inputs)

            self.assertEqual(len(written), 10)
            for path in written:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 1000, path)


if __name__ == "__main__":
    unittest.main()
