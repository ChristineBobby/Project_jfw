#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vla_coreset.visualization.report_figures import make_ablation_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Step 8 ablation comparison figure.")
    parser.add_argument("--ablation-summary", type=Path, default=Path("results/tables/ablation_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = make_ablation_figure(pd.read_csv(args.ablation_summary), args.output_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
