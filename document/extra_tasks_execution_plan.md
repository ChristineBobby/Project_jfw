# Extra Tasks Execution Plan

Goal: finish the remaining recommended and optional experiments before writing the final report.

Execution principles:

- Run project code inside Docker.
- Use `nohup` for long-running experiments.
- Keep logs under `experiments/logs/`.
- Commit only lightweight scripts, tests, tables, figures, and JSON metadata.
- Keep `data/cache/`, `data/features/`, `results/predictions/`, and `experiments/logs/` out of Git.

## Batch 1: CLIP Upper Bound And Analysis Artifacts

- Train the CLIP MLP on all `candidate_train` episodes and write `results/tables/full_clip_summary.csv`.
- Produce selected-episode explanations in `results/tables/pc_ras_selected_episode_explanations.csv`.
- Produce score timeline figures for selected episodes `36`, `20`, `6`, and a low-score comparison episode.
- Produce selected episode action curves comparing Random-10% and PC-RAS.
- Produce phase-aware MSE tables and figure from existing CLIP baseline predictions.
- Produce key-frame visualization tiles for selected high-score frames.

## Batch 2: Budget Curve

- Evaluate PC-RAS and Random episode budgets corresponding to 2%, 5%, 10%, and 20% of candidate episodes.
- Use the same CLIP MLP hyperparameters and seeds as the main baselines.
- Write `results/tables/budget_curve_summary.csv` and `results/figures/budget_curve.{png,pdf}`.

## Batch 3: ResNet18 Robustness

- Extract frozen ResNet18 features to `data/features/resnet18_top_left7/`.
- Run Random-10% and PC-RAS baselines with the same MLP training protocol.
- Write `results/tables/resnet18_random10_summary.csv`, `results/tables/resnet18_pc_ras_summary.csv`, and a compact comparison figure.

## Final

- Run the full test suite in Docker.
- Update `document/remaining_tasks.md`.
- Commit and push lightweight outputs.
- Write the final report after all experiment outputs are present.
