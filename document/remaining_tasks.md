# Remaining Project Tasks

This checklist records what is still useful or required after the current PC-RAS CLIP baseline and main figure package. It is meant as a working memory aid, not a new specification.

## Current Verified State

- [x] Step 1 dataset inspection outputs exist:
  - `results/tables/dataset_summary.csv`
  - `results/tables/dataset_summary.json`
  - `results/tables/action_ranges.csv`
  - `results/figures/dataset_samples.png`
- [x] Step 2 fixed episode split exists:
  - `data/splits/split_v1.json`
  - candidate train episodes `0..39`
  - validation episodes `40..44`
  - test episodes `45..49`
- [x] Step 3 CLIP features exist locally:
  - `data/features/clip_vit_b32_top_left7/features.npy`
  - `data/features/clip_vit_b32_top_left7/actions_left7.npy`
  - `data/features/clip_vit_b32_top_left7/text_features.npy`
  - `data/features/clip_vit_b32_top_left7/index.parquet`
- [x] Step 4 Random-10% CLIP baseline is complete:
  - `data/coresets/random_episode_seed0.json` through `random_episode_seed4.json`
  - `results/tables/random10_clip_seed*_metrics.csv`
  - `results/tables/random10_clip_summary.csv`
  - current test original MSE mean: `0.006564`
- [x] Step 5 PC-RAS frame and episode scoring is complete:
  - `data/cache/selection/pc_ras_frame_scores.parquet`
  - `results/tables/pc_ras_frame_scores_summary.csv`
  - `results/tables/pc_ras_episode_scores.csv`
- [x] Step 6 PC-RAS episode coreset selection is complete:
  - `data/coresets/pc_ras_episode_top5.json`
  - `results/tables/pc_ras_episode_selection_trace.csv`
  - selected episodes: `[36, 20, 6, 30, 18]`
- [x] Step 7 PC-RAS CLIP MLP baseline is complete:
  - `results/tables/pc_ras_clip_seed*_metrics.csv`
  - `results/tables/pc_ras_clip_summary.csv`
  - current test original MSE mean: `0.005854`
- [x] Step 9 main figure package exists:
  - `results/figures/main_mse_comparison.{png,pdf}`
  - `results/figures/per_dimension_mse_heatmap.{png,pdf}`
  - `results/figures/episode_score_ranking.{png,pdf}`
  - `results/figures/selection_trace.{png,pdf}`
  - `results/figures/coverage_pca.{png,pdf}`
- [x] Latest fresh test run passed in Docker:
  - `PYTHONPATH=src /workspace/.conda/envs/coredataset/bin/python -m unittest discover -s tests -p "test_*.py"`
  - result: `Ran 42 tests ... OK`

## Must Do Before Final Report

- [x] Step 8 ablation experiments.
  - [x] Implement selector variants:
    - `ActionDelta-only`
    - `VisualDelta-only`
    - `PC-only`
    - `PC+RAS`
    - `Coverage-only`
    - `PC+RAS+Coverage` / full method
  - [x] Save selected coreset JSON for each variant under `data/coresets/`.
  - [x] Train the same CLIP MLP for each variant with consistent seeds and hyperparameters.
  - [x] Save metrics under `results/tables/`.
  - [x] Produce `results/tables/ablation_summary.csv`.
  - [x] Add focused tests for selector variants and summary aggregation.
  - [x] Run ablations with `nohup` and logs under `experiments/logs/`.

- [x] Add an ablation figure after Step 8.
  - [x] Produce `results/figures/ablation_mse_comparison.{png,pdf}`.
  - [x] Keep the same paper style as Step 9 figures.
  - [x] Visually inspect for label, legend, and error-bar overlap.

- [ ] Write the final report.
  - [ ] Use `document/topic2_vla_coreset_project_guide.md` as the main source.
  - [ ] Include dataset/task definition and fixed split.
  - [ ] Include Random-10% and PC-RAS results with mean/std over seeds.
  - [ ] Include ablation results once Step 8 is complete.
  - [ ] Include limitations: one task, constant language embedding, small dataset, MSE is not closed-loop success.
  - [ ] Include source-code and reproducibility notes.

## Strongly Recommended

- [x] Full-data upper bound.
  - [x] Train CLIP MLP on all `candidate_train` episodes `0..39`.
  - [x] Save `results/tables/full_clip_summary.csv`.
  - [x] Use it as an upper-bound reference, not as the main comparison.

- [x] Selected episode explanation table.
  - [x] Produce `results/tables/pc_ras_selected_episode_explanations.csv`.
  - [x] Include:
    - episode index
    - episode saliency
    - top frame indices
    - phase coverage
    - mean visual delta
    - mean action delta
    - mean gripper delta
    - diversity distance or selection trace score
    - short selected reason

- [x] Score timeline visualizations.
  - [x] Produce representative `results/figures/score_timeline_episode_*.{png,pdf}`.
  - [x] Prefer a small set such as selected episodes `36`, `20`, `6`, plus one low-score episode.
  - [x] Avoid dense labels; use shaded phase bins and minimal annotations.

- [x] Action curve comparison figure.
  - [x] Compare Random-10% selected episodes and PC-RAS selected episodes.
  - [x] Plot only selected action dimensions or small multiples to avoid clutter.
  - [x] Save `results/figures/selected_episode_action_curves.{png,pdf}`.

## Optional Enhancements

- [x] Phase-aware MSE.
  - [x] Define phase bins over `frame_index / episode_length`.
  - [x] Compute phase-wise MSE for Random-10% and PC-RAS predictions.
  - [x] Save `results/tables/phase_mse_summary.csv`.
  - [x] Save `results/figures/phase_mse_bar_chart.{png,pdf}`.

- [x] Key-frame visualization.
  - [x] Extract selected high-score frames from raw data/video.
  - [x] Annotate PC score, RAS score, phase, and reason.
  - [x] Keep annotations outside image tiles or in a consistent caption band to avoid occlusion.

- [x] ResNet18 robustness experiment.
  - [x] Extract ResNet18 features under `data/features/resnet18_top_left7/`.
  - [x] Run Random-10% and PC-RAS with the same MLP.
  - [x] Report whether the selection conclusion depends on CLIP.

- [x] Budget curve.
  - [x] Evaluate 2%, 5%, 10%, and 20% episode or frame budgets.
  - [x] Save `results/tables/budget_curve_summary.csv`.
  - [x] Save `results/figures/budget_curve.{png,pdf}`.

## Reproducibility And Workflow Reminders

- [ ] Run project code inside Docker, not directly on the host.
- [ ] Push to GitHub from the host, not from inside Docker.
- [ ] Use `PYTHONPATH=src` for scripts and tests.
- [ ] Use `nohup` for long-running jobs and write logs under `experiments/logs/`.
- [ ] Report command, cwd, container ID, PID, log file, and GPU selection before long jobs.
- [ ] Keep large local artifacts out of Git:
  - `data/cache/`
  - `data/features/`
  - `results/predictions/`
  - `experiments/logs/`
- [ ] Commit lightweight reproducibility artifacts:
  - selected coreset JSON
  - summary CSV tables
  - final report figures
  - scripts and tests
