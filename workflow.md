# Project Workflow

This file records the working conventions for `Project_jfw`. Keep it updated as the project workflow becomes clearer.

## Git And GitHub

- Do not push to GitHub from inside the Docker container.
- Use Docker for project environment work: dataset inspection, feature extraction, model training, evaluation, and tests.
- Use a separate host terminal for GitHub authentication and `git push`.
- The project Git repository lives at:

```text
/home/xiaobo.xia/JiafengWu/code_folder/area1/Project_jfw
```

- The GitHub remote is:

```text
https://github.com/ChristineBobby/Project_jfw.git
```

- Before pushing, verify the working tree and latest commit from the host terminal:

```bash
cd /home/xiaobo.xia/JiafengWu/code_folder/area1/Project_jfw
git status --short --branch
git log --oneline --decorate -1
```

- Push from the host terminal only:

```bash
git push -u origin main
```

## Docker Usage

- Use the Docker container for reproducible Python execution.
- The project path inside the container is:

```text
/workspace/code_folder/area1/Project_jfw
```

- Run project scripts from the project root with `PYTHONPATH=src`.

Example:

```bash
cd /workspace/code_folder/area1/Project_jfw
PYTHONPATH=src /workspace/.conda/envs/coredataset/bin/python scripts/01_inspect_dataset.py
```

## Long-Running Jobs

- Long-running jobs should be launched in a way that survives terminal disconnects.
- Prefer `nohup` for single-command jobs unless an interactive terminal is specifically needed.
- Every long job must write a log file under `experiments/logs/` or another explicit log directory.
- Every long job should expose progress with `tqdm` or clear periodic logging.
- Before starting a long job, report:
  - host/container context
  - working directory
  - exact command
  - log file path
  - PID or terminal/session identifier
  - GPU selection, if applicable
- After starting a long job, report how to monitor it.

Example pattern:

```bash
mkdir -p experiments/logs
nohup bash -lc 'cd /workspace/code_folder/area1/Project_jfw && PYTHONPATH=src /workspace/.conda/envs/coredataset/bin/python scripts/<script>.py --arg value' \
  > experiments/logs/<job-name>.log 2>&1 &
echo $!
tail -f experiments/logs/<job-name>.log
```

- For GPU jobs, prefer using the available 4090 capacity effectively instead of conservative defaults.
- Increase batch size after a small smoke test passes, while watching memory and utilization.
- Check GPU status with:

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

## Data And Artifacts

- Keep large dataset caches out of Git.
- `data/cache/` is local-only and ignored by `.gitignore`.
- Lightweight reproducibility outputs such as Step 1 summary tables and the sample figure may be committed.

## Verification

- Before committing project code, run the relevant tests inside Docker:

```bash
cd /workspace/code_folder/area1/Project_jfw
PYTHONPATH=src /workspace/.conda/envs/coredataset/bin/python -m unittest discover -s tests -p "test_*.py"
```
