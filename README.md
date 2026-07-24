# One Layer Deeper Submissions

This repository tracks model experiments for the [One Layer Deeper competition](https://onelayerdeeper.ai/) and its [official GitHub repository](https://github.com/tilde-research/one-layer-deeper).
The competition evaluates a self-contained PyTorch model and optimizer under a fixed model-state limit and H100 training-time budget.

## Submission workflow

Install the competition CLI, authenticate once, and install this project's dependencies:

```bash
uv tool install git+https://github.com/tilde-research/one-layer-deeper.git
one-layer login
uv sync
```

Commit the exact `submission.py` that should be evaluated, then submit it and record the complete result with one command:

```bash
uv run scripts/submit_and_record.py \
  --note "gpt-2 small" \
  --tier easy \
  --dataset e1
```

The command requires a clean Git worktree so the recorded commit identifies the exact submitted source.
It runs `one-layer submit submission.py --wait`, streams the remote status, writes one row to `submissions.duckdb`, and regenerates the table below.
Commit the resulting database and README changes after each run.

The DuckDB `submissions` table is the source of truth.
Each row stores the experiment number, note, source commit, source hash, validation and queue state, tier, dataset, remaining daily attempts, hosted IDs, final status and score, depth metrics, command exit code, and complete CLI output.
Inspect it with:

```bash
uv run python -c "import duckdb; duckdb.sql(\"FROM 'submissions.duckdb'\").show()"
```

## Submission history

<!-- SUBMISSIONS_TABLE_START -->
| Number | Note | Commit | Status | Score |
| ---: | --- | --- | --- | ---: |
| 1 | official test submission file | [`a66155a`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/a66155afa3596d7c43ed2813d45299d4f833d173) | succeeded | 3.83% |
<!-- SUBMISSIONS_TABLE_END -->
