# One Layer Deeper Submissions

This repository tracks model experiments for the [One Layer Deeper competition](https://onelayerdeeper.ai/) and its [official GitHub repository](https://github.com/tilde-research/one-layer-deeper).
The competition evaluates a self-contained PyTorch model and optimizer under a fixed model-state limit and H100 training-time budget.

## Public datasets

Every example supplies `(N, x, T)` and asks the model to predict the decimal digits of `x^(2^T) mod N`.
The public Easy and Medium datasets isolate different combinations of fixed or sampled moduli and fixed or varying computation depths.
The ordinary OOD split holds out the listed OOD value of `T`; it is distinct from the separate `OOD N Max T` profile described below.

| Dataset | Training modulus `N` | Training `T` | Ordinary OOD `T` | Training budget |
| --- | --- | --- | ---: | ---: |
| E1 | Fixed `N=323` | 1, 2, 3 | 6 | 60 seconds |
| E2 | Fixed `N=899` | 1, 2, 4 | 7 | 60 seconds |
| E3 | Sampled 10/11-bit `N` | 2 | 4 | 60 seconds |
| E4 | Sampled 11/12-bit `N` | 2 | 4 | 60 seconds |
| E5 | Sampled 10/11-bit `N` | 1, 2, 3 | 6 | 60 seconds |
| M1 | Fixed `N=10,403` | 4, 8, 16 | 32 | 600 seconds |
| M2 | Fixed `N=38,021` | 4, 8, 16 | 32 | 600 seconds |
| M3 | Sampled 11/13/15-bit `N` | 2 | 4 | 600 seconds |
| M4 | Sampled 14/18/22-bit `N` | 8 | 16 | 600 seconds |
| M5 | Sampled 12/14/16-bit `N` | 2, 4, 8 | 16 | 600 seconds |

## E1 in detail

E1 fixes:

```text
N = 17 × 19 = 323
Training/test depths: T = 1, 2, 3
OOD depth:           T = 6
```

For each familiar depth `T ∈ {1,2,3}`, the generator creates 250 examples: 200 train and 50 test.
Across the three familiar depths, this produces 600 train and 150 test examples.
The generator separately creates 100 OOD examples at `T=6`.

| Split | Values of `T` | Examples | Purpose |
| --- | --- | ---: | --- |
| Train | 1, 2, 3 | 600 | Parameters are updated on these |
| Test | 1, 2, 3 | 150 | New prompts at familiar depths |
| OOD | 6 | 100 | New prompts at an unseen depth |

## Tokenization

The competition converts `(N,x,T)` and the target result into token IDs before calling the submitted model.
The vocabulary contains 17 tokens:

| Symbol | Token ID |
| --- | ---: |
| `[PAD]` | 0 |
| `[BOS]` | 1 |
| `[N]` | 2 |
| `[X]` | 3 |
| `[T]` | 4 |
| `[ANS]` | 5 |
| `[EOS]` | 6 |
| Decimal digit `d` | `7 + d` |

The input is tokenized in this order:

```text
[N] digits(N) [X] digits(x) [T] digits(T)
```

For example, input `(N=323, x=5, T=2)`:

```text
symbols:   [N] [3] [2] [3] [X] [5] [T] [2]
token IDs:  2  10   9  10   3  12   4   9
```

The output `302`:

```text
symbols:   [3] [0] [2]
token IDs: 10   7   9
```

`[PAD]` fills unused batch positions.
`[BOS]`, `[ANS]`, and `[EOS]` belong to an alternative combined causal representation and are not used in the public Easy and Medium input and output tensors.

## Architecture notes

A model intended to learn the repeated computation needs four functional parts:

1. An encoder maps the tokenized inputs `(x,N,T)` into an initial learned working state.
2. The working state carries a representation of the current residue, the modulus, the requested depth, and recurrence progress.
3. A tied recurrent transition updates that state from its previous value using the same learned parameters at every step.
4. A decoder maps the final state to the decimal digits of `x_T`.

The high-level computation is:

```text
(x, N, T) → Encoder → z₀ → fθ → z₁ → fθ → … → zₖ → Decoder → x_T
```

Equivalently:

```text
z₀ = Encoder(x, N, T)
zₖ₊₁ = fθ(zₖ)
prediction = Decoder(z_final)
```

The unified state `zₖ` must retain `T`, or a representation derived from `T`, so the model knows the requested computation depth.
It also needs progress information such as the current iteration, remaining depth, an update mask, or a learned halting signal.
Keeping `N` and `T` as immutable context beside the evolving residue state is equivalent to storing them inside one unified state.

The recurrence should be tied, meaning every step reuses the same transition `fθ`.
An untied stack with different parameters at every layer can learn separate mappings for familiar depths without learning a transition that extrapolates to larger `T`.

## Depth certification

`Max T` is a certification threshold over the fixed ladder `T = 1, 2, 4, 8, 16, 32, 64`.
For each rung, the evaluator generates fresh prompts using familiar modulus identities or problem families.
A rung passes only when every example at that rung is exactly correct, and certification must be a consecutive prefix starting at `T=1`.
For example, perfect results at `T=1` and `T=2` followed by any error at `T=4` produce `Max T = 2`; a later perfect `T=8` result cannot raise it.
`Max T <1` means the model failed to answer every `T=1` certification example correctly.

`OOD N Max T` applies the same ladder and consecutive-prefix rule to fresh, unseen modulus identities at nearby dataset-scale bit sizes.
It does not mean the largest numerical value of `N`, nor does `Max T` mean the largest `T` example present in a dataset.
Easy and Medium report both values as diagnostics, but their scores remain the mean ordinary exact accuracy.
The Hard leaderboard ranks submissions lexicographically by `Max T`, then `OOD N Max T`, and then earlier submission time.

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
uv run python -c "import duckdb; duckdb.connect('submissions.duckdb').sql('FROM submissions').show()"
```

## Submission history

<!-- SUBMISSIONS_TABLE_START -->
| Number | Note | Commit | Status | Score |
| ---: | --- | --- | --- | ---: |
| 1 | official test submission file | [`a66155a`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/a66155afa3596d7c43ed2813d45299d4f833d173) | succeeded | 3.83% |
| 2 | gpt-2 small | [`655c650`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/655c650bfcd5a460c9f18089bb58e91ea6622acd) | succeeded | 1.33% |
| 3 | 4-layer width-256, warmup+cosine, dropout+label smoothing | [`6630a45`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/6630a4507c8d9a7bb33d7034210bd96e50136ce2) | succeeded | 4.33% |
| 4 | 2-layer width-128, 80 steps, warmup+cosine, dropout+label smoothing | [`4ac12ae`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/4ac12ae68373c55ed3dd30b827fb535688ed9276) | succeeded | 5.17% |
| 5 | tied second block repeated exactly T, 80 steps | [`2340c1c`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/2340c1c6224c992c0fc91177186b670ecb7aab2d) | failed | — |
| 6 | tied second block repeated exactly T, 80 steps | [`6c91e99`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/6c91e99a671d9edaba0571bfe2d52c516157f999) | succeeded | 4.33% |
| 7 | T2MLR gated middle cache, 80 steps | [`47646db`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/47646db668748230211074d0463b75aa2f91c133) | succeeded | 4.17% |
<!-- SUBMISSIONS_TABLE_END -->
