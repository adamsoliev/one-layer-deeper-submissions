# One Layer Deeper Submissions

This repository tracks model experiments for the [One Layer Deeper competition](https://onelayerdeeper.ai/) and its [official GitHub repository](https://github.com/tilde-research/one-layer-deeper).
The competition evaluates a self-contained PyTorch model and optimizer under a fixed model-state limit and H100 training-time budget.
Every submission must strictly follow the [official competition rules](https://github.com/tilde-research/one-layer-deeper#rules), which are authoritative.

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
  --architecture gpt2-small \
  --note "gpt-2 small" \
  --tier easy \
  --dataset e1
```

The command requires a clean Git worktree so the recorded commit identifies the exact submitted source.
It runs `one-layer submit submission.py --wait`, streams the remote status, writes one row to `submissions.duckdb`, and regenerates the table below.
Commit the resulting database and README changes after each run.

The DuckDB database is the source of truth.
The `architectures` table defines stable architecture identities and canonical source commits, while each `submissions` row stores one dataset run with its experiment number, note, source hash, validation and queue state, hosted IDs, final status and score, depth metrics, command exit code, and complete CLI output.
The table below shows the latest recorded E1 and E5 result for each architecture.
Inspect it with:

```bash
uv run python -c "import duckdb; duckdb.connect('submissions.duckdb').sql('FROM submissions').show()"
```

## Architecture results

<!-- SUBMISSIONS_TABLE_START -->
| Architecture | Source | E1 | E5 |
| --- | --- | ---: | ---: |
| 1-layer width-64 Transformer | [`a66155a`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/a66155afa3596d7c43ed2813d45299d4f833d173) | [3.83%](https://onelayerdeeper.ai/submissions/27071dce-9e87-4a51-800e-12c4b16b6f5b) | [0.33%](https://onelayerdeeper.ai/submissions/78524f0b-3f59-4ba3-abe7-1ac4613a205d) |
| 12-layer width-768 Transformer | [`655c650`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/655c650bfcd5a460c9f18089bb58e91ea6622acd) | [1.33%](https://onelayerdeeper.ai/submissions/092923b6-28f2-47ee-9d18-ea7b8c156854) | [0.42%](https://onelayerdeeper.ai/submissions/1f2e6688-dcf5-4278-b7cd-e0a132da4443) |
| 4-layer width-256 Transformer | [`6630a45`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/6630a4507c8d9a7bb33d7034210bd96e50136ce2) | [4.33%](https://onelayerdeeper.ai/submissions/cb2a48ac-a331-4d6e-8612-72d1eb15f0ac) | [0.33%](https://onelayerdeeper.ai/submissions/e7e1263f-ef36-4503-add4-7dffe7b7d44a) |
| 2-layer width-128 Transformer | [`4ac12ae`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/4ac12ae68373c55ed3dd30b827fb535688ed9276) | [5.17%](https://onelayerdeeper.ai/submissions/7362f0f8-9b9c-41a5-8d34-2cb3550ccd09) | [0.71%](https://onelayerdeeper.ai/submissions/d66b4c71-334c-4421-9a4d-aaa3bc75f717) |
| T-step tied width-128 Transformer | [`6c91e99`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/6c91e99a671d9edaba0571bfe2d52c516157f999) | [4.33%](https://onelayerdeeper.ai/submissions/e5bc8c6c-5d66-4a71-982c-3479de825133) | [0.71%](https://onelayerdeeper.ai/submissions/69d7c10c-0b6b-4978-9db6-193db4759931) |
| T²MLR gated-cache Transformer | [`47646db`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/47646db668748230211074d0463b75aa2f91c133) | [4.17%](https://onelayerdeeper.ai/submissions/a6eb3958-62c8-401e-84d2-87ca895b23ef) | [0.75%](https://onelayerdeeper.ai/submissions/6ac5eac6-7540-474d-95eb-8feafd32f78a) |
| Soft Thoughtbubbles Transformer | [`a408a7d`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/a408a7dd27ae4648571102948be2f098429dd8aa) | [5.50%](https://onelayerdeeper.ai/submissions/ce357172-8f06-4be9-a224-e482f8bd0eee) | — |
| Single-layer width-128 bidirectional RNN | [`77788bd`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/77788bde51bff99006d3512cfad34e3f62a3856f) | [6.33%](https://onelayerdeeper.ai/submissions/de736b5d-8d32-45a3-ae56-3a6e0899e0c7) | — |
| Hierarchical T-step GRU | [`d0d326f`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/d0d326fc4aee81c9c9eb6d4b9298db3aabb2d7b7) | [6.00%](https://onelayerdeeper.ai/submissions/01694dfb-5660-4045-bc55-882e51a5bb57) | — |
| Explicit residue-state Transformer | [`628703b`](https://github.com/adamsoliev/one-layer-deeper-submissions/commit/628703bfd11f748a66e65d0dd88f30178b7348c4) | [7.83%](https://onelayerdeeper.ai/submissions/ab93ce56-b9bf-49e8-a0a1-42af6d15224c) | [0.33%](https://onelayerdeeper.ai/submissions/db9928d7-bee6-48c7-9d6d-ee99d44232e1) |
<!-- SUBMISSIONS_TABLE_END -->
