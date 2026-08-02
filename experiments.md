# Local Experiments

## L1: Worst-digit neural cellular automaton

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that a small range-independent binary cellular automaton could learn repeated modular squaring more reliably than a large keyed memory if the same local/global transition were reused for every squaring step and the loss emphasized the weakest output digit.
The model represented `x` and `N` on a 24-cell bit grid, used one tied attention-and-convolution refinement block, composed one learned squaring transition according to `T`, decoded decimal positions with learned queries, and contained 208,851 persistent state elements.
The optimizer was a device-resident AdamW implementation with decay excluded from normalization and query parameters, controlled by a wall-clock warmup and cosine schedule.
The objective combined whole-sequence negative log likelihood, a smooth maximum over per-digit losses, and input-bit reconstruction.
The architecture contained no whole-value embeddings, `(N,x,T)` keys, factor candidates, residue tables, solver-derived labels, or branches specialized to a dataset or a particular `T`.

The official public datasets were generated without inspecting their records, and the frozen candidate was screened with the official evaluator on an Apple M2 Pro using 30 seconds for each Easy dataset and 60 seconds for each Medium dataset.
These local time budgets are a rejection screen rather than estimates of hosted H100 scores, but every official train, test, ordinary OOD, matched-depth, and unseen-modulus depth split was retained.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 155 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E2 | 168 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E3 | 166 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 167 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 167 | 0.92% | 0.17% | 0.54% | <1 | <1 |
| M1 | 331 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 322 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 320 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 317 | 0.02% | 0.00% | 0.01% | <1 | <1 |
| M5 | 321 | 0.42% | 0.33% | 0.38% | <1 | <1 |

Held-out token losses remained close to the uniform ten-digit baseline of `ln(10)`, while exact accuracy stayed near the accidental whole-value rate and did not improve consistently with depth.
The tied latent transition therefore learned marginal digit statistics rather than multiplication, carry propagation, or modular reduction.
The counterintuitive hypothesis is rejected, and no hosted E3 or E4 quota was consumed.
The next intuitive experiment should expose the fundamental computation more directly through shared digitwise multiplicative interactions and learned carry/reduction state while preserving end-to-end supervision and range-independent parameters.
