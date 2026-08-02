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

## L2: Learned long-division recurrence

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that the cellular model failed because it had to discover multiplication and modular reduction simultaneously, and that exposing exact digitwise product coefficients would let a shared learned quotient-and-carry transition acquire long division from final answers alone.
The model represented residues as soft distributions over decimal digits, formed range-independent pairwise multiplicative interactions between digit positions, estimated an unsupervised quotient, formed the learned coefficient residual `x² - qN`, and decoded that residual into the next decimal residue.
One attention-and-convolution refinement block was shared between quotient estimation and residual decoding, and the entire long-division transition was tied across every requested squaring step.
Only active batch rows were processed at each recurrence step, providing serial computation proportional to the largest requested `T` without moving model state or arithmetic to the CPU.
The 107,444 persistent state elements included no whole-value embeddings, prompt keys, residue or factor tables, enumerated numeric ranges, T-specific transition operators, or solver-derived intermediate labels.
Training used final-answer cross entropy and the same device-resident AdamW with wall-clock warmup and cosine decay as L1, isolating the architectural hypothesis.

The candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L1.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 269 | 0.67% | 0.00% | 0.33% | <1 | <1 |
| E2 | 255 | 1.88% | 1.00% | 1.44% | <1 | <1 |
| E3 | 722 | 0.63% | 1.88% | 1.25% | <1 | <1 |
| E4 | 711 | 0.63% | 0.50% | 0.56% | <1 | <1 |
| E5 | 265 | 0.83% | 1.33% | 1.08% | <1 | <1 |
| M1 | 248 | 0.10% | 0.17% | 0.13% | <1 | <1 |
| M2 | 246 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M3 | 1,422 | 0.33% | 0.33% | 0.33% | <1 | <1 |
| M4 | 492 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M5 | 393 | 0.33% | 0.17% | 0.25% | <1 | <1 |

The explicit product coefficients and adaptive row selection improved optimization throughput and lowered most held-out token losses below `ln(10)`, but the extra updates did not improve exact modular arithmetic.
Relative to L1, L2 improved E1, E2, E5, and M1 while regressing E3, E4, M2, M3, M4, and M5.
The complete loss of accuracy on M2 and M4 shows that the latent quotient did not extrapolate with modulus size, and the absence of T=1 certification shows that the transition never learned a reliable single modular square.
The intuitive hypothesis is rejected, and no hosted E3 or E4 quota was consumed.
The next counterintuitive experiment should remove the unidentifiable quotient bottleneck and instead test whether target-derived final-state semantic supervision plus randomized recurrent-depth training can force a shared transition to preserve a compositional residue representation.

## L3: Hard semantic refinement recurrence

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that L2's soft residue distributions permitted average-digit shortcuts, and that a hard straight-through residue bottleneck combined with dense final-state semantics would force the tied transition to preserve compositional decimal meaning.
The model removed the unidentifiable quotient, retained range-independent digitwise multiplicative coefficients, and refined product and modulus cells directly into the next residue with one tied attention-and-convolution block.
Predicted digits were discretized between squaring steps with a straight-through estimator, while the number of physical refinement iterations was sampled independently between two and six during training and fixed at six during evaluation.
The custom loss reconstructed the official final target from its decimal label tokens and supervised the final latent state as eight decimal digits and 24 binary bits in addition to the evaluator-facing sequence loss.
These auxiliary targets were alternate representations of the final answer rather than solver-derived labels for any intermediate squaring step.
The 186,266 persistent state elements contained no whole-value embeddings, prompt keys, residue or factor tables, enumerated numeric ranges, or operators specialized to a dataset or particular `T`.

The candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L1 and L2.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 200 | 6.00% | 10.00% | 8.00% | <1 | <1 |
| E2 | 175 | 1.88% | 1.00% | 1.44% | <1 | <1 |
| E3 | 429 | 1.13% | 2.75% | 1.94% | <1 | <1 |
| E4 | 408 | 0.50% | 1.50% | 1.00% | <1 | <1 |
| E5 | 176 | 0.42% | 0.83% | 0.63% | <1 | <1 |
| M1 | 154 | 0.07% | 0.20% | 0.13% | <1 | <1 |
| M2 | 154 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M3 | 868 | 0.33% | 0.37% | 0.35% | <1 | <1 |
| M4 | 265 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 266 | 0.10% | 0.10% | 0.10% | <1 | <1 |

L3 produced the best valid local E1, E3, E4, and M4 results so far, showing that dense final-state semantics reduced the soft average-digit shortcut on some fixed-depth tasks.
The gains remained small, E2 did not improve over L2, E5 and M5 regressed, M2 remained exactly 0%, and every matched and unseen-modulus certification profile failed at T=1.
Hundreds of fixed-T updates therefore optimized useful output statistics without producing a correct single modular-square transition, while randomized physical refinement did not make the hard latent state stable under composition.
The counterintuitive hypothesis is rejected, and no hosted E3 or E4 quota was consumed.
The next intuitive experiment should replace global refinement with a shared bidirectional digit-scan transducer whose recurrent carry state explicitly follows the low-to-high multiplication and high-to-low reduction directions while retaining final-state semantic supervision.
