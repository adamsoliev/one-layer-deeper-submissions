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

## L4: Bidirectional digit-scan recurrence

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that L3 failed because attention over unordered coefficient cells did not encode the opposing traversal directions needed for arithmetic: multiplication carries move from low to high digits, while reduction decisions depend on high-to-low comparisons.
The model formed exact pairwise decimal product coefficients from soft residue digits, passed them through a shared low-to-high GRU carry scan, passed the reversed sequence through a shared high-to-low GRU reduction scan, and fused the two streams into the next soft residue.
The same transition was tied across every requested squaring step, and only active batch rows were evaluated at each step, so computation scaled with requested depth while persistent state remained independent of the largest input integer and `T`.
Training retained L3's final-answer decimal and binary semantic losses, but removed the straight-through discretization and randomized physical refinement depth to isolate the effect of explicit scan direction.
The 145,882 persistent state elements contained no whole-value embeddings, prompt keys, residue or factor tables, enumerated numeric ranges, dataset branches, T-specific operators, or solver-derived intermediate labels.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L1 through L3.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 232 | 6.00% | 10.00% | 8.00% | <1 | <1 |
| E2 | 201 | 1.88% | 1.00% | 1.44% | <1 | <1 |
| E3 | 699 | 0.50% | 2.25% | 1.38% | <1 | <1 |
| E4 | 683 | 0.06% | 0.08% | 0.07% | <1 | <1 |
| E5 | 222 | 0.58% | 0.00% | 0.29% | <1 | <1 |
| M1 | 185 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 187 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 1,386 | 0.46% | 0.40% | 0.43% | <1 | <1 |
| M4 | 465 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 287 | 0.17% | 0.03% | 0.10% | <1 | <1 |

The directional recurrence matched L3 on E1, E2, M1, M4, and M5 within sampling resolution, modestly recovered M2 and M3, and regressed sharply on the target tasks: E3 fell from 1.94% to 1.38% and E4 from 1.00% to 0.07%.
Every depth profile again failed to certify even T=1, so exposing traversal order increased update throughput without teaching a correct modular-square transition.
The intuitive hypothesis is rejected, no hosted E3 or E4 quota was consumed, and neither the hosted-only DuckDB records nor the README Architecture results table was changed.
The next counterintuitive experiment should test a redundant signed-digit latent state, whose local carry-free product representation may be easier to compose than canonical decimal digits, while retaining final-answer-only supervision and a range-independent tied transition.

## L5: Redundant-digit invariant recurrence

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that canonical decimal states made modular reduction unnecessarily difficult because subtraction requires coordinated borrows, whereas redundant signed coefficients permit value-preserving subtraction before a learned carry pass restores canonical digits.
Each tied square used eight high-to-low decimal Horner steps, reducing the learned quotient at every step from an unbounded whole-square value to one of the nineteen values from 0 through 18.
The candidate subtracted the selected multiple of the modulus in a redundant signed-digit workspace, used a shared low-to-high GRU to choose carries and canonical output digits, and composed the same square transition once per requested `T` step.
Straight-through categorical decisions preserved a gradient through quotient, carry, and digit expectations.
The custom loss combined final sequence and fixed-width decimal supervision with label-free invariants requiring each reduced value to lie in `[0,N)`, each normalized coefficient to lie in `[0,9]`, the final carry to vanish, and the selected canonical digits to preserve the reduced value.
These constraints were algebraic properties rather than solver-derived intermediate labels.
A wall-clock curriculum first exposed one Horner step, progressively exposed all eight, and enabled the complete requested squaring depth and final-answer loss after 60% of the budget; this prevented random early carries from recursively producing non-finite coefficients.
The recurrent state itself was chosen from learned decimal digit logits, so failed internal guesses remained bounded without clamping or excluding any input example.
The 23,389 persistent state elements contained no whole-value or prompt-key memories, residue or factor tables, enumerated ranges, dataset branches, T-specific operators, or parameters that scaled with the largest representable integer.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as the preceding experiments.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 550 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E2 | 524 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E3 | 613 | 0.00% | 0.50% | 0.25% | <1 | <1 |
| E4 | 602 | 0.12% | 0.08% | 0.10% | <1 | <1 |
| E5 | 527 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M1 | 1,045 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M2 | failed at update 1,001 | — | — | — | — | — |
| M3 | 1,292 | 0.08% | 0.07% | 0.07% | <1 | <1 |
| M4 | 1,070 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M5 | 1,102 | 0.00% | 0.00% | 0.00% | <1 | <1 |

The curriculum made the implementation numerically usable on nine datasets, and several invariant losses fell by orders of magnitude, but exact accuracy became worse than the simpler L1 through L4 models.
Low invariant loss therefore did not identify the correct discrete quotient, carry, and digit decisions: the model found soft constraint-satisfying behavior whose argmax computation was not an arithmetic transducer.
The longer Medium budgets did not rescue the representation, and M2 ended with a non-finite training loss at update 1,001, so L5 also failed the mandatory full-coverage gate.
The counterintuitive hypothesis is rejected, no hosted quota was consumed, and the hosted-only DuckDB and README Architecture results remain unchanged.
The next intuitive experiment should use a fully bounded binary state with one tied local gate network for multiplication, comparison, and conditional subtraction, avoiding both unbounded quotient classification and indirectly learned decimal canonicalization while keeping every bit decision learned end to end.

## L6: Bounded radix-gate recurrence

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that L5 failed because its decimal quotient, carry, and digit decisions were too numerous and its recurrent coefficients could become unstable, whereas a fully bounded low-radix circuit would reduce each local decision to a small reusable gate.
The model represented each residue and modulus with twelve base-4 digits and evaluated a square by scanning the significant multiplier digits from high to low.
At each Horner step, the candidate was at most `7N` for a valid state, so a shared network chose one of seven quotient values; a low-to-high GRU then chose bounded local carries and one of four output digits at each position.
The selected recurrent state always remained in `{0,1,2,3}¹²`, and the same complete square transition was tied across every requested `T` step.
Decimal outputs were decoded by a learned MLP from eight harmonics of the state's phase at each decimal place, avoiding a separate learned decimal canonicalizer.
The custom loss combined official sequence cross entropy, a final-answer base-4 semantic loss, and the same label-free range, coefficient, overflow, and value-preservation invariants as L5.
The final candidate removed L5's curriculum because bounded state made every update stable; a short E3 runtime check showed that immediate full supervision improved exact accuracy from 0.31% to 1.81% without changing the architecture or optimizer.
The 21,061 persistent state elements contained no prompt or whole-value memories, residue or factor tables, numeric-range enumeration, dataset branches, T-specialized operators, solver-derived intermediate labels, or range-scaled parameters.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as the preceding experiments.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 97 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E2 | 87 | 0.21% | 0.00% | 0.10% | <1 | <1 |
| E3 | 177 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 267 | 0.44% | 0.33% | 0.39% | <1 | <1 |
| E5 | 54 | 0.25% | 0.17% | 0.21% | <1 | <1 |
| M1 | 45 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M2 | 40 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M3 | 243 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 57 | 0.07% | 0.12% | 0.09% | <1 | <1 |
| M5 | 49 | 0.13% | 0.20% | 0.17% | <1 | <1 |

L6 eliminated L5's non-finite failure and recovered L1's E3 score, confirming that low-cardinality bounded gates are materially easier to optimize than redundant decimal decisions.
The E3 score plateaued at exactly 1.81% between the 10-second runtime check and the 30-second screen, E4 remained below L3, and every depth profile again failed at T=1.
The architecture also nested a significant-digit scan inside every requested square: the fixed-T=2 datasets completed 177 to 267 updates, while the deeper Medium datasets completed only 40 to 57 despite twice the local wall-clock allowance.
The candidate therefore learned marginal output statistics rather than gate semantics, and its serial work reduced optimizer exposure precisely where deeper latent computation was needed.
The intuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next counterintuitive experiment should retain the bounded base-4 circuit but replace straight-through hard gates during training with a temperature-annealed soft relaxation, testing whether gradient mismatch rather than circuit expressivity prevented the algebraic invariants from identifying the discrete transition.
