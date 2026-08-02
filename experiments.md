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

## L7: Annealed soft radix gates

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that L6's circuit was expressive enough but its straight-through estimator optimized soft expectations in the backward pass while composing hard argmax values in the forward pass, creating a gradient mismatch at every quotient, carry, and digit gate.
L7 retained L6's architecture, base-4 state, algebraic invariant loss, immediate full-depth supervision, optimizer, and learning-rate schedule.
During training only, every categorical gate composed its probability-weighted expected value; a wall-clock exponential schedule reduced the softmax temperature from 2.0 to 0.05 across the budget.
Evaluation used the original hard argmax circuit, so any score gain required the relaxed training dynamics to identify discrete gate semantics rather than relying on fractional states at inference.
The unchanged 21,061 persistent state elements remained range-independent and contained none of the prohibited memories, tables, branches, intermediate labels, or specialized operators.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L6.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 108 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 86 | 0.21% | 0.00% | 0.10% | <1 | <1 |
| E3 | 185 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 264 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 58 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 46 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 41 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 253 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 60 | 0.07% | 0.10% | 0.08% | <1 | <1 |
| M5 | 53 | 0.02% | 0.03% | 0.03% | <1 | <1 |

Relative to hard-gate L6, the annealed relaxation improved E1 from 0.00% to 8.33%, E4 from 0.39% to 0.85%, E5 from 0.21% to 1.42%, M1 from 0.00% to 0.03%, and M2 from 0.00% to 0.57%.
E3 and M3 were unchanged at 1.81% and 0.48%, while E2 was unchanged within one example and M4 and M5 regressed.
Thus the gradient mismatch was consequential for shallow fitting but was not the primary barrier to modular composition: no profile certified T=1, and the nested digit scan still limited deeper datasets to 41–60 optimizer updates.
The counterintuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next intuitive experiment should replace the inner serial Horner scan with a parallel carry-lookahead-style reducer built from shared dilated local blocks, preserving bounded radix interactions while reducing physical depth per square from linear to logarithmic in the digit count.

## L8: Parallel carry-lookahead recurrence

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that L6 and L7 spent too much of the fixed clock scanning digits inside every square, and that parallel quotient and carry fields with logarithmic receptive-field growth could preserve their bounded arithmetic bias while exposing the parameters to several times more optimizer updates.
L8 formed all base-4 product coefficients through explicit pairwise multiplicative interactions, predicted twelve bounded base-4 quotient digits in parallel, and formed the coefficient residual `x² - qN` without a whole-value quotient.
One local block was shared over dilations 1, 2, 4, and 8 to give the quotient predictor full receptive field in four updates; a second shared dilated block predicted a continuous carry field and twelve bounded residue digits from the coefficient residual.
The invariant loss required `raw + incoming_carry - 4·outgoing_carry` to equal the predicted residue digits in the low positions and zero in the high positions, while also requiring the residue to lie in `[0,N)` and the final carry to vanish.
This coefficient identity supplied a dense algebraic constraint without quotient, carry, or residue labels for any intermediate square.
L8 retained L7's final-answer sequence and base-4 losses, annealed soft training decisions, hard evaluation decisions, Fourier decimal decoder, optimizer, and outer transition tied across `T`.
Its 135,274 persistent state elements remained range-independent and contained none of the prohibited lookup or solver mechanisms.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as the preceding experiments.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 244 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 248 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E3 | 628 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 624 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 272 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 216 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 213 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M3 | 1,244 | 0.25% | 0.40% | 0.33% | <1 | <1 |
| M4 | 414 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 370 | 0.42% | 0.33% | 0.38% | <1 | <1 |

The parallel design completed between two and five times as many updates as L7 and reduced E3 held-out token loss from about 2.49 to 2.27.
Nevertheless, E1, E3, E4, E5, and M1 produced exactly the same exact-example counts as L7, E2 and M2 regressed to zero, and M3 regressed from 0.48% to 0.33%; only M4 and M5 improved.
By the end of E3 training, the combined loss had fallen to 2.81, showing that the coefficient invariants were optimizable, but the hard transition still failed every T=1 certification profile.
The physical-depth hypothesis is therefore only partly supported: parallel lookahead solved the update-throughput bottleneck, but the final-answer and algebraic constraints still did not identify the correct latent quotient and residue.
The intuitive candidate is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next counterintuitive experiment should add target-conditioned bidirectional latent consistency during training: a learned reverse transition starts from the provided final answer and meets the forward tied recurrence at an interior latent state, providing dense credit assignment from final labels without exposing solver-derived intermediate residues.

## L9: Target-conditioned midpoint consistency

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that L8's forward quotient was underidentified by a final label and coefficient invariants alone, and that a learned target-conditioned reverse process could shorten the credit-assignment path without revealing the true intermediate residue.
L9 retained L8's complete forward architecture and training objective, and added a separate reverse transition consisting of one input projection, one local block shared across dilations 1, 2, 4, and 8, and a bounded base-4 digit head.
The model recorded the forward state after `floor(T/2)` tied squares.
Inside the custom training loss, the reverse transition started from the provided final-answer digits, ran for `ceil(T/2)` tied steps, and incurred a smooth latent-consistency loss against the forward midpoint.
The evaluator still performed the only backward pass, the reverse computation remained in the autograd graph and on the accelerator, and evaluation predictions used only the original forward recurrence.
The learned reverse state was not a solver-derived root or intermediate label: modular squaring is generally many-to-one, and the reverse network was free to select any latent preimage that agreed with the learned forward path.
The 198,014 persistent state elements remained range-independent and contained no prompt memories, bounded residue/factor tables, numeric enumeration, dataset branches, or T-specific parameters.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L8.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 213 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 205 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E3 | 514 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 510 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 211 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 167 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 162 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 1,018 | 0.33% | 0.43% | 0.38% | <1 | <1 |
| M4 | 349 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 268 | 0.42% | 0.33% | 0.38% | <1 | <1 |

All five Easy exact-example counts were identical to L8, as were M1, M4, and M5; M2 recovered the recurring 0.57% marginal score and M3 changed by only a few examples.
The reverse path reduced optimizer throughput and increased E3 held-out loss from about 2.27 to 2.28 in the full screen, while every matched and unseen-modulus profile still failed T=1.
Learned bidirectional agreement therefore supplied an easier auxiliary objective but did not select the arithmetic forward transition; the two learned paths could agree on latent behavior without improving final exactness.
The counterintuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next intuitive experiment should return to L8's faster forward-only model and correct the final-answer objective: mask base-4 semantic supervision to modulus-significant positions so padded high zeros cannot dominate, and emphasize the weakest official output digit with a smooth sequence-level maximum.

## L10: Significance-balanced sequence supervision

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that L8's twelve-position base-4 semantic loss was dominated by padded high zeros on the smaller Easy moduli, allowing a trivial zero-heavy state to overwhelm the output positions that determine exact accuracy, while mean token cross entropy permitted one weak decimal digit to invalidate an otherwise useful sequence.
L10 restored L8's faster forward-only architecture and retained its parallel quotient, dilated carry-lookahead, coefficient invariants, annealed soft gates, optimizer, and hard evaluation decisions.
The base-4 semantic loss was changed to average only positions at or below the modulus's highest nonzero radix digit, retaining every position that can represent a valid residue while removing structurally unused high padding.
The official decimal-token objective was changed from a per-sequence mean to a normalized smooth maximum with temperature 0.25, increasing the gradient from the worst predicted answer digit without changing labels or examples.
No architecture, optimizer, local budget, or dataset handling changed relative to the L8 control, and the 135,274 state elements remained competition-compliant and range-independent.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L8 and L9.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 286 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 251 | 0.21% | 0.00% | 0.10% | <1 | <1 |
| E3 | 614 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 609 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 279 | 0.92% | 0.17% | 0.54% | <1 | <1 |
| M1 | 218 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M2 | 218 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 1,216 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 411 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 354 | 0.42% | 0.33% | 0.38% | <1 | <1 |

The revised objective recovered the recurring E2, M2, and M3 marginal modes relative to L8, but E1, E3, E4, M4, and M5 were unchanged and E5 and M1 regressed.
E3 and E4 again produced exactly the same example counts as L7 through L9, despite E3's combined training loss falling to 3.33 and every model receiving hundreds of updates.
Padded high zeros and weak-digit averaging therefore affected which marginal decoder solution won on some families but did not cause the arithmetic-identifiability failure.
The intuitive loss hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next counterintuitive experiment should isolate the arithmetic learner from final-label shortcuts: train the quotient, carry, and residue transition only through its algebraic coefficient invariants, detach its recurrent state before the decimal decoder, and let official labels train only the readout from that state.

## L11: Label-isolated invariant transition

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that final-label gradients encouraged L8 through L10's transition to abandon arithmetic in favor of dataset-level output marginals, even though the coefficient invariants might identify the correct modular reducer when optimized without that conflicting shortcut.
L11 retained L10's complete forward architecture, parallel coefficient identities, annealed soft gates, hard evaluation decisions, optimizer, and smooth weakest-decimal-digit objective.
During training, the recurrent radix state was detached before the Fourier decimal decoder, so official answer gradients updated only the readout.
The final-answer base-4 semantic loss was removed entirely; quotient, carry, and residue parameters received gradients only from coefficient preservation, residue range, final-carry, and entropy constraints at every tied square.
This separation used the same examples and final labels, introduced no pseudo-labels or hidden solver state, and left the 135,274 persistent state elements unchanged.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L8 through L10.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 254 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 225 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E3 | 630 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 623 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 253 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 210 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 205 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M3 | 1,240 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 414 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 344 | 0.42% | 0.33% | 0.38% | <1 | <1 |

On E3, the combined training loss fell from 43.37 at the first update to 2.30 by update 100 and 2.29 by the end, showing that the invariant contribution had become negligible beside decoder cross entropy.
Despite that collapse, E1, E3, E4, E5, M1, M4, and M5 exactly matched L8's hard example counts, E2 and M2 were zero, and every T=1 profile failed.
Final-label interference is therefore not the barrier: the continuous quotient, carry, and residue expectations admit a low-invariant solution whose annealed argmax decisions are not the corresponding arithmetic circuit.
The counterintuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next intuitive experiment should keep invariant-only transition learning but compose hard straight-through quotient and residue digits during training, making the forward computation identical to evaluation so a soft degenerate solution cannot satisfy the constraints without correct discrete decisions.

## L12: Hard invariant-only transition

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that L11's algebraic constraints were sufficient in principle but were satisfied by fractional quotient and residue expectations that did not correspond to the hard argmax circuit used at evaluation.
L12 retained L11's parallel architecture, detached decoder, invariant-only transition gradients, entropy regularization, optimizer, and temperature schedule.
During training, every quotient and residue gate composed its hard argmax value exactly as evaluation did, while a straight-through estimator supplied gradients through the temperature-scaled categorical expectation.
This made the physical forward computation identical in training and evaluation without adding labels, memories, tables, branches, specialized operators, or state elements.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L11.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 240 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 214 | 0.83% | 0.00% | 0.42% | <1 | <1 |
| E3 | 609 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 606 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 220 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 195 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M2 | 215 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 1,237 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 406 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 368 | 0.42% | 0.33% | 0.38% | <1 | <1 |

Hard training gates improved E2 to the best valid local result for that dataset, but E1, E3, E4, E5, M2, M3, M4, and M5 remained at recurring marginal modes and M1 regressed to zero.
On E3, the invariant-only combined loss again fell to about 2.30 by update 100 and stayed there through update 609 without changing a single held-out exact example.
The soft-forward mismatch is therefore not sufficient to explain the degeneracy: the continuous carry field can absorb coefficient discrepancies even when quotient and residue digits are discrete, or the learned Fourier readout may conceal any improvement in the radix state.
The intuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next counterintuitive experiment should test the readout explanation directly by replacing the Fourier phase decoder with a tied learned base-4-to-decimal conversion recurrence whose local value-preservation invariant and final token loss train an explicit decimal state.

## L13: Tied radix-conversion readout

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that L12's invariant-only modular transition had learned more arithmetic than its exact scores revealed, but the direct Fourier phase readout could not convert the resulting base-4 state into a consistent decimal sequence.
L13 retained L12's complete modular transition, hard straight-through quotient and residue gates, detached arithmetic state, optimizer, schedule, and loss weights.
It replaced only the Fourier readout with a twelve-step conversion recurrence that consumed the base-4 state from most to least significant digit and maintained an explicit eight-digit decimal accumulator.
At each tied step, one local block shared across dilations 1, 2, and 4 predicted decimal digits and base-4-sized carries in parallel.
The unlabeled constraint `4 * old digit + source addend + incoming carry = new digit + 10 * outgoing carry` enforced local value preservation, while only the final decimal logits received official answer labels.
The candidate used 192,808 persistent state elements and added no intermediate solver labels, lookup memory, range enumeration, dataset branches, T-specialized operators, or value-sized parameter tables.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L12.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 169 | 6.00% | 10.00% | 8.00% | <1 | <1 |
| E2 | 168 | 2.29% | 2.00% | 2.15% | <1 | <1 |
| E3 | 307 | 0.88% | 0.63% | 0.75% | <1 | <1 |
| E4 | 307 | 0.19% | 0.17% | 0.18% | <1 | <1 |
| E5 | 173 | 1.33% | 1.00% | 1.17% | <1 | <1 |
| M1 | 179 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 175 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 583 | 0.21% | 0.07% | 0.14% | <1 | <1 |
| M4 | 300 | 0.22% | 0.18% | 0.20% | <1 | <1 |
| M5 | 249 | 0.42% | 0.33% | 0.38% | <1 | <1 |

The tied converter raised E2 from 0.42% to 2.15%, but that gain was isolated: E1, E3, E4, E5, and M3 regressed; M2 and M5 exactly reproduced L12's split counts; and M4 gained only two test examples while remaining at the marginal floor.
No familiar- or unseen-modulus profile certified even T=1.
The combined losses fell from 12.54--40.11 initially to 2.22--2.33, but E3 completed only 307 updates versus L12's 609 and M3 completed 583 versus 1,237 because the twelve recurrent conversion steps approximately doubled the cost of shallow-T training.
The Fourier decoder was therefore not concealing a broadly correct radix state: an explicit value-preserving conversion neither exposed target-family arithmetic nor transferred its small fixed-modulus E2 gain, and its additional serial depth starved the unchanged modular transition of updates.
The counterintuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next intuitive experiment should restore L12's faster Fourier readout and make its remaining continuous carry field discrete with hard straight-through integer rounding, eliminating the last source of fractional slack in the coefficient identities without adding a carry table or range-sized parameters.

## L14: Hard integer carry field

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that L12's hard quotient and residue decisions still admitted a non-arithmetic low-loss solution because its scalar carry field remained continuous and could absorb coefficient discrepancies fractionally.
L14 restored L12's complete fast Fourier control and changed only the existing scalar carry head: both training and evaluation rounded every carry to an integer in the physical forward pass, while training used an identity straight-through gradient through that rounding operation.
The arithmetic transition remained isolated from official labels, the quotient and residue gates remained hard straight-through choices, and the candidate retained exactly 135,274 persistent state elements.
Integer rounding introduced no carry table, bounded value inventory, additional parameter, dataset branch, T-specialized operation, intermediate label, or range-sized state.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L12 and L13.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 249 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 225 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| E3 | 609 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 589 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 232 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 203 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 197 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 1,241 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 411 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 333 | 0.42% | 0.33% | 0.38% | <1 | <1 |

E1, E3, E4, E5, M2, M3, M4, and M5 reproduced L12's test and OOD counts exactly, E2 regressed from four test examples to zero, and M1 gained only two OOD examples among 3,000 while retaining zero test examples.
Every familiar- and unseen-modulus certification again failed at T=1.
The combined final losses lay between 2.275 and 2.304, and throughput matched L12 closely, including 609 E3 updates and 1,241 M3 updates, so neither compute starvation nor numerical failure explains the identical behavior.
Continuous carry slack was therefore not the operative loophole: making the entire quotient/carry/residue forward circuit discrete did not make the invariant-only optimizer find exact positional arithmetic.
The remaining averaged, coefficient-normalized invariant can become negligible while discrete equations are still violated, leaving the detached decoder at its recurring label marginals.
The intuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next counterintuitive experiment should retain this fully discrete circuit but replace its mean of tiny normalized coefficient penalties with a smooth worst-constraint objective over unnormalized integer violations, forcing optimization to repair the least-satisfied positional identity rather than hiding it among 24 coefficients.

## L15: Worst discrete coefficient objective

Date: 2026-08-02.

Idea class: counterintuitive.

Status: rejected before hosted submission.

The hypothesis was that L14's fully discrete circuit still failed because its invariant averaged 24 coefficient errors after division by 109, allowing a few violated positional identities to contribute almost no gradient even though exact arithmetic requires every identity to hold.
L15 retained L14's architecture, hard straight-through quotient, integer carry, and residue decisions, detached Fourier decoder, optimizer, schedule, entropy term, and 135,274 persistent state elements.
It changed only the algebraic objective: for each example it took a temperature-0.1 smooth maximum of `log(1 + absolute violation)` over the 24 unnormalized integer coefficient equations and the final carry, then squared that worst constraint.
Exact coefficient satisfaction still had zero loss, but a single one-unit error could no longer disappear through coefficient averaging or large-range normalization.
The loss added no labels, learned state, tables, branches, enumeration, or specialized computation.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L14.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 242 | 6.67% | 10.00% | 8.33% | <1 | <1 |
| E2 | 220 | 0.21% | 0.00% | 0.10% | <1 | <1 |
| E3 | 626 | 1.13% | 2.25% | 1.69% | <1 | <1 |
| E4 | 620 | 0.37% | 1.33% | 0.85% | <1 | <1 |
| E5 | 240 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 204 | 0.00% | 0.07% | 0.03% | <1 | <1 |
| M2 | 199 | 0.00% | 0.00% | 0.00% | <1 | <1 |
| M3 | 1,235 | 0.50% | 0.47% | 0.48% | <1 | <1 |
| M4 | 412 | 0.19% | 0.18% | 0.18% | <1 | <1 |
| M5 | 335 | 0.42% | 0.33% | 0.38% | <1 | <1 |

E3 lost one exact test and one exact OOD example relative to L14, E4 was unchanged, M2 fell from 44 test and 33 OOD examples to zero, and the remaining families either exactly reproduced L14 or moved by only one fixed-family test example.
No familiar- or unseen-modulus profile certified T=1.
The stronger objective did not disappear into decoder cross entropy: final combined loss remained 2.743 on E3 and 2.725 on M3, compared with about 2.30 under L14, after 626 and 1,235 updates respectively.
Concentrating gradient on the least-satisfied equation therefore exposed rather than repaired the residual constraint violations.
The failure is not explained by normalized averaging; the invariant-only, straight-through parallel quotient/carry/residue predictor does not optimize the exact discrete constraint system within the budget, even when the loss continues to signal its worst error.
The counterintuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next intuitive experiment should keep L15's discrete circuit and worst-constraint loss but add direct cross-entropy on the final radix state obtained by re-encoding only the official final answer, supplying the transition with an identifiable end-state gradient without introducing any solver-derived intermediate label.

## L16: Final radix endpoint supervision

Date: 2026-08-02.

Idea class: intuitive.

Status: rejected before hosted submission.

The hypothesis was that L15's exact constraints supplied error detection but no identifiable direction toward the correct discrete quotient and residue choices, whereas the official final answer could supervise the requested endpoint directly without revealing any intermediate square.
L16 retained L15's fully discrete quotient/carry/residue forward circuit, worst-coefficient constraint, detached Fourier decoder, optimizer, schedule, and 135,274 persistent state elements.
The loss parsed only the official final decimal target, re-encoded that same value into base-4, and applied weight-0.75 cross-entropy to the final recurrent state's digit logits over positions significant for the input modulus.
The decimal token loss continued to train only the detached readout, while the transition received the new endpoint loss plus its algebraic constraints.
This used no intermediate solver label, lookup, enumeration, branch, specialized T operation, or added model state.

The frozen candidate was screened with the same official datasets, splits, Apple M2 Pro device, 30-second Easy budgets, and 60-second Medium budgets as L15.

| Dataset | Updates | Test | OOD | Mean exact accuracy | Max T | OOD N Max T |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1 | 242 | 6.00% | 10.00% | 8.00% | <1 | <1 |
| E2 | 215 | 0.83% | 0.67% | 0.75% | <1 | <1 |
| E3 | 604 | 1.25% | 2.37% | 1.81% | <1 | <1 |
| E4 | 601 | 0.31% | 1.00% | 0.66% | <1 | <1 |
| E5 | 233 | 1.33% | 1.50% | 1.42% | <1 | <1 |
| M1 | 204 | 0.03% | 0.00% | 0.02% | <1 | <1 |
| M2 | 197 | 0.49% | 0.66% | 0.57% | <1 | <1 |
| M3 | 1,200 | 0.21% | 0.30% | 0.25% | <1 | <1 |
| M4 | 406 | 0.16% | 0.19% | 0.17% | <1 | <1 |
| M5 | 326 | 0.42% | 0.33% | 0.38% | <1 | <1 |

Endpoint supervision raised fixed-modulus E2 to 0.75%, but the gain was isolated: E3 exactly returned to the recurring 1.81% split pair, E4 regressed from 0.85% to 0.66%, M3 regressed from 0.48% to 0.25%, and the other families remained at marginal counts.
Every familiar- and unseen-modulus profile failed at T=1.
The intended auxiliary signal remained active rather than disappearing: total final training losses were 3.48--3.94 while decimal-only evaluation losses were 2.18--2.38, leaving approximately an `ln(4)`-scale radix-and-constraint gap after hundreds to 1,200 updates.
An identifiable final state target is therefore insufficient when its gradient must coordinate earlier hard straight-through states and a parallel quotient/carry factorization.
The tied circuit neither fit its directly supervised endpoint reliably nor converted that signal into a reusable one-step arithmetic rule.
The intuitive hypothesis is rejected, no hosted quota was consumed, and the hosted results stores remain unchanged.
The next counterintuitive experiment should preserve L16's forward computation and losses but detach the hard recurrent state between requested squares, truncating backpropagation through time so each tied call learns from its own algebraic constraint and the last call learns the endpoint directly instead of letting long straight-through credit corrupt earlier local transitions.
