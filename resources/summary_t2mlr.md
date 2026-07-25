# T²MLR: Transformer with Temporal Middle-Layer Recurrence

- Authors: Ziyang Cai, Xingyu Zhu, Yihe Dong, Yinghui He, and Sanjeev Arora.
- ArXiv identifier: 2607.15178v2.
- Source URL: https://arxiv.org/src/2607.15178.
- PDF URL: https://arxiv.org/pdf/2607.15178.
- Date read: 2026-07-25.

## Overall assessment

The paper introduces a deep-to-shallow recurrent connection that carries one token's middle-layer state into an earlier layer while processing the next token.
Its central architectural result is credible at moderate confidence: under matched parameter count, training data, and inference compute, the recurrent pathway usually improves language-model and reasoning evaluations, and a fixed-width ablation favors middle-layer placement over equally wide early- or late-layer placement.
The evidence does not establish a training-compute advantage, because the Jacobi approximation makes training roughly two to four times slower and a wall-clock-matched Transformer beats the reported 135M T²MLR model on zero-shot NLP.
The claim that the method preserves useful latent reasoning state is plausible but not isolated cleanly, because the paper lacks multi-seed estimates, most downstream comparisons are not training-compute matched, and its future-token probe is confounded by the recurrent model's better language-modeling loss.
The paper has strong conceptual relevance to persistent learned state and tied iterative computation, but the architecture has only weak direct relevance to One Layer Deeper unless recurrence is moved from the token axis to the latent squaring-step axis.
The most useful transferable idea is therefore not the paper's token-to-token recurrence itself, but its identity-initialized gated fusion of a persistent cache into a reusable computation block.

## 1. Claims, evidence, and limits

The authors identify a depth-time barrier in an autoregressive Transformer: a deep representation computed for token `t` cannot directly influence the shallow processing of token `t+1`.
This formulation is more precise than the paper's broader token-bottleneck rhetoric, because an ordinary Transformer already preserves continuous information about earlier tokens in its per-layer key-value caches.
The actual architectural novelty is a cross-layer temporal shortcut from a later layer at one token position to an earlier layer at the next token position, rather than the first continuous communication channel between tokens.

Section 2 defines recurrence boundaries `l_start` and `l_end`, stores a width-`d` cache from the representation after `l_end`, and fuses the previous cache before block `l_start` at the next token.
Equation 2 is the defining change to the Transformer forward pass, Equation 3 specifies the gated fusion, and Equation 4 updates the cache with an RMS-normalized temporal residual.
The remaining layers, the ordinary token stream, self-attention, and the usual key-value caches stay intact.

The cleanest evidence for the hybrid state-tracking and retrieval capability is the `S_5`-Retrieval experiment in Section 3 and Figure 4.
The task requires a model to maintain a cumulative permutation state while retrieving a freshly randomized string associated with that state from an in-context dictionary.
Training uses state lengths from 1 through 32 and evaluation extends to 48, so the experiment measures a limited form of length extrapolation.
The four-layer T²MLR model is nearly perfect in distribution and retains about 0.86 average token accuracy at length 48, while its exact-sequence accuracy falls sharply beyond the training range and approaches zero by length 48.
The parameter-matched LSTM and Transformer baselines have zero sequence exact match throughout the plotted range and only low token accuracy, which demonstrates a large empirical separation but also suggests that optimization details deserve closer control before attributing the entire gap to expressivity.
This experiment uses full recurrence over the four-layer model, so it supports the hybrid attention-plus-recurrence thesis but does not support the separate claim that middle-layer placement is superior.
The task also emits every intermediate cumulative state, which aligns dense supervision and recurrent token positions with algorithmic steps much more directly than One Layer Deeper does.

The main 135M pretraining comparison in Section 4.1 and Table 1 matches parameter count at about 136.4M and trains every model for one epoch on 10B FineWeb-Edu tokens.
The baseline's seven-task zero-shot average is 42.83, while T²MLR configurations with recurrence depths 30, 22, 14, 6, and 2 obtain 43.36, 43.73, 44.01, 44.14, and 42.89.
The best configuration recurs over layers 13 through 18, which is only 20 percent of the 30-layer network.
Appendix F reports perplexity improvements from 22.64 to 21.54 at 135M and from 16.12 to 15.79 at roughly 368M.
Figure 5 shows that language-modeling cross-entropy improves as recurrence widens through much of the sweep, whereas downstream task averages favor a narrower middle block, so perplexity does not by itself explain the placement result.

The fixed-width location ablation in Appendix C.1 and Table 11 is the strongest evidence that placement matters independently of recurrent-block width.
At width 6, early, middle, and late placement yield zero-shot averages of 42.74, 44.14, and 42.87 against a 42.83 baseline.
At width 14, early, middle, and late placement yield 42.21, 44.01, and 43.14.
These comparisons favor middle placement in both widths, but they are reported only at 135M, on one evaluation suite, and without variance estimates.

Section 4.2 and Table 2 compare T²MLR with pause-token, full-looped, and middle-looped models at matched parameter count and training data.
T²MLR reaches a best seven-task average of 44.14, compared with 43.31 for the pause-token model, 42.99 for the full-looped model, and 42.68 for the middle-looped model.
The comparison supports the claim that T²MLR obtains useful extra computation without multiplying autoregressive decoding depth, but it does not match training wall-clock across architectures.

Section 4.3 and Figure 6 report improvements on variable assignment, ProsQA-Hard, HotpotQA-Easy, and natural-language and symbolic GSM-Aug.
The largest absolute gain is on depth-five variable assignment, where accuracy rises from 0.494 for the baseline to 0.945 for the narrowest middle-recurrence variant.
The best ProsQA-Hard result is 0.178 versus 0.151, the best HotpotQA-Easy result is 0.268 versus 0.214, the best symbolic GSM-Aug result is 0.436 versus 0.381, and the best natural-language GSM-Aug result is 0.314 versus 0.265.
Different recurrence widths win different tasks, and the narrow six-layer variant is nearly equal to the baseline on HotpotQA-Easy despite winning variable assignment.
The appendix states that medium and hard HotpotQA subsets were omitted because both architectures struggled, which is an important boundary condition on the reported multi-hop result.
The reasoning comparisons are parameter- and data-matched but not training-wall-clock matched, so the extra Jacobi computation remains a credible alternative explanation for part of the gain.

Section 4.4 and Table 3 retrofit a recurrent pathway into SmolLM2-1.7B-Instruct and finetune for one epoch on OpenMathReasoning.
The retrofit improves GSM8K from 35.78 to 39.88 and MATH500 from 12.80 to 18.00.
This establishes that the new path can be learned without pretraining the whole model from scratch, but the comparison adds parameters and training computation to the recurrent model and reports no multi-seed uncertainty.

Appendix C.1 extends the trend to 361M and 1B models and to 50B-token pretraining.
At 10B tokens, the seven-task average improves from 48.48 to 49.22 near 361M and from 51.38 to 52.37 near 1B.
At 50B tokens, the average improves from 48.35 to 49.42 near 135M and from 52.83 to 54.78 near 361M.
The scaling evidence shows persistence across three model sizes and two data scales, but it remains a small collection of single reported runs rather than a scaling law.

Appendix D probes whether intermediate states predict the next token more effectively.
Two-layer MLP probes trained for 20,000 steps consistently obtain lower next-token loss from T²MLR representations than from baseline representations.
This result is compatible with the proposed temporal credit-assignment mechanism, but it does not distinguish a specialized recurrent state from the general consequence of improved pretraining loss.
A stronger intervention would disable, scramble, or substitute the recurrent cache at evaluation time while measuring both next-token loss and reasoning accuracy.

Appendix B.4 compares Jacobi training with exact recurrent training on the first 1B FineWeb-Edu tokens at 135M, 360M, and 1B.
The Jacobi-trained validation losses are within 0.0045 of exact recurrent training, and approximate versus exact evaluation produces essentially identical perplexity for those models.
This supports the approximation as a quality-preserving training device at the tested sequence length and distribution.
The gradient study in Appendix B.3 uses one checkpoint and one batch and shows that forward depth below 8 or backward depth below roughly 4 can change gradients sharply, while the main setting of forward depth 16 and backward depth 4 is close to the 32-by-32 anchor.

The primary negative result is Appendix C.2 and Table 12.
When a 135M Transformer is trained for 2.24 epochs to match the wall-clock of T²MLR over one epoch, its zero-shot average reaches 45.30 and exceeds T²MLR's 44.14.
The paper therefore demonstrates an architectural benefit at fixed data and inference cost, not better quality per unit of training time.
This distinction is especially important for a competition whose dominant constraint is a 60-second or 600-second H100 training budget.

Appendix B.5 measures autoregressive generation overhead between 4.1 and 8.2 percent across model sizes and generation lengths, with less than 0.1 percent extra peak inference memory.
That result concerns cached single-token decoding and does not make the additional training passes free.
Table 8's prefill latency annotation is internally inconsistent because the parenthesized quantity is called a speedup over exact prefill but increases monotonically as the number of Jacobi iterations rises.
The released source code also implements truncated backpropagation by detaching early Jacobi iterates and retaining the final backward window, while Algorithm 1 in Appendix B.1 prints the opposite detachment condition.
These inconsistencies do not overturn the main empirical results, but they reduce confidence in the precise prefill and pseudocode claims until clarified.

The paper reports no multi-seed standard deviations or confidence intervals, and Section 6 explicitly leaves multi-seed estimates to future work.
The main `S_5` text states a learning rate of `1e-3`, batch size 64, 150,000 T²MLR steps, and 400,000 baseline steps, while Appendix F's training table states `5e-4` and 200,000 steps.
The absence of a consistent experiment record makes the synthetic result harder to reproduce from the paper alone.

## 2. Mechanism and intuition

The intuitive mechanism is a persistent scratch register that bypasses token decoding and re-enters the network where reusable abstract features are still represented.
At token `t`, layers through `l_end` produce a candidate reasoning state, and that state is added residually into a recurrent cache.
At token `t+1`, the cache is projected and fused with the current residual stream immediately before `l_start`, after which the same middle block can refine it again.
The recurrent cache therefore accumulates computation over token time while attention remains available for content-addressed access to the complete token history.

Equation 3 keeps an unconditional identity path for the current representation and adds two gated corrections.
One correction rescales the current stream, and the other injects a learned projection of the recurrent cache.
Two learned scalar gates provide global path strength, two sigmoid vector gates provide input-dependent feature selection, and both scalar gates start at zero.
Zero initialization makes the new module exactly equivalent to the base Transformer at initialization and lets recurrence enter training gradually.

Equation 4 updates the cache as `RMSNorm(h_t^(l_end) + R_(t-1))`.
The temporal residual gives old state a direct path across arbitrarily many generated tokens, while normalization limits scale drift.
Appendix E observes that the recurrent scalar gate becomes positive and the current-stream correction becomes negative, which is consistent with recurrence acting as an additive refinement to the ordinary residual path.

Teacher-forced training creates a circular-looking dependency across token positions because every token is normally processed in parallel but its recurrent input depends on the preceding token's later-layer output.
The paper resolves this dependency as a fixed-point problem and performs Jacobi iterations over the entire sequence in parallel.
A standard forward pass seeds every shifted cache, repeated middle-block passes refine all caches, and only a final limited window of iterations retains gradients, analogous to truncated backpropagation through time.
The default forward depth is 16 and backward depth is 4.

The mechanism's main assumption is that useful computation can advance once per token.
It is best aligned with tasks whose reasoning trajectory contains enough generated or supervised tokens to carry the state, as in `S_5`-Retrieval and chain-of-thought finetuning.
It does not create `T` hidden updates from a short input that merely contains the numeral `T`.

## 3. Relevance to One Layer Deeper

One Layer Deeper asks for `x^(2^T) mod N`, whose natural algorithm applies the same modular-squaring transition exactly `T` times.
The desired inductive bias is recurrence over algorithmic steps, with a state holding the current residue, immutable modulus context, requested depth, and progress.
T²MLR instead applies recurrence over token positions, so its recurrence count is determined by prompt and generated sequence length rather than the numeric value of `T`.

The public separate-input/output representation contains `[N] digits(N) [X] digits(x) [T] digits(T)` and does not expose intermediate residues or chain-of-thought tokens.
The evaluator requests all output digits from selected positions of the same short input sequence, so there is no autoregressive output trajectory on which the paper's cache can perform one squaring per token.
Since the decimal representation of `T` grows as `O(log T)`, token-time recurrence cannot by itself supply `O(T)` modular-squaring updates.
This axis mismatch makes a literal T²MLR wrapper around `IntermediateTransformer` unlikely to improve ordinary OOD `T` or the `Max T` ladder for the intended reason.

The `S_5` result remains relevant because it demonstrates that a fixed-width recurrent register and attention can coexist successfully when a task needs both state evolution and retrieval.
For modular squaring, the analogous division is an evolving residue register plus stable access to `N`, `T`, and digit-level input structure.
The competition differs critically because only the final residue is supervised, whereas `S_5` supervises every intermediate state.

The middle-layer placement result suggests a useful decomposition of each learned squaring step into read, compute, and write phases.
An early sub-block can read or normalize the current residue and immutable context, a tied middle sub-block can approximate one modular squaring, and a late sub-block can prepare digit decoding.
The persistent cache should feed the compute sub-block rather than overwrite raw digit embeddings or enter only at the vocabulary head.
This is an interpretation for the competition, not an experiment reported by the paper.

The fixed-size cache is compatible with the model-state constraint because activation state does not count as persistent parameters and the public manifests allow up to 500 million persistent model-state elements.
The present model has about 405,000 parameters for a 32-position `ModelSpec`, so parameter capacity is not the limiting resource.
At width 128, the paper's two `2d`-to-`d` gates, one `d`-to-`d` recurrent projection, biases, and scalar gates add about 82,000 parameters.
The actual constraint is H100 wall-clock, because exact latent unrolling costs roughly linearly in the number of squaring steps and the paper's Jacobi method sacrifices two to four times more training time.

T²MLR does not provide a halting or progress mechanism.
One Layer Deeper requires either an exact update mask derived from `T`, a progress counter that generalizes beyond training depths, or a learned halting rule that is separately validated on unseen `T`.
Without such a mechanism, a recurrent state may learn repeated updates but still stop at the wrong depth.

The paper provides no direct evidence for generalization across unseen moduli.
Its attention-plus-state design could preserve `N` as immutable context while updating the residue, but the reported experiments do not test learned arithmetic under changes in operand bit width or modulus identity.
Any claim about `OOD N Max T` is therefore a repository-specific hypothesis rather than a paper result.

## 4. Mapping onto `submission.py`

The current `IntermediateTransformer` is an encoder-only two-block Transformer followed by a token-level vocabulary head.
Its two `TransformerBlock` instances are untied, it has no persistent working state, it performs no explicit recurrent steps, and `T` influences predictions only through ordinary self-attention.
Its attention mask is a bidirectional padding mask in this data representation, so the model is not executing the paper's causal token-by-token recurrence.

A faithful competition adaptation should first separate input encoding from latent iteration instead of inserting a cache between the existing two token layers.
`IntermediateTransformer.__init__` could expose an `InputEncoder`, a single shared `TiedSquaringTransition`, a T²MLR-inspired `StateFusion`, and a `DigitDecoder`.
`IntermediateTransformer.forward` could encode all prompt tokens once, split the result into immutable context and mutable working registers, execute the same transition repeatedly, and decode the final register back into logits at every input position required by the evaluator.

`StateFusion` could implement Equation 3 with `gamma_current` and `gamma_recurrent` initialized to zero, two linear gates over the concatenated current and cached states, and a recurrent projection.
`TiedSquaringTransition` should be instantiated once and called at every latent step, unlike the present `ModuleList` of independent blocks.
The cache update should preserve separate residue, modulus, requested-depth, and progress channels rather than RMS-normalizing all information into one undifferentiated vector.
Keeping `N` and `T` as immutable side context avoids accumulating numerical drift in quantities that must remain stable across steps.

The current output contract requires logits of shape `(batch, input_length, 17)`.
The decoder can satisfy it by combining the final sequence-level working state with position-to-end features and broadcasting digit predictions across all prompt positions, after which the evaluator selects the target positions.
This approach does not require changing `training_loss` or the submission API.

`build_optimizer` is the concrete place to assign a distinct parameter group to zero-initialized fusion scalars if the 80-step schedule otherwise leaves recurrence dormant.
Any gate learning-rate multiplier would be a competition-specific optimization choice, because the active paper text does not establish one as necessary.
`training_loss` can retain final token cross-entropy while using the returned `auxiliary` object for cache-use, halting, or trajectory regularizers.
The official labels expose only the final residue, so intermediate-residue supervision should not be assumed unless it is generated explicitly and judged acceptable under competition rules.

The paper's Jacobi approximation should not be the first implementation in this repository.
Prompt sequences are short, the desired latent horizon is at most the certification ladder through 64, and exact unrolling gives an unambiguous correspondence between one shared transition and one squaring step.
Jacobi training becomes worth considering only if profiling shows exact unrolling prevents enough optimizer steps from fitting inside the H100 budget.

The present constants `NUM_LAYERS = 2`, `WIDTH = 128`, and `SCHEDULE_STEPS = 80` leave no meaningful early-middle-late depth decomposition.
A low-cost adaptation can define early, recurrent-middle, and late transformations inside one tied transition rather than expanding to a deep 30-layer Transformer.
This preserves the paper's functional placement idea without importing its language-model scale.

## 5. Smallest decisive experiment

The smallest decisive experiment should use E1 because fixed `N = 323` isolates extrapolation in `T` from modulus generalization.
The control should be an explicit tied latent transition with the same encoder, decoder, state width, number of unrolled steps, parameter count, optimizer, and wall-clock budget as the treatment.
The treatment should add only the identity-initialized T²MLR-style fusion and temporal residual around the mutable working state.
Both models should use exact latent unrolling and the same deterministic or learned progress policy.

Training and model selection should use only the official E1 training depths `T = 1, 2, 3`.
Evaluation should report familiar-depth exact accuracy, ordinary OOD accuracy at `T = 6`, the consecutive `Max T` prefix, per-digit accuracy, completed training steps, elapsed time, and gate magnitudes.
A local diagnostic set at `T = 4, 8, 16, 32, 64` should be generated only for evaluation to reveal where the transition begins to drift.
Intermediate decoded residues can be probed against exact repeated-squaring states for analysis without adding them to the training loss.

The hypothesis is confirmed if the gated model improves exact accuracy at `T = 6` and extends the consecutive depth prefix while matching familiar-depth accuracy and training wall-clock.
The mechanism is further supported if intermediate probes track the true residue, the recurrent gate moves away from zero, and shuffling or zeroing the cache at evaluation destroys the depth gain.
The hypothesis is falsified if gains are confined to familiar depths, disappear under equal wall-clock, survive cache ablation, or come only from a larger parameter count.

Only after a positive E1 result should the same transition be tested on E5 or M5 for joint unseen-`T` and unseen-`N` behavior.
A result on E1 alone cannot establish modulus generalization because the transition may specialize to arithmetic modulo 323.

## 6. Controlled E1 tied-depth result

Submission 6 tested the smallest parameter-matched recurrence change against submission 4, the existing two-layer width-128 E1 control.
The treatment preserves the control's embeddings, two Transformer blocks, output head, initialization, optimizer, warmup-plus-cosine schedule, dropout, label smoothing, 80-step limit, and seed.
It changes only the forward topology: block 0 encodes once, block 1 is reused for exactly the input's requested `T` state updates, and a GPU-resident mask leaves completed examples unchanged.
Training unrolls the public E1 maximum of three updates, while evaluation executes a fixed 64-iteration GPU loop and masks state updates after each example's decoded `T`.
This design adds no parameters, performs no modular arithmetic, and uses the public `T` field only for input-dependent routing.
For `T = 1`, the treatment is numerically identical to the control at equal weights.

The hosted H100 run succeeded as submission `e5bc8c6c-5d66-4a71-982c-3479de825133` from source commit `6c91e99`.
Both models contain 402,816 persistent state elements, create 805,661 optimizer-state elements after the first step, use batch size 512, complete all 80 training steps, and run with seed 74.
The tied-depth model scored 4.33 percent mean exact accuracy against 5.17 percent for the control, a decrease of 0.83 percentage points or about 16 percent relative.
Familiar-depth test exact accuracy increased from 1.33 percent, or 2 of 150 examples, to 2.67 percent, or 4 of 150 examples.
OOD exact accuracy at `T = 6` decreased from 9 percent, or 9 of 100 examples, to 6 percent, or 6 of 100 examples.
Neither model certified the first `T = 1` rung on the ordinary or OOD-`N` depth ladder, so both report `Max T <1` and `OOD N Max T <1`.

Cross-entropy moved in the opposite direction from exact match.
The treatment reduced familiar-depth test loss from 1.989 to 1.980, OOD loss from 1.909 to 1.785, mean evaluation loss from 1.949 to 1.882, and final training loss from 1.933 to 1.924.
The most conservative interpretation is that tied recurrence improved average token likelihood slightly but did not organize the full output digit sequence into a more exact repeated-squaring computation.
The two additional familiar-depth successes and three lost OOD successes are too few for a reliable claim about small accuracy differences from this single seed.
The failed certification is nevertheless decisive for the architectural hypothesis as tested: exact `T`-conditioned block reuse alone did not produce measurable algorithmic depth extrapolation.

The likely failure is underdetermined transition learning under final-answer-only supervision.
The shared block receives gradients through one, two, or three applications during training, but nothing forces one application to represent one modular squaring or keeps its hidden-state dynamics stable at six or more applications.
Lower OOD cross-entropy alongside lower OOD exact match is consistent with better marginal digit probabilities but continued sequence-level errors, rather than a learned latent state machine.
A useful next treatment would introduce an explicit mutable residue register and identity-initialized gated fusion while leaving immutable modulus context outside the recurrent state.
That experiment should be compared with this tied-depth model, not only with the original static encoder, because it would isolate the paper's state-fusion mechanism from recurrence itself.

## 7. T²MLR gated-cache E1 result

Submission 7 implemented the paper's gated fusion and temporal cache residual as closely as the separate-input/output benchmark permits.
The architecture uses one early Transformer block, one middle Transformer block, and one late Transformer block.
At each latent step, the fixed early representation is fused with a separate recurrent cache using Equation 3's two random linear feature gates, two zero-initialized scalar gates, and learned recurrent projection.
The middle block processes the fused representation, after which Equation 4 updates the cache with an RMS-normalized temporal residual.
The late block processes only the final active middle representation, matching the paper's separation between the recurrent middle path and the ordinary late path.
Exact unrolling replaces the paper's Jacobi approximation because E1 exposes at most three training steps and the benchmark requires latent recurrence over numeric `T`, not recurrence over an autoregressive output trajectory.

The hosted H100 run succeeded as submission `a6eb3958-62c8-401e-84d2-87ca895b23ef` from source commit `47646db`.
The E1 manifest instantiated 683,522 model-state elements and 1,367,094 optimizer-state elements after the first step, compared with 402,816 and 805,661 for both earlier controls.
All models used batch size 512, seed 74, and completed the same 80 optimizer steps.
The gated-cache model required 42.81 training seconds and 9.28 evaluation seconds, compared with 24.44 and 4.99 seconds for the static control and 21.30 and 3.98 seconds for direct tied depth.

Mean exact accuracy was 4.17 percent, below 5.17 percent for the static two-layer control and 4.33 percent for direct tied depth.
Familiar-depth test accuracy was 1.33 percent, or 2 of 150 examples, equal to the static control and below the tied-depth model's 2.67 percent.
OOD accuracy at `T = 6` was 7 percent, or 7 of 100 examples, between the static control's 9 percent and the tied-depth model's 6 percent.
Mean evaluation loss was 1.923, better than the static control's 1.949 but worse than direct tied depth's 1.882.
Final training loss was 1.937, slightly worse than 1.933 for the static control and 1.924 for direct tied depth.
The ordinary and OOD-`N` profiles again failed their first certification rung, so the result remains `Max T <1` and `OOD N Max T <1`.

This result does not support the hypothesis that Equation 3 and Equation 4 alone turn a shallow tied transition into a learned modular-squaring state machine.
The modest loss improvement over the static model without an exact-match improvement again indicates better marginal digit probabilities rather than reliable sequence computation.
The experiment also exposes the cost of the more faithful mechanism: the extra late block and approximately 281,000 fusion and block parameters nearly doubled evaluation time while producing no depth gain.
The single seed and low exact-success counts do not resolve small differences among the three accuracy scores, but the complete certification failure rules out meaningful systematic extrapolation in this setting.

The remaining mismatch with literal T²MLR is structural rather than an omitted module.
The paper receives a new token representation at every recurrent step and trains with dense next-token supervision, whereas this adaptation repeatedly combines a fixed encoded prompt with its cache and receives supervision only on the final residue digits.
With only one early block and one middle block, the cached state may also lack the abstract structure that the paper obtains before and across a substantially deeper recurrent span.
Zero-initialized global gates stabilize the initial network but leave only 80 updates to activate and shape the recurrent path.
Consequently, this run tests an equation-faithful latent-step adaptation of T²MLR, not the paper's original token-temporal architecture or its language-modeling claims.

## Implementation hypotheses

The highest-priority hypothesis is that a dedicated persistent residue register fused into a tied latent transition will extrapolate across `T` better than the current untied two-layer encoder.
The second hypothesis is that zero-initialized gated fusion will stabilize exact recurrent training better than an unconditional additive cache.
The third hypothesis is that keeping `N` and `T` immutable while updating only residue and progress channels will improve `OOD N Max T` relative to recurrently rewriting the entire state.
The fourth hypothesis is that exact unrolling will outperform Jacobi approximation per unit of H100 wall-clock at the repository's short sequence lengths and modest target horizons.
The fifth hypothesis is that cache ablation will reveal whether any observed depth gain actually depends on iterative state rather than a stronger static encoder.

## Expected failure modes

The main failure mode is the recurrence-axis mismatch, where a token-level implementation receives too few recurrent updates to represent `T` squarings.
A tied transition may memorize fixed-modulus mappings for training depths without learning modular squaring, producing high familiar-depth accuracy and immediate failure at `T = 6`.
Final-answer-only supervision may not identify the intended intermediate residue trajectory, especially when many latent algorithms fit the small E1 dataset.
A learned halting gate may interpolate among `T = 1, 2, 3` but fail to count to unseen depths.
Repeated RMS normalization may erase magnitude or digit structure needed for exact arithmetic.
The fusion scalars may remain near zero during the current 80-step schedule, making the treatment effectively identical to the control.
Exact unrolling may reduce the number of optimizer steps enough that the recurrent architecture loses under the fixed training-time budget.
Decimal token decoding may dominate exact-match errors even when the latent residue is approximately correct.
The model may learn a modulus-specific cycle for E1 and fail immediately on unseen `N`.
Additional parameters or repeated compute may create an apparent gain that vanishes after parameter- and wall-clock matching.

## Unresolved questions

It is unclear whether the paper's middle-layer advantage survives multi-seed evaluation and training-compute matching on reasoning tasks.
It is unclear whether recurrence helps because it preserves a semantically meaningful state, improves gradient routing, adds effective training depth, or combines all three.
It is unclear how much of the retrofit gain remains after matching the added parameters and optimization compute.
The correct interpretation of the prefill timing multipliers in Appendix Table 8 requires clarification.
The detachment condition printed in Appendix Algorithm 1 conflicts with the released implementation and should be corrected before reproducing the approximation literally.
For One Layer Deeper, the central open question is whether final-answer supervision alone can make a tied transition converge to one modular-squaring step.
The competition also needs a precise policy for deriving or learning the update count from decimal `T` without hard-coding the target algorithm.
The best state representation for exact residue arithmetic under changes in modulus bit width remains unresolved.
