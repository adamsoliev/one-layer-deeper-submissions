# Thoughtbubbles, Length Generalization, Depth Embeddings, and Zero-Order Training

This note analyzes the proposal “Thoughtbubbles + length-generalization training + layerwise depth embeddings” and the alternative “François Chaubard’s zero-order approach” for One Layer Deeper.

The first source is *Thoughtbubbles: an Unsupervised Method for Parallel Thinking in Latent Space* by Houjun Liu, Shikhar Murty, Christopher D. Manning, and Róbert Csordás.

The paper is arXiv:2510.00219v2, and its source is https://arxiv.org/src/2510.00219v2.

The second source is *Scaling Recurrent Neural Networks to a Billion Parameters with Zero-Order Optimization* by François Chaubard and Mykel Kochenderfer.

The paper is arXiv:2505.17852v1, and its source is https://arxiv.org/src/2505.17852v1.

The papers and repository were read on 2026-07-25.

“LengthGen training” does not identify a canonical method or paper in the quoted shorthand, so this note interprets it as training across varied computation lengths or recurrence depths to promote extrapolation.

## 1. Claims and evidence

Thoughtbubbles modifies a decoder-only Transformer so selected token residual streams can be duplicated or deleted between layers.

Each stream receives learned keep and fork scores, cumulative scores are propagated across layers, and a top-k operation enforces a fixed stream budget.

The cumulative scores attenuate attention and residual writes so the language-modeling loss can train the allocation policy without explicit reasoning traces or allocation labels.

At the output, the full method decodes each fork separately and mixes its token probabilities, while the largest model uses a cheaper approximation that first averages residuals and then decodes.

The paper reports better perplexity and zero-shot evaluations than standard decoder-only and non-adaptive extra-compute baselines across models from 150 million to 1.9 billion parameters.

Its strongest evidence concerns adaptive parallel computation in language modeling, not serial function composition or exact arithmetic.

Its learned bubbles commonly form around high-entropy or decisive tokens, which supports the allocation interpretation but does not establish that branching can extrapolate to unseen recurrent depths.

Chaubard and Kochenderfer replace backpropagation through time with central-difference random gradient estimation, abbreviated CD-RGE.

For each random Rademacher direction uᵢ, CD-RGE measures the loss at θ + εuᵢ and θ − εuᵢ and uses their difference to estimate a directional gradient.

The estimator is ĝ = (1/m) ∑ᵢ ([L(θ + εuᵢ) − L(θ − εuᵢ)] / 2ε)uᵢ.

Because it only needs inference-mode forward evaluations and scalar losses, it avoids retaining the recurrent trajectory for backward propagation.

The paper reports that 96 or 512 perturbations per update can match or exceed BPTT on small transduction tasks and Penn Treebank language modeling, and it trains recurrent models up to 1.1 billion parameters where BPTT does not fit on the comparison hardware.

The claimed wall-clock benefit depends on distributing many independent perturbation evaluations and using optimized recurrent inference kernels.

The language-model experiment uses Penn Treebank with sequence length 10, and the transduction experiments use small synthetic tasks, so the evidence does not directly establish an advantage for Transformer recurrence, exact modular arithmetic, or a single-H100 60-second budget.

## 2. Mechanisms

Thoughtbubbles increases parallel latent workspace rather than serial algorithmic depth.

A difficult token can own several related hidden states, each state can evolve through subsequent layers, and a learned budget controller later prunes or merges those states.

In a recurrent-depth model, an adaptation could maintain a set Bₖ = {zₖ¹, zₖ², …} of working streams at iteration k, score possible keeps and forks, apply the same tied transition to the survivors, and merge them before decoding.

Length-generalization training would expose the tied transition to more than one unroll length so the model is pressured to learn a reusable update instead of a separate shortcut for each familiar depth.

For One Layer Deeper, “length” should mean the requested composition depth T or the number of tied recurrent applications, not the tokenized input length.

A layerwise depth embedding supplies iteration information to a tied transition, for example zₖ₊₁ = Fθ(zₖ + d(k,T)).

This breaks the symmetry between early, middle, and late recurrent steps and can represent progress or remaining work.

A learned lookup table dₖ is dangerous for extrapolation because entries beyond the training depths receive no useful gradient.

An extrapolatable encoding based on k, T, and T − k, such as sinusoidal features or bounded scalar features, is more compatible with evaluation at unseen T.

Zero-order training attacks a different problem.

It leaves the recurrent architecture largely unchanged and replaces exact reverse-mode gradients with a noisy gradient estimate obtained from many perturbed forward passes.

Its benefit is activation memory that does not grow with unroll length, while its main cost is roughly 2m model evaluations for m central-difference probes per optimizer update.

## 3. Relevance to One Layer Deeper

The current `submission.py` already has a tied middle Transformer block, a recurrent cache, and exact input-dependent unrolling up to T.

It trains on T ∈ {1, 2, 3} in E1 and evaluates the same tied transition at unseen T, including T = 6.

The model does not inject the current iteration index into the tied transition, so a depth or progress encoding is the most direct missing component from the first proposal.

Thoughtbubbles could replace the single recurrent cache with a bounded collection of specialized cache streams, but this would increase activation compute and require a precise fork, prune, and merge policy.

Repeated modular squaring has the serial dependency xₖ₊₁ = xₖ² mod N, so parallel bubbles cannot replace the required sequence of T state transitions.

The task also has one exact residue at every step rather than several plausible semantic hypotheses, which weakens the ordinary uncertainty-based rationale for branching.

Bubbles remain potentially useful if they specialize into stable computational roles, such as residue state, modulus context, decimal-digit interactions, or independent error-correcting representations.

The E1 evaluator controls the training examples, so the submission cannot introduce a new supervised distribution at larger T.

It can exploit the existing variation over T = 1, 2, and 3 and can choose architecture-level recurrence behavior, but it cannot obtain correct labels for extra composition depths from a custom outer training loop.

The official rules allow recurrence, adaptive computation, and depth curricula, but the evaluator fixes the data and the one-forward/one-backward loop.

The official rules also require all input-dependent computation to remain in the autograd graph with an unbroken gradient path from the loss to predictive parameters.

True CD-RGE therefore does not fit the current submission interface because it needs many loss-bearing perturbed forward evaluations under participant control instead of the evaluator’s one forward and one backward.

Even if the interface allowed it, 96 to 512 perturbations per update would be a poor fit for E1’s 60-second single-H100 budget unless extremely aggressive batching and a much smaller model changed the cost balance.

## 4. Mapping onto `submission.py`

The encoder can remain the token and position embedding followed by the first Transformer block.

The recurrent state can become a fixed-width tensor with a small bubble axis rather than a single `recurrent_cache`.

A score head can assign keep and fork logits to each bubble, and a fixed top-k budget can keep tensor shapes predictable.

The same `blocks[1]` transition should remain tied across all recurrent iterations and all bubbles.

A functional depth encoding d(k,T) can be injected into each active bubble before `fusion` or before the recurrent Transformer block.

The decoder should merge bubbles using normalized learned scores and then apply the existing final block and output projection.

The architecture should preserve exactly T serial recurrent updates, because adding parallel streams is not a substitute for composition depth.

CD-RGE cannot be implemented faithfully by changing `build_optimizer` alone because an ordinary PyTorch optimizer receives gradients after the evaluator has already executed its fixed forward and backward.

## 5. Smallest decisive experiment

The clean control should preserve the current width, recurrent block, optimizer, training steps, and exact-T unrolling while adding only an extrapolatable depth encoding.

That experiment tests whether iteration awareness helps familiar-depth accuracy and T = 6 extrapolation without confounding the result with branching compute.

A second treatment can add two fixed bubble streams with learned soft merge weights but no discrete top-k controller.

This tests whether parallel working states help before paying the complexity and gradient-estimation cost of paper-faithful dynamic forking.

The hypothesis is supported only if ordinary accuracy improves and the consecutive depth certification extends beyond the control without reducing the number of completed training steps.

It is falsified if bubbles merely increase familiar-depth fitting while leaving or worsening unseen-depth accuracy.

## Implementation hypotheses

The depth encoding is the highest-value and lowest-risk component of the first proposal.

Variable-depth training on the evaluator-provided T values is useful, but a separate embedding lookup for each depth is likely to encourage memorization.

Soft fixed-count bubbles are a better first test than discrete dynamic forking because the arithmetic task is deterministic and the wall-clock budget is tight.

Zero-order optimization is not a valid submission strategy under the current evaluator contract and should not be mixed into the first experiment.

Expected failure modes include untrained depth embeddings at unseen T, bubbles collapsing to identical states, bubbles specializing by familiar T rather than computational role, excessive recurrence compute reducing optimizer steps, and learned merge weights suppressing useful branches.

The unresolved question is whether parallel latent streams can learn stable arithmetic roles that improve a fundamentally serial transition, rather than acting as an expensive ensemble.

## 6. Soft-Thoughtbubbles E1 result

Submission 8 implemented the first proposal as a two-stream differentiable approximation to Thoughtbubbles.

The treatment retained the T²MLR gated-cache backbone from submission 7, duplicated the recurrent working state into two learned bubble identities, allowed attention across the flattened bubble streams, accumulated a learned score for each stream, attenuated updates using the resulting softmax weights, and merged the streams before the final Transformer block.

Every recurrent application also received a functional six-feature depth encoding derived from the current iteration, requested T, remaining progress, and sinusoidal progress.

This functional encoding avoided untrained per-depth lookup entries at the unseen evaluation depths.

The evaluator-provided mixture T ∈ {1, 2, 3} supplied the length-generalization training distribution, and the same middle Transformer block remained tied across every recurrent application.

The official validator accepted the source, and local checks confirmed finite outputs through the 64-step evaluation path and nonzero gradients for the bubble embeddings, score head, depth projection, and tied recurrent block.

The hosted E1 run succeeded as submission `ce357172-8f06-4be9-a224-e482f8bd0eee` from source commit `a408a7d`.

The run completed all 80 optimizer steps in 23.93 seconds and evaluated in 6.53 seconds.

The E1 manifest instantiated 701,315 model-state elements and 1,402,687 optimizer-state elements after the first update.

Mean exact accuracy was 5.50 percent, the highest of the first eight repository submissions.

The static width-128 control scored 5.17 percent, direct tied depth scored 4.33 percent, and the T²MLR gated-cache model scored 4.17 percent.

The apparent improvement over the static control was only 0.33 percentage points and came from one additional familiar-depth success.

The Thoughtbubbles model solved 3 of 150 familiar-depth test examples compared with 2 for the static control.

Both models solved the same 9 of 100 ordinary OOD examples at T = 6.

Consequently, this run provides no evidence that bubbles or depth encoding improved ordinary depth extrapolation.

The mean evaluation loss was 1.937, compared with 1.949 for the static control, 1.882 for direct tied depth, and 1.923 for T²MLR.

The Thoughtbubbles model therefore achieved the best exact-match score without achieving the best token-level likelihood.

On the fixed-N certification ladder it solved 0 of 38 examples at T = 1, 2 of 38 at T = 2, and 0 of 38 at every rung from T = 4 through T = 64.

It consequently reported `Max T <1`.

On the unseen-N ladder it solved between 0 and 4 of 512 examples at each rung and also reported `OOD N Max T <1`.

These sparse, non-monotonic successes are inconsistent with a reusable repeated-squaring transition.

The result falsifies the strong proposal-1 hypothesis in this implementation: soft parallel streams plus explicit functional progress information did not create systematic depth generalization under final-answer-only supervision.

It does not isolate the individual contributions of bubbles and depth encoding because both were added together, and it cannot verify bubble specialization because the evaluator does not return trained parameters or routing statistics.

The next controlled experiment should add only the functional depth encoding to the direct tied-depth model.

That ablation would determine whether the encoding supplies useful progress information or whether the aggregate score change came from the parallel-stream ensemble and sampling noise.
