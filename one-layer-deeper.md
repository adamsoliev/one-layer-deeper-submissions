# [One Layer Deeper](https://blog.tilderesearch.com/blog/one-layer-deeper)

how to think longer vs transformer/CoT
* looped layers
    * universal transformers/adaptive computation time
    * deep equilibrium models
    * deep thinking models
* recurrent state updates
* attractor-style models
* other forms of latent computation

questions
* can the architecture represent the required computation?
* can optimization discover the correct computation?
* can that co-design generalize beyond training depth?

benchmark
* inherently serial
* no shortcut
* can't be memorized
* testing depth > training depth

ask
* model | optimizer | learning rate scheduler | loss function

high-level evaluator flow
```text
_train()
  model.train()
  loop:
    optimizer.zero_grad()
    logits, auxiliary = model(input_ids, attention_mask)
    valid_logits, valid_labels = align_and_filter(logits, labels, target_positions)
    loss = submission.training_loss(valid_logits, valid_labels, auxiliary)
    loss.backward()
    clip_gradients()
    optimizer.step()
    scheduler.step()  # if supplied

_evaluate()
  model.eval()
  with no_grad():
    for batch:
      logits, _ = model(input_ids, attention_mask)
      valid_logits, valid_labels = align_and_filter(logits, labels, target_positions)
      loss = cross_entropy(valid_logits, valid_labels)  # evaluator-owned
      accumulate loss weighted by number of valid tokens
      accumulate exact whole-answer accuracy
  average_loss = weighted_loss_sum / valid_token_count
```

## Scope

The motivating limitation is not that every Transformer has exactly 100 layers, but that its serial computational depth is fixed at training time and is usually only tens to low hundreds of layers.

This makes a standard forward pass highly parallel, but it also makes computations whose required number of sequential steps grows with the input difficult to represent or extrapolate.

Chain-of-thought partly escapes the fixed-depth limit by putting intermediate state into the token stream, so every generated reasoning token supplies another full model evaluation.

The escape hatch is expensive because conventional decoding is autoregressive: a model cannot compute token \(t+1\) until token \(t\) exists.

The literature below distinguishes three different goals that are often conflated: adding serial depth inside the model, generating several output tokens in parallel, and merely reducing the cost of each still-sequential autoregressive step.

## PhD theses

These entries require the dissertation as a whole, or at least a substantial chapter, to address fixed-depth Transformer computation, recurrent or adaptive-depth alternatives, or the latency imposed by serial autoregressive decoding.

The final subsection is deliberately qualified because speculative decoding preserves the target model's causal distribution, recurrent language models can still generate one token at a time, and speech recognition is not text-only language modeling.

### Direct diagnoses of fixed depth, expressivity, and length extrapolation

- [A Theory of the Computational Power and Limitations of Language Modeling Architectures](https://lambdaviking.com/assets/pdf/dissertation.pdf) (William C. Merrill, New York University, 2025) characterizes bounded-depth Transformers through circuit complexity, identifies state tracking and other sequential computations outside their efficient parallel regime, and studies chain-of-thought, looped Transformers, and recurrent architectures as ways to trade parallelism for expressivity.
- [Transformers as Recognizers and Transducers](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1997321) (Lena Strobl, Umeå University, 2025) gives formal-language and circuit-complexity accounts of what constant-depth, finite-precision Transformers can recognize and compute.
- [Guiding Machine Learning Design With Insights From Simple Sandboxes](https://www.ml.cmu.edu/research/phd-dissertation-pdfs/bingbinl_phd_mld_2024.pdf) (Bingbin Liu, Carnegie Mellon University, 2024) uses automata-learning sandboxes to show how shallow non-recurrent Transformers can rely on brittle shortcuts and fail to extrapolate to longer sequences.
- [Toward Length-Extrapolatable Transformers](https://kilthub.cmu.edu/articles/thesis/Toward_Length-Extrapolatable_Transformers/25933873) (Ta-Chung Chi, Carnegie Mellon University, 2024) diagnoses failures on formal languages and develops RegularGPT, which combines weight sharing, adaptive depth, and sliding-window attention.
- [Systematic Generalization in Connectionist Models](https://susi.usi.ch/usi/documents/326205) (Róbert Csordás, Università della Svizzera italiana, 2023) studies the poor systematic and length generalization of standard architectures and develops recurrent, connectionist alternatives including the Neural Data Router.

### Recurrent, adaptive, and implicit internal computation

- [Equilibrium Approaches to Modern Deep Learning](https://www.ml.cmu.edu/research/phd-dissertation-pdfs/phd_thesis_shaojie_bai.pdf) (Shaojie Bai, Carnegie Mellon University, 2022) replaces explicit layer stacks with fixed-point computation, including a deep-equilibrium Transformer whose input-dependent solver iterations provide effectively unbounded and adaptive depth.
- [Next-Gen AI: Advancing Watermarking, Algorithm Synthesis and Diverse Generative Strategies](https://drum.lib.umd.edu/items/790ba140-6f86-472d-a52f-7b1370ec9128) (Arpit Bansal, University of Maryland, 2025) devotes chapters to recurrent algorithm learners and looped Transformers with weight tying, input injection, and variable recurrence for improved length extrapolation.
- [Enhanced Architectures and Optimization Methods for Efficient Language Modeling](https://infoscience.epfl.ch/entities/publication/c93b22d8-1580-4b4c-b10a-8e5a0006ebff) (Amirkeivan Mohtashami, EPFL, 2025) develops DenseFormer and CoTFormer variants that reuse a shallow model recurrently, including adaptive per-token recurrence that varies computational cost.
- [Accelerating Large Language Model Inference via Early-Exiting Algorithms](https://library.kaist.ac.kr/search/detail/view.do?bibCtrlNo=1142797&flag=dissertation) (Sangmin Bae, KAIST, 2025) combines synchronized parallel decoding, deep parameter sharing, and learned per-token recursion depths to make adaptive internal computation practical in batched inference.
- [Efficient and Expressive Architectures for Language Modeling](https://www.csail.mit.edu/event/thesis-defense-songlin-yang-efficient-and-expressive-architectures-language-modeling) (Songlin Yang, Massachusetts Institute of Technology, 2025 dissertation defense) treats quadratic Transformer cost and limited state tracking together through gated linear attention, delta-rule recurrence, and hybrid recurrent-attention architectures.
- [Making the Most of Your Model: Methods for Finetuning and Applying Pretrained Transformers](https://arxiv.org/abs/2408.16241) (Davis Yoshida, Toyota Technological Institute at Chicago, 2023) includes methods that add recurrence to pretrained Transformers and RUM-SUNDAE, a non-autoregressive iterative generator built from a masked language model.

### Parallel and non-autoregressive generation

- [Efficient Neural Machine Translation](https://hub.hku.hk/handle/10722/265405) (Jiatao Gu, University of Hong Kong, 2018) introduces non-autoregressive neural machine translation to replace token-by-token decoding with parallel generation.
- [Structured Neural Models and Structured Decoding for Natural Language Processing](https://digicoll.lib.berkeley.edu/record/221995?v=pdf) (Mitchell Stern, University of California, Berkeley, 2020) develops blockwise parallel decoding and the Insertion Transformer, whose out-of-order generation can reduce linear left-to-right decoding depth to logarithmic time.
- [Latent Variable Models and Iterative Refinement for Non-Autoregressive Neural Machine Translation](https://cs.nyu.edu/media/publications/jason-lee-thesis.pdf) (Jason Lee, New York University, 2021) develops latent-variable and iterative-refinement models that trade a small number of parallel refinement rounds for substantially faster translation.
- [Neural Structured Prediction Using Iterative Refinement with Applications to Text and Molecule Generation](https://cs.nyu.edu/media/publications/Elman_Mansimov_Thesis_Jan2021.pdf) (Elman Mansimov, New York University, 2021) replaces fixed left-to-right generation with learned-order iterative refinement that updates all or selected positions in parallel.
- [Non-Autoregressive Neural Machine Translation](https://dspace.cuni.cz/handle/20.500.11956/174086) (Jindřich Helcl, Charles University, 2022) systematically studies the speed-quality tradeoff of fully parallel and connectionist-temporal-classification-based translation.
- [Towards Efficient Neural Machine Translation](https://www.lti.cs.cmu.edu/people/alumni/alumni-thesis/kong-xiang-thesis.pdf) (Xiang Kong, Carnegie Mellon University, 2022) develops semi-autoregressive local translation and fully non-autoregressive models to parallelize decoding.
- [Towards Efficient Universal Neural Machine Translation](https://era.ed.ac.uk/handle/1842/39298) (Biao Zhang, University of Edinburgh, 2022) combines lightweight recurrent modeling with interleaved bidirectional semi-autoregressive decoding and other efficiency techniques.
- [Towards Faster Inference of Transformers: Strategies for Accelerating Decoding Processes](https://ink.library.smu.edu.sg/etd_coll/613/) (Cunxiao Du, Singapore Management University, 2024) studies fully non-autoregressive Transformers, speculative decoding, and constant-attention Markov autoregressive models as complementary attacks on decoding latency.
- [Representation Modeling Based Language GANs: From Autoregressive Models to Non-Autoregressive Models](https://theses.lib.polyu.edu.hk/handle/200/13412) (Da Ren, Hong Kong Polytechnic University, 2024) develops adversarial non-autoregressive text and caption generators motivated by the high latency of autoregressive decoding.
- [Improving Efficient Inference for Large Language Models: Non-Autoregressive Models and Speculative Decoding Approaches](https://tdr.lib.ntu.edu.tw/handle/123456789/96573?locale=en) (Shen-Sian Syu, National Taiwan University, 2025) combines a fully non-autoregressive conditional generation model with hierarchical speculative decoding.

### Qualified recurrent successors and autoregressive accelerators

- [Modeling Sequences with Structured State Spaces](https://search.worldcat.org/title/Modeling-sequences-with-structured-state-spaces/oclc/1382654057) (Albert Gu, Stanford University, 2023) develops structured state-space models with recurrent and convolutional views that improve long-sequence processing and autoregressive inference, although generation remains causally ordered.
- [Understanding Language Models: Optimization, Architecture, and Emergent Abilities](https://deepblue.lib.umich.edu/items/a6319375-103f-408a-bcfb-045b699844a7) (Yingcong Li, University of Michigan, 2025) analyzes softmax attention against recurrent linear-attention and state-space alternatives, including how depth and chain-of-thought affect expressivity.
- [Efficient Learning for Large Language Models](https://uwspace.uwaterloo.ca/items/38dbf804-a257-4b45-bbb1-b8fd3fed1cc6) (Hossein Rajabzadeh, University of Waterloo, 2026) develops depth-aware submodels, dynamic layer substitution during autoregressive decoding, and cross-layer attention sharing to reduce per-step latency and cache cost without removing serial token dependence.
- [Hardware-Aware Software Optimizations for and with Machine Learning](https://lists.cs.princeton.edu/hyperkitty/list/talks%40lists.cs.princeton.edu/thread/WKD4TJ23IERME4GEWHSOTAERDWBNW3RY/) (Rohan Baskar Prabhakar, Princeton University, 2026 final public oral) contributes the inference-oriented Kraken Transformer and a learned filter that reduces speculative-verification work, while leaving causal output ordering intact.
- [xLSTM: Recurrent Neural Network Architectures for Scalable and Efficient Large Language Models](https://research.jku.at/en/publications/xlstm-recurrent-neural-network-architectures-for-scalable-and-eff/) (Maximilian Beck, Johannes Kepler University Linz, 2026) presents scalable recurrent language-model architectures with linear memory and favorable reported inference scaling, while retaining causal token generation.
- [Novel Generative and Language Model Architectures With Applications](https://uknowledge.uky.edu/math_etds/121/) (Edison Mucllari, University of Kentucky, 2025) includes the Compact Recurrent Transformer, which couples shallow local-context Transformers to a persistent recurrent memory to avoid global quadratic attention over long sequences.
- [Efficient Uncertainty Estimation and Sequence Modelling](https://www.repository.cam.ac.uk/items/23be6c3b-b52d-4209-ab25-1d2732f9fd26) (Yassir Fathullah, University of Cambridge, 2025) includes structured recurrent state-space models and a non-autoregressive proxy that bypasses full autoregressive decoding for sequence-level attribute estimation.
- [Efficient LLM System with Speculative Decoding](https://escholarship.org/uc/item/2cm0c15n) (Xiaoxuan Liu, University of California, Berkeley, 2025) develops online and hardware-aware speculative-decoding systems that verify multiple proposed tokens in parallel without changing the target model's output distribution.
- [Improving Inference Efficiency of Hybrid State Space Models through Speculative Decoding](https://escholarship.org/uc/item/3qh1c99h) (Yangchao Wu, University of California, Los Angeles, 2026) adapts tree-based speculative decoding to state-space and hybrid recurrent-attention models, accelerating but not eliminating causal generation.
- [Resource-Aware Distributed Machine Learning: Unified Approaches for Private and Efficient On-Device Intelligence](https://elischolar.library.yale.edu/gsas_dissertations/1546/) (Yeshwanth Venkatesha, Yale University, 2025) contains a chapter on edge-cloud speculative decoding that distributes draft generation and parallel verification across resource-constrained devices and servers.
- [Efficiently Scaling Machine Learning Systems Across Heterogeneous Resources](https://deepblue.lib.umich.edu/items/8f6ae1be-91f2-4860-b4c3-6c6e167c44c6) (Shuowei Jin, University of Michigan, 2025) includes Plato, a semantic dependency-graph and pipelined-execution framework that parallelizes independent parts of an LLM answer instead of decoding the whole response as one chain.
- [Computation-Communication Co-Design for Efficient Deployment of Deep Learning and Vision-Language Models on Resource-Constrained IoT Devices](https://www.cics.umass.edu/events/phd-thesis-defense-jin-huang) (Jin Huang, University of Massachusetts Amherst, 2026 thesis defense) contains a vision-language chapter combining visual-token pruning, cloud assistance, and speculative decoding, so it is a cross-modal systems match rather than a text-model successor.
- [Improving the Accuracy and Inference Efficiency for Low-Resource Automatic Speech Recognition](https://escholarship.org/uc/item/9281v84q) (Ruchao Fan, University of California, Los Angeles, 2024) develops a single-step non-autoregressive Transformer recognizer for parallel speech decoding, making it a modality-qualified rather than text-only match.
- [Advancing End-to-End Speech AI with Knowledge Transfer](https://etd.ohiolink.edu/acprod/odb_etd/r/etd/search/10?clear=10&p10_accession_num=osu174040399846635) (Vishal Sunder, Ohio State University, 2025) contains a non-autoregressive multimodal speech-to-text and text-to-speech model that improves its parallel output through iterative refinement.
- [Fast and Low-Latency End-to-End Speech Recognition and Translation](https://repository.kulib.kyoto-u.ac.jp/items/35b826c1-747e-44d4-9745-ba44a75cdcaa) (Hirofumi Inaguma, Kyoto University, 2021) develops non-autoregressive Mask-CTC and related systems for low-latency speech recognition and translation, again outside the text-only setting.

## 1. Direct diagnoses of fixed computational depth

### Surveys and framing

- [Barriers to Discrete Reasoning with Transformers: A Survey Across Depth, Exactness, and Bandwidth](https://aclanthology.org/2026.eacl-long.87/) synthesizes circuit-complexity, approximation, and communication-complexity barriers to exact symbolic computation.
- [A Survey on Latent Reasoning](https://arxiv.org/abs/2507.06203) organizes activation recurrence, hidden-state propagation, trace compression, and diffusion-based latent reasoning.
- [What comes after transformers? — A selective survey connecting ideas in deep learning](https://arxiv.org/abs/2408.00386) surveys potentially disruptive alternatives as well as incremental improvements.
- [The Serial Scaling Hypothesis](https://serial-scaling-hypothesis.github.io/) is a technical essay separating a fixed Transformer's constant-depth forward pass from the unbounded serial computation supplied by recurrent or autoregressive execution.

### Circuit-complexity and communication lower bounds

- [Theoretical Limitations of Self-Attention in Neural Sequence Models](https://aclanthology.org/2020.tacl-1.11/) shows that fixed-size self-attention cannot recognize several periodic or hierarchical languages unless its depth or number of heads grows with input length.
- [On the Ability and Limitations of Transformers to Recognize Formal Languages](https://aclanthology.org/2020.emnlp-main.576/) empirically connects formal-language failures to architectural choices and positional encodings.
- [Formal Language Recognition by Hard Attention Transformers: Perspectives from Circuit Complexity](https://aclanthology.org/2022.tacl-1.46/) places hard-attention variants in constant-depth circuit classes.
- [Saturated Transformers are Constant-Depth Threshold Circuits](https://aclanthology.org/2022.tacl-1.49/) characterizes saturated-attention Transformers through constant-depth threshold circuits.
- [The Parallelism Tradeoff: Limitations of Log-Precision Transformers](https://aclanthology.org/2023.tacl-1.31/) argues that the same constant-depth parallelism that makes Transformers efficient also restricts their computational power.
- [A Logic for Expressing Log-Precision Transformers](https://papers.nips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html) characterizes log-precision Transformer classifiers with first-order logic plus majority quantifiers.
- [Representational Strengths and Limitations of Transformers](https://papers.neurips.cc/paper_files/paper/2023/hash/73bf692447f174984f30499ec9b20e04-Abstract-Conference.html) gives conditional separations between constant-depth Transformers and recurrent models.
- [On Limitations of the Transformer Architecture](https://arxiv.org/abs/2402.08164) uses communication complexity to prove limitations on function composition, even with infinite precision.
- [Transformers, Parallel Computation, and Logarithmic Depth](https://proceedings.mlr.press/v235/sanford24a.html) relates attention layers to rounds of massively parallel computation and shows why logarithmic, rather than constant, depth changes what can be computed.
- [Separations in Representational Capabilities of Transformers and Recurrent Architectures](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3f630b20b7b3ac76d3a0016fe29b6dc0-Abstract-Conference.html) proves tasks for which recurrent architectures are more parameter-efficient than Transformers.
- [Theoretical limitations of multi-layer Transformer](https://arxiv.org/abs/2412.02975) gives an unconditional depth separation for iterated function composition and formalizes how an additional layer or chain-of-thought can close the gap.
- [Circuit Complexity Bounds for RoPE-based Transformer Architecture](https://aclanthology.org/2025.emnlp-main.561/) shows that rotary positional embeddings do not remove constant-depth bounds for arithmetic and Boolean formula evaluation.
- [Limits of Deep Learning: Sequence Modeling through the Lens of Complexity Theory](https://proceedings.iclr.cc/paper_files/paper/2025/hash/62868cc2fc1eb5cdf321d05b4b88510c-Abstract-Conference.html) extends compositional-complexity analysis to both Transformers and state-space models.

### Scope qualifiers and counterresults

- [Attention is Turing-Complete](https://jmlr.org/papers/v22/20-302.html) proves universality under hard attention and unbounded autoregressive decoding, illustrating why Turing completeness does not contradict fixed-depth or finite-precision lower bounds.
- [Self-Attention Networks Can Process Bounded Hierarchical Languages](https://aclanthology.org/2021.acl-long.292/) constructs Transformers for bounded-depth Dyck languages and makes the required number of layers explicit.
- [Overcoming a Theoretical Limitation of Self-Attention](https://arxiv.org/abs/2202.12172) demonstrates that some apparent formal-language limits disappear with carefully chosen normalization, positional signals, or constructions.
- [Ask, and it shall be given: On the Turing completeness of prompting](https://arxiv.org/abs/2411.01992) proves that a finite Transformer can be universal when its prompt and generated sequence provide unbounded external state and time.

### Chain-of-thought as externalized computational depth

- [Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective](https://arxiv.org/abs/2305.15408) shows that bounded-depth Transformers need super-polynomial size for some arithmetic problems without intermediate steps, while chain-of-thought supplies effective depth.
- [Compositional Reasoning with Transformers, RNNs, and Chain of Thought](https://arxiv.org/abs/2503.01544) shows that model depth, width, recurrent steps, or chain-of-thought length must grow with compositional depth.

### Empirical and mechanistic evidence

- [Faith and Fate: Limits of Transformers on Compositionality](https://proceedings.neurips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract.html) finds rapidly worsening autoregressive accuracy as a compositional task requires more sequential operations.
- [Transformers Learn Shortcuts to Automata](https://openreview.net/forum?id=De4FYqjFueZ) finds that Transformers often fit bounded-length shortcuts instead of learning the recurrent computation that would extrapolate.
- [Why are Sensitive Functions Hard for Transformers?](https://aclanthology.org/2024.acl-long.800/) explains persistent failures on parity and related tasks through a low-sensitivity learning bias.
- [Transformers need glasses! Information over-squashing in language tasks](https://arxiv.org/abs/2406.04267) proves a last-token representational collapse that can destroy information needed for counting and copying.
- [A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](https://papers.nips.cc/paper_files/paper/2025/hash/88dd7aa6979e352fda7c4952ca8eac59-Abstract-Conference.html) shows that small increases in recurrent or effective depth can produce large algorithmic-generalization gains.
- [The Impact of Depth on Compositional Generalization in Transformer Language Models](https://aclanthology.org/2024.naacl-long.402/) is useful counterevidence: greater physical depth does not uniformly improve compositional generalization, so recurrence and training signal matter in addition to layer count.
- [The Depth Delusion: Why Transformers Should Be Wider, Not Deeper](https://arxiv.org/abs/2601.20994) is additional counterevidence that simply stacking more distinct layers can worsen language-modeling efficiency beyond a width-dependent critical depth.

## 2. Architectures that put serial computation inside the model

### Foundations for adaptive and recurrent depth

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) couples a recurrent controller to differentiable external memory so computation length and storage are not fixed by a feed-forward stack.
- [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279) composes learned subprograms through a recurrent core and explicit scratchpad.
- [Hybrid Computing Using a Neural Network with Dynamic External Memory](https://www.nature.com/articles/nature20101) introduces the Differentiable Neural Computer and demonstrates learned read–write memory on graph and planning problems.
- [Neural GPUs Learn Algorithms](https://research.google/pubs/neural-gpus-learn-algorithms/) applies the same learned transition repeatedly and extrapolates some algorithms beyond training lengths.
- [Extensions and Limitations of the Neural GPU](https://arxiv.org/abs/1611.00736) documents both the promise and optimization fragility of deep recurrent computation.
- [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) learns how many recurrent updates each input requires.
- [Universal Transformers](https://arxiv.org/abs/1807.03819) replaces a fixed stack with a recurrently applied Transformer block and optional adaptive halting.
- [Deep Equilibrium Models](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html) solves directly for the fixed point of an effectively infinite weight-tied network.
- [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) learns a probabilistic halting policy over recurrent computation steps.
- [Learning Iterative Reasoning through Energy Minimization](https://proceedings.mlr.press/v162/du22d.html) treats reasoning as repeated refinement toward a low-energy solution.
- [End-to-end Algorithm Synthesis with Recurrent Networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html) trains recurrent networks to synthesize and execute algorithms with length generalization.
- [Transformer Working Memory Enables Regular Language Reasoning and Natural Language Length Extrapolation](https://aclanthology.org/2023.findings-emnlp.397/) combines weight tying, adaptive depth, and sliding attention in RegularGPT.
- [Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks](https://proceedings.neurips.cc/paper/2021/hash/3501672ebc68a5524629080e3ef60aef-Abstract.html) studies when recurrent networks trained on easy instances extrapolate to harder ones.
- [Adaptivity and Modularity for Efficient Generalization Over Task Complexity](https://openreview.net/forum?id=tI3eqOV6Yt) combines adaptive Universal-Transformer depth with hypernetworks that generate step-specific functions.
- [Addressing Some Limitations of Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402) lets a current token's lower layers consume high-level representations from earlier tokens, turning autoregressive time into recurrent feedback.
- [Change of Thought: Adaptive Test-Time Computation](https://arxiv.org/abs/2507.13569) iteratively refines self-attention to a fixed point so computation can scale with input difficulty without emitting thought tokens.

### Looped and depth-recurrent Transformers

- [Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647) repeatedly applies a shared Transformer block and studies extrapolation beyond the training loop count.
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) scales computation by recurrently unrolling a latent block instead of emitting more reasoning tokens.
- [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416) analyzes the computational power of looped latent reasoning.
- [The Recurrent Transformer: Greater Effective Depth and Efficient Decoding](https://arxiv.org/abs/2604.21215) directly diagnoses Transformers as temporally shallow and adds layerwise recurrent memory without increasing standard autoregressive decoding cost.
- [Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization](https://arxiv.org/abs/2603.21676) compares internal recurrent depth with longer textual reasoning traces.
- [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822) studies how looped computation and intermediate supervision affect reasoning and extrapolation.
- [Stability and Generalization in Looped Transformers](https://arxiv.org/abs/2604.15259) analyzes the dynamical stability required to run more loops at inference than during training.
- [Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](https://arxiv.org/abs/2605.26733) regularizes the recurrent Jacobian so extra test-time iterations approach a stable fixed point instead of eventually collapsing.
- [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping](https://arxiv.org/abs/2606.29983) treats stopping as a learned stochastic component of recurrent-depth extrapolation.
- [LoopFormer: Elastic-Depth Looped Transformers for Lifelong Language Modeling](https://arxiv.org/abs/2602.11451) proposes a language model whose depth can be changed at training and inference.
- [SpiralFormer: Looped Transformers with Multi-Resolution Recursion](https://arxiv.org/abs/2602.11698) adds recursive computation at multiple temporal resolutions.
- [MeSH: Memory-as-State-Highways for Recursive Transformers](https://arxiv.org/abs/2510.07739) separates persistent memory from transient loop state and routes reads and writes differently at each recurrence.
- [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Thinking](https://openreview.net/forum?id=YtQtGsNr64) routes individual tokens through different numbers of applications of a shared Transformer stack.
- [Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves](https://arxiv.org/abs/2601.21582) adds attention over recurrent depth and sparse experts to relax the fixed-width state bottleneck.
- [LOTUS: Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779) processes blocks of latent thoughts in parallel to reduce the latency of a recurrent thought phase.
- [Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning](https://openreview.net/forum?id=9Pba4rcQbE) combines recurrent depth with routing so different tokens receive different amounts of test-time compute.
- [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://openreview.net/forum?id=Oq3Xblt0x1) studies how to convert an existing pretrained Transformer into a recurrent-depth model rather than pretraining one from scratch.
- [Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers](https://arxiv.org/abs/2606.04678) shows controllable recurrent depth and non-autoregressive output outside text-only language modeling.
- [Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning](https://arxiv.org/abs/2602.07845) replaces explicit action-chain reasoning with a weight-tied recurrent action head and convergence-based stopping.
- [Unlocking Out-of-Distribution Generalization in Transformers via Latent Space Reasoning](https://openreview.net/forum?id=8bFgEyRLrO) combines adaptive recurrence, state anchoring, algorithmic supervision, and explicit error correction on algorithmic tasks.
- [Turbo Connection: Reasoning as Information Flow from Higher to Lower Layers](https://arxiv.org/abs/2602.17993) feeds higher-layer state at token \(t\) into lower layers at token \(t+1\), increasing effective path depth with sequence length.
- [Latent Recurrent Transformer: Architecture Exploration, Training Strategies, and Scaling Behavior](https://arxiv.org/abs/2605.26797) reuses a high-level state from the preceding token as recurrent memory while retaining the standard cache interface.
- [The Context-Ready Transformer](https://arxiv.org/abs/2606.27538) pre-contextualizes each new token with a recurrent correction from the preceding position before it enters a shallow Transformer block.

### Continuous and latent thoughts

- [Think before you speak: Training Language Models With Pause Tokens](https://proceedings.iclr.cc/paper_files/paper/2024/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html) gives a model extra hidden-state positions before it must commit to an answer.
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) introduces Coconut, which feeds hidden states back as continuous thoughts instead of decoding every thought into language.
- [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074) distills explicit reasoning traces into a shorter sequence of continuous thoughts.
- [Expediting and Elevating Large Language Model Reasoning via Hidden Chain-of-Thought Decoding](https://arxiv.org/abs/2409.08561) compresses several textual reasoning steps into hidden-state transitions.
- [Continuous Chain of Thought Enables Parallel Exploration and Reasoning](https://arxiv.org/abs/2505.23648) uses continuous thoughts to represent several candidate reasoning traces at once.
- [Thoughtbubbles: An Unsupervised Method for Parallel Thinking in Latent Space](https://openreview.net/forum?id=pNpnqsn0Si) learns to fork and delete residual streams so hard tokens receive adaptive parallel computation during an ordinary language-modeling objective.
- [ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-Time Compute](https://arxiv.org/abs/2509.04475) generates isolated reasoning branches concurrently and then synthesizes them, trading thought width against sequential thought depth.
- [Parallel Continuous Chain-of-Thought with Jacobi Iteration](https://aclanthology.org/2025.emnlp-main.47/) updates several latent thought positions concurrently instead of computing the continuous thoughts sequentially.
- [Parallel Test-Time Scaling for Latent Reasoning Models](https://aclanthology.org/2026.acl-long.2069/) introduces stochastic continuous trajectories and aggregation methods for sampling latent reasoning paths in parallel.
- [Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models](https://arxiv.org/abs/2510.14961) refines recurrent latent states while decoding new positions and reports substantial speedups without retraining the model.

### Non-Transformer recursive successors

- [Neural Computers](https://arxiv.org/abs/2604.06425) is a position and prototype paper arguing that the model itself should become a persistent learned runtime integrating computation, memory, and I/O.
- [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) uses coupled recurrent modules, iterative refinement, and adaptive halting to create substantial internal depth without a textual chain-of-thought.
- [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) argues that a very small recursively applied network can outperform much larger models on several exact reasoning tasks.
- [Continuous Thought Machines](https://arxiv.org/abs/2505.05522) makes the temporal dynamics of neurons part of the representation and uses adaptive internal recurrence.
- [Continuous Thought Machines: Machines That Learn to Think in Time](https://pub.sakana.ai/ctm/) is the authors' interactive technical report and demonstration.

### Critical and negative evidence

- [Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer](https://arxiv.org/abs/2507.02199) finds only limited evidence that additional recurrent depth implements an interpretable latent chain-of-thought.
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449) finds faithful intermediate computation on short tasks but partial paths and shortcuts on longer ones.
- [Tiny Autoregressive Recursive Models](https://arxiv.org/abs/2603.08082) finds no reliable benefit from the full hierarchical TRM mechanism under compute-matched causal language-modeling comparisons.
- [The Hidden Drivers of HRM's Performance on ARC-AGI](https://arcprize.org/blog/hrm-analysis) reproduces much of HRM's headline result but attributes the largest gain to outer-loop refinement rather than its claimed hierarchy and identifies weak cross-task transfer.

## 3. The autoregressive decoding bottleneck

### Surveys and direct diagnoses

- [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/abs/2508.08712) organizes autoregressive accelerators and genuinely non-autoregressive generators around the token-by-token bottleneck.
- [Awesome Parallel Text Generation](https://github.com/zhanglingzhe0820/Awesome-Parallel-Text-Generation) is the survey authors' living index of papers organized by generation mechanism.
- [Alternatives to Next Token Prediction in Text Generation: A Survey](https://arxiv.org/abs/2509.24435) covers multi-token prediction, plan-then-generate methods, latent reasoning, continuous generation, and non-Transformer architectures.
- [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574) reviews diffusion approaches and their speed-quality tradeoff.
- [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115) begins from the premise that ordinary autoregressive generation cannot parallelize across output positions.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) profiles incremental decoding as a sequential, memory-bandwidth-heavy workload.
- [On the Computational Complexity of Self-Attention](https://proceedings.mlr.press/v201/duman-keles23a.html) proves conditional quadratic-time lower bounds for exact and approximate self-attention, which compounds the serial output-position bottleneck at long context.

### Parallel and iterative non-autoregressive generation

- [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281) generates output positions in parallel using a latent fertility model.
- [Semi-Autoregressive Neural Machine Translation](https://aclanthology.org/D18-1044/) preserves dependencies between groups while producing several successive tokens within each group in parallel.
- [Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement](https://arxiv.org/abs/1802.06901) trades one left-to-right pass for a small number of parallel refinement rounds.
- [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://aclanthology.org/D19-1633/) repeatedly remasks low-confidence positions and fills the sequence in parallel.
- [Insertion Transformer: Flexible Sequence Generation via Insertion Operations](https://arxiv.org/abs/1902.03249) supports balanced generation orders that can reduce \(n\) sequential output steps to \(O(\log n)\) rounds.
- [Levenshtein Transformer](https://arxiv.org/abs/1905.11006) generates through parallel insertion and deletion edits.
- [UT5: Pretraining Non-Autoregressive T5 with Unrolled Denoising](https://arxiv.org/abs/2311.08552) trains an encoder-decoder model for iterative non-autoregressive refinement.
- [Breaking the Autoregressive Chain: Hyper-Parallel Decoding for Efficient LLM-Based Attribute Value Extraction](https://aclanthology.org/2026.findings-acl.1832/) exploits conditional independence between structured outputs to decode many positions out of order with an unchanged LLM.
- [Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation](https://arxiv.org/abs/2307.15337) first generates an answer outline and then expands independent outline points concurrently.
- [Skeleton-of-Thought: Parallel Decoding Speeds Up and Improves LLM Output](https://www.microsoft.com/en-us/research/blog/skeleton-of-thought-parallel-decoding-speeds-up-and-improves-llm-output/) is the authors' accessible technical account and demonstration.

### Diffusion and masked-diffusion language models

- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) provides a general discrete diffusion framework used by later text models.
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) performs iterative denoising in continuous embedding space.
- [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) scales diffusion language modeling beyond narrow generation tasks.
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) introduces SEDD and improves likelihood-based discrete diffusion.
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) shows that masked diffusion can be competitive with a comparatively simple training recipe.
- [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891) converts pretrained autoregressive models into diffusion models.
- [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1b3d1e2cf53bb28cabd801bd58b3521-Abstract-Conference.html) argues that global denoising can learn difficult subgoals that left-to-right objectives neglect.
- [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) introduces LLaDA, an eight-billion-parameter masked-diffusion language model.
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) uses autoregression between blocks and bidirectional diffusion within each block.
- [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://arxiv.org/abs/2512.13586) uses diffusion to plan weakly dependent slots and fills those slots autoregressively but concurrently.
- [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487) develops a general-purpose seven-billion-parameter diffusion language model.
- [Improved Large Language Diffusion Models](https://arxiv.org/abs/2606.25331) revisits masking, training, and decoding choices for large diffusion models.
- [Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding](https://arxiv.org/abs/2607.01775) models unordered sets of token-position assignments to avoid committing to a left-to-right order.
- [Recursive Scaling in Masked Diffusion Models](https://openreview.net/forum?id=QZwBhjogcY) adds recurrent depth within each denoising step, jointly exposing recursion and diffusion rounds as test-time compute axes.
- [Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive) Decoding](https://arxiv.org/abs/2602.23225) is important negative evidence that many fast diffusion models learn autoregressive-like denoising trajectories.
- [The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models](https://arxiv.org/abs/2601.15165) finds that arbitrary-order reasoning can avoid uncertain but necessary steps and collapse exploration.
- [Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow](https://aclanthology.org/2026.findings-acl.357/) measures actual finalization order across models up to 100 billion parameters and finds weakened inter-token dependencies at high parallelism.
- [Accelerating Diffusion Large Language Models via Adaptive Parallel Decoding](https://openreview.net/forum?id=xwqTt26NJf) dynamically chooses how many positions to decode together.
- [Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192) combines blockwise causality, inter-block parallelism, and pipelining to demonstrate faster-than-autoregressive inference at matched scale.
- [Diffuse Thinking: Exploring Diffusion Language Models as Efficient Thought Proposers for Reasoning](https://aclanthology.org/2026.acl-long.1231/) uses a diffusion model to propose diverse thoughts in parallel and an autoregressive model to evaluate them.
- [Gemini Diffusion](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/) is an official demonstration of a production-scale model that refines several tokens concurrently.
- [Introducing Mercury](https://www.inceptionlabs.ai/blog/introducing-mercury) is an industry report on diffusion language models optimized for high-throughput code generation.

### Autoregressive-preserving accelerators

These methods reduce wall-clock latency, but they do not remove the causal dependency in the target distribution.

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) drafts several tokens cheaply and verifies them with the target model in parallel.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) gives an exact rejection-sampling construction for accepting draft tokens without changing the target distribution.
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) predicts and verifies a tree of future-token candidates.
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) drafts in feature space to improve acceptance.
- [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) parallelizes Jacobi-style fixed-point iteration while preserving autoregressive outputs.
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) trains auxiliary heads to predict several future tokens and uses them for faster decoding.
- [Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion](https://aclanthology.org/2025.naacl-long.601/) uses a diffusion model so both drafting and target-model verification can be parallelized.
- [Accelerating Gemini Nano Models on Pixel with Frozen Multi-Token Prediction](https://www.research.google/blog/accelerating-gemini-nano-models-on-pixel-with-frozen-multi-token-prediction/) reports a deployed multi-token drafter that verifies candidates in parallel while leaving the model's outputs unchanged.

### Cheaper recurrent decoding that remains autoregressive

These architectures reduce attention, cache, or per-token costs, but an autoregressive language model built from them still emits tokens sequentially.

- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) combines parallel training with constant-state recurrent inference.
- [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) proposes parallel, recurrent, and chunkwise-recurrent forms of the same retention mechanism.
- [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866) replaces attention with long convolutions whose sequence cost is subquadratic.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) uses input-dependent state-space dynamics and linear sequence scaling.
- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427) combines recurrent state with local attention.
- [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://arxiv.org/abs/2404.07839) scales the Griffin architecture and removes a growing global-attention cache.
- [RecurrentGemma](https://deepmind.google/models/gemma/recurrentgemma/) is the official model and deployment overview.
- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) revisits recurrent language modeling with exponential gating and matrix memories.
- [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) turns the hidden state update into a learned optimization problem.
- [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201) simplifies several modern recurrent architectures into common components.
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) adds a long-term neural memory that updates during inference.

## 4. Adjacent algorithmic-generalization literature

- [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761) frames neural execution of classical algorithms and the need to generalize across problem sizes.
- [The CLRS Algorithmic Reasoning Benchmark](https://proceedings.mlr.press/v162/velickovic22a.html) supplies supervised traces for evaluating neural algorithm execution.
- [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399) shows that representation choices can remove some arithmetic failures without replacing the architecture.
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) documents delayed generalization after memorization.
- [The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon](https://arxiv.org/abs/2206.04817) connects grokking dynamics to optimizer behavior.
- [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217) identifies internal algorithmic circuits before test accuracy improves.
- [Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization](https://arxiv.org/abs/2405.15071) studies when a Transformer internalizes reasoning that otherwise appears in an explicit trace.

## 5. Architecture–optimizer co-design

The working constraint is approximately 500 million parameters and one hour on one H100, with depth, stability, and training-signal density treated as coupled design variables.

- [Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints](https://arxiv.org/abs/1905.04753) studies optimization when the training horizon is a hard budget.
- [ReZero is All You Need: Fast Convergence at Large Depth](https://proceedings.mlr.press/v161/bachlechner21a.html) initializes residual branches to preserve trainability at extreme depth.
- [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) is a useful reminder that physical depth can exceed 100 layers when residual scaling is designed for it, although inference cost still grows with every layer.
- [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179) compares learned and hand-designed optimizers under standardized workloads.
- [Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models](https://arxiv.org/abs/2404.02258) routes only selected tokens through each block under a fixed compute budget.

## 6. Open design questions

- Can one weight-tied transition support linear, logarithmic, or learned growth in serial depth without destabilizing beyond its training horizon?
- Should halting be learned jointly with the transition, and should stopping be deterministic or stochastic?
- Can residual or short connections preserve a stable state while still allowing meaningful iteration?
- Can a latent recurrent phase carry reasoning across depth and reserve autoregressive generation for the final answer?
- Can several latent states or output positions be refined in parallel without collapsing to an autoregressive-like schedule?
- Should the loss supervise every token or state, only the whole output, or both?
- How should the objective penalize damage to already-correct partial states while still allowing revision?
- Which benchmark separates memorized finite-depth shortcuts from an algorithm that actually extrapolates in depth?
