# One Layer Deeper

1. Serial problem solving / serial computation.
   - Universal transformer / looped transformer; latent-reasoning model, recurrent models, neural GPU.
     - Residual stream / short connections.
     - Treat stopping as a learned part.
     - Keep it general.
   - Task / cryptography: [Time-lock Puzzles and Timed-release Crypto](https://people.csail.mit.edu/rivest/pubs/RSW96.pdf); [Verifiable Delay Functions](https://theory.stanford.edu/~dabo/abstracts/VDF.html); [Generically Speeding-Up Repeated Squaring is Equivalent to Factoring](https://eprint.iacr.org/2020/812); [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://openreview.net/forum?id=3EWTEy9MTM).
2. Generalization beyond training depth.
   - Dynamic depth: linear growth, log(n) growth.
   - Static depth: chain-of-thought but no intermediate results.
   - Recurrence / latent depth: [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983); [Universal Transformers](https://arxiv.org/abs/1807.03819); [Deep Equilibrium Models](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html); [Can You Learn an Algorithm?](https://proceedings.neurips.cc/paper/2021/hash/3501672ebc68a5524629080e3ef60aef-Abstract.html); [PonderNet](https://arxiv.org/abs/2107.05407); [End-to-end Algorithm Synthesis with Recurrent Networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html); [Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647); [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416); [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171); [Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822); [Stability and Generalization in Looped Transformers](https://arxiv.org/abs/2604.15259); [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping](https://arxiv.org/abs/2606.29983).
   - Depth / shortcuts: [Theoretical Limitations of Self-Attention in Neural Sequence Models](https://aclanthology.org/2020.tacl-1.11/); [Transformers Learn Shortcuts to Automata](https://openreview.net/forum?id=De4FYqjFueZ); [A Little Depth Goes a Long Way](https://papers.nips.cc/paper_files/paper/2025/hash/88dd7aa6979e352fda7c4952ca8eac59-Abstract-Conference.html).
   - Arithmetic generalization: [Grokking](https://arxiv.org/abs/2201.02177); [The Slingshot Mechanism](https://arxiv.org/abs/2206.04817); [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217); [Grokked Transformers are Implicit Reasoners](https://arxiv.org/abs/2405.15071).
3. Exact computation (token-level).
   - Algorithm learning: [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761); [The CLRS Algorithmic Reasoning Benchmark](https://proceedings.mlr.press/v162/velickovic22a.html); [Learning Iterative Reasoning through Energy Minimization](https://proceedings.mlr.press/v162/du22d.html).
4. Algorithm-optimizer co-design given x, y, z constraints.
   - 500M parameters.
   - 1 hour H100.
   - Depth vs. training signal.
   - Optimization / budgets: [Budgeted Training](https://arxiv.org/abs/1905.04753); [ReZero is All You Need](https://proceedings.mlr.press/v161/bachlechner21a.html); [DeepNet](https://arxiv.org/abs/2203.00555); [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179); [Mixture-of-Depths](https://arxiv.org/abs/2404.02258).
5. Representation.
   - Representation / model family: [Neural GPUs Learn Algorithms](https://research.google/pubs/neural-gpus-learn-algorithms/); [Extensions and Limitations of the Neural GPU](https://arxiv.org/abs/1611.00736); [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399).

6. Output.
   - Not autoregressive: carry logic, etc. must happen across model depth / global attention.
   - Token-level probability vs. whole-output probability (cross-entropy loss).
   - Loss should include stability of previous rows (?).
