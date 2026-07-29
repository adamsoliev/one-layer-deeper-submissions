# One Layer Deeper

1. Serial problem solving / serial computation.
2. Generalization beyond training depth.
   - Dynamic depth: linear growth, log(n) growth.
   - Static depth: chain-of-thought but no intermediate results.
3. Exact computation (token-level).
4. Algorithm-optimizer co-design given x, y, z constraints.
5. Representation / 7.1.
   - 500M parameters.
   - 1 hour H100.
   - Depth vs. training signal.
   - Universal transformer / looped transformer; latent-reasoning model, recurrent models, neural GPU.
     - Residual stream / short connections.
     - Treat stopping as a learned part.
     - Keep it general.

## Output

6. ~~Not autoregressive: carry logic, etc. must happen across model depth / global attention.~~
   - Not autoregressive.
   - Token-level probability vs. whole-output probability (cross-entropy loss).
   - Loss should include stability of previous rows (?).
