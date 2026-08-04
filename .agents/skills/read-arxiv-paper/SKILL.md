---
name: read-arxiv-paper
description: Read and analyze an arXiv paper from an arXiv abstract, PDF, or source URL by downloading its TeX source, following the LaTeX document graph, and connecting its findings to the One Layer Deeper competition. Use when asked to read, explain, summarize, critique, or extract implementation and experiment ideas from an arXiv paper for this repository.
---

# Read an arXiv Paper

Use the paper's TeX source as the primary artifact rather than extracting text from the PDF.
Treat the paper's claims, your interpretation, and repository-specific proposals as separate categories.

## Normalize the arXiv URL

Extract the arXiv identifier from an `abs`, `pdf`, or `src` URL and preserve any explicit version suffix such as `v2`.
Convert the URL to `https://arxiv.org/src/{arxiv_id}`.
Reject unrelated URLs instead of guessing an identifier.

## Download and unpack the source

Use `${XDG_CACHE_HOME:-$HOME/.cache}/one-layer-deeper/arxiv/{cache_id}` as the cache directory, replacing `/` with `_` in legacy arXiv identifiers only for the local directory name.
Download the source archive only when it is not already cached.
Use a temporary archive name and rename it only after a successful download.
Inspect archive paths before extraction and reject absolute paths or paths containing `..`.
Extract the archive into a version-specific source directory without overwriting a different paper version.
If arXiv does not provide TeX source, state that constraint and fall back to the PDF.

## Reconstruct the paper

Locate the root TeX file by looking for `\documentclass` and `\begin{document}` rather than assuming it is named `main.tex`.
Follow `\input`, `\include`, and equivalent commands in document order.
Read the abstract, main argument, methods, experiments, tables, figure captions, ablations, appendices, and limitations.
Inspect referenced figures when they carry information that the TeX text or caption does not fully express.
Use the rendered PDF only when necessary to resolve source ordering, equations, tables, or figures.

## Evaluate the paper

Identify the precise problem, claimed contribution, mechanism, assumptions, and strongest supporting evidence.
Check whether the experiments isolate the claimed mechanism and whether the baselines, ablations, and evaluation distributions support the conclusions.
Record negative results, boundary conditions, and credible alternative explanations.
Explain the core intuition without notation first, then introduce only the notation required for precision.
Do not present an extrapolation from the paper as a result reported by the authors.

## Connect it to this repository

Read `README.md` and `submission.py` before making repository-specific recommendations.
Inspect `scripts/` when the paper affects evaluation, training, or submission mechanics.
Relate relevant ideas to repeated modular squaring, systematic generalization to unseen `T`, generalization to unseen `N`, tied recurrent transitions, working-state design, halting or progress tracking, token-level decoding, the fixed model-state limit, and the H100 time budget.
Map architectural and implementation proposals onto concrete embeddings, recurrent state, tied transitions, timing signals, readouts, losses, schedules, or constants in `submission.py`.
Say explicitly when a paper has weak or no direct relevance to the competition.
Do not modify the model or submit an experiment unless the user separately asks for implementation or submission.

## Preserve requested findings

Do not create a separate research artifact unless the user requests one.
Use the user's path when provided; otherwise write the result to `summary_{tag}.md` at the repository root.
Do not recreate `resources/`.
Choose a short descriptive `tag` and never overwrite an existing note.
Use one sentence per source line and follow the repository's Markdown style.
Include the paper title, authors, arXiv identifier and version, source URL, and the date read.
Organize the note around these questions:

1. What does the paper actually claim, and what evidence supports or limits that claim?
2. What is the mechanism and its intuitive explanation?
3. Which parts matter for learned iterative computation and depth or modulus generalization in One Layer Deeper?
4. How could the idea map onto `submission.py` within the competition constraints?
5. What is the smallest decisive experiment, and what outcome would confirm or falsify the hypothesis?

End with concrete implementation hypotheses, expected failure modes, and unresolved questions.
Reference paper sections, figures, tables, and equations precisely enough that a reader can verify the analysis.
