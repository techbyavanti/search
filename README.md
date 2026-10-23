# Semantic Product Search

A Java service for semantic product search over the [WANDS](https://github.com/wayfair/WANDS)
(Wayfair) furniture catalog of ~43,000 products. Seven ranking strategies —
from plain BM25 up to hybrid fusion, cross-encoder reranking, and two
custom-trained models — are each scored against 480 real queries and
~233,000 human relevance judgments, so every quality claim has a measured
number behind it.

> **[Live snapshot demo →](https://techbyavanti.github.io/search/)** — real,
> captured queries comparing all seven strategies side by side (a static
> GitHub Pages page, not a live backend — [run it yourself](HOWTO.md) for
> the real thing).

## Results at a glance

| Strategy | nDCG@10, all 480 queries | nDCG@10, 96 held-out queries | p95 latency |
|---|---|---|---|
| Lexical (BM25) | 0.6720 | 0.6845 | 3 ms |
| Semantic (bge-small, default BBQ HNSW) | 0.6958 | 0.7291 | 6 ms |
| Semantic HNSW (float, tuned) | 0.6973 | 0.7254 | 6 ms |
| Hybrid (RRF fusion) | 0.7322 | 0.7468 | 8 ms |
| **Hybrid + Cross-Encoder Rerank** | **0.7447** | **0.7798** | 224 ms |
| **Neural Rerank (LambdaRank MLP)** | — | **0.7730** | **10 ms** |
| Learned Tower (fine-tuned embeddings) | — | 0.6561 | 8 ms |

The trained strategies learn from the other 384 queries, so they are
scored only on the 96 held-out ones; that column compares all seven on
the same queries. Each stage of the main pipeline improves on the one
before it. The trained MLP reranker gets 79% of the cross-encoder's gain
over Hybrid at 1/22 of its latency. Tuning HNSW made approximate search
almost exact (0.986 recall@50 against brute force) but didn't move
nDCG@10, which shows the embedding model, not the index, limits semantic
quality. The fine-tuned embedding model scores below the off-the-shelf
one; the docs show by how much and why. Full metrics:
[RESULTS.md](RESULTS.md).

## What this project demonstrates

- **Search and ranking engineering** — BM25, dense kNN retrieval,
  hand-written Reciprocal Rank Fusion, and cross-encoder reranking, all
  behind one pluggable `SearchStrategy` interface.
- **Performance engineering** — parallel chunked cross-encoder inference
  cut reranker p95 from ~1.5 s to ~225 ms with no loss in nDCG@10 (A/B on
  the same index); a score cache serves repeated queries in under 30 ms;
  a query-embedding cache encodes each query once per request.
- **Vector search tuning** — an HNSW sweep (vector type, m,
  ef_construction, num_candidates) against exact brute-force kNN, picking
  the cheapest setting with ≥ 98% recall@50.
- **Evaluation discipline** — an offline IR harness (nDCG@10, MRR,
  Recall@10/50, Precision@10, p95 latency) that runs the *same* strategy
  code the API serves, a held-out query split for trained models, and a
  regression floor per strategy.
- **ML in production Java** — embedding, reranking, and trained models all
  run in-process through ONNX Runtime; no Python server anywhere.
- **Learning to rank** — an MLP reranker trained with a LambdaRank loss on
  17 features that Java exports with the same code it serves with (no
  training/serving skew), plus a fine-tuned embedding model
  (shared-encoder vs. two-tower comparison). Both export to ONNX and are
  served by the Java code.
- **Backend and delivery** — Spring Boot REST API with Swagger UI, a
  six-module Maven build on JDK 26, Docker Compose, and a CI pipeline with
  CVE scanning, SAST, secret scanning, image scanning, and an SBOM.
- **Clear reporting of results** — strategies that underperform are kept,
  measured, and explained, not hidden.

## Where to go

**HR / recruiters** — open the [live demo](https://techbyavanti.github.io/search/)
and read [What this project demonstrates](#what-this-project-demonstrates)
above. That takes about two minutes.

**Hiring managers** — read [Results at a glance](#results-at-a-glance), then
[ARCHITECTURE.md § Reading the results](ARCHITECTURE.md#reading-the-results)
for the cost/quality tradeoffs, and [CI.md](CI.md) for the security
posture.

**Technical interviewers** — start with [ARCHITECTURE.md](ARCHITECTURE.md)
(module layout, data flow, design decisions, implementation details), then
[TRAINING.md](TRAINING.md) for the two trained models. [HOWTO.md](HOWTO.md)
runs the whole stack locally, with a verification step after each stage.

## Tech stack

Java 26 · Spring Boot 4.1 · Elasticsearch · ONNX Runtime · Hugging Face
tokenizers (via DJL) · PyTorch (offline training only) · Maven · Docker
Compose · GitHub Actions · SonarQube · JaCoCo · Trivy · Semgrep · gitleaks ·
hadolint · Syft

## Docs

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, module, data-flow, class-design, and request-flow diagrams; the seven ranking strategies; how to read the results; key design decisions |
| [RESULTS.md](RESULTS.md) | Current evaluation numbers (regenerate with `./scripts/run-eval.sh`) |
| [TRAINING.md](TRAINING.md) | Setup, process, and results for the two custom-trained models |
| [CI.md](CI.md) | The CI pipeline and security posture (CVE/secret scanning, SAST, Dependabot) |
| [HOWTO.md](HOWTO.md) | Step-by-step local setup, with a verification step after each stage |
