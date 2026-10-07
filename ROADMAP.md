# Roadmap

- [x] **M0 — Environment + scaffolding.** Toolchain installed (JDK 21, Maven,
      Colima/Docker), Maven reactor scaffolded, Elasticsearch + Kibana running
      via Docker Compose, WANDS dataset downloaded, `search-common` domain
      models + CSV parsing + unit tests.
- [x] **M1 — Lexical baseline + eval harness.** `LexicalSearchStrategy`, index
      mapping/creation, bulk ingestion of all ~43K products, full
      `search-eval` pipeline, first row of the results table.
- [x] **M2 — Embeddings + semantic search + eval.** `EmbeddingService`
      (ONNX + tokenizer, CLS-pool, normalize, BGE query-prefix handling),
      `dense_vector` field, `SemanticSearchStrategy`, second results row.
- [x] **M3 — Hybrid + RRF + eval.** `RrfFusionService`,
      `HybridRrfSearchStrategy`, sweep of the RRF constant, third results row.
- [x] **M4 — Cross-encoder rerank + eval.** `RerankerService`,
      `HybridRerankStrategy`, fourth results row, latency measurement.
- [x] **M5 — Polish.** Swagger UI, `/api/search/compare` side-by-side
      endpoint, minimal demo surface, Dockerized `search-api`, architecture
      diagram, full write-up.
- **M6+ — ongoing.**
  - [x] CI-enforced eval — see [CI.md](CI.md).
  - [x] Feature-based neural reranker (`NeuralRerankStrategy`,
        `RerankFeatureBuilder`, `ProductFeatureCache`, fifth results row) —
        trained via `scripts/train-neural-reranker.sh`, evaluated on a
        held-out 20% query split. Didn't beat Hybrid on nDCG@10; kept as a
        documented latency/quality tradeoff point, not a replacement for
        the cross-encoder — see [WRITEUP.md](WRITEUP.md#m6-feature-based-neural-reranker--a-real-negative-result).
  - [x] Fine-tuned embedding tower comparison (`LearnedTowerSearchStrategy`,
        sixth results row) — `bge-small-en-v1.5` fine-tuned on WANDS in two
        configurations (shared encoder vs. true two-tower), compared via
        `training/train_embedding_towers.py` + the real eval harness on a
        held-out split. Shared-tower won the head-to-head but neither beat
        the off-the-shelf pretrained model — see [WRITEUP.md](WRITEUP.md#m6-continued-fine-tuned-two-tower-vs-shared-tower-embeddings).
  - [ ] Server-side RRF comparison (Elasticsearch's/OpenSearch's built-in
        fusion vs. the hand-rolled version, now that both are provably
        correct), query understanding (spelling/synonyms), a
        pairwise/listwise ranking loss for the neural reranker instead of
        pointwise MSE, more training data for the tower fine-tune (the
        two-tower-vs-shared-tower result above might well flip with enough
        of it), and observability.

## See also

- [WRITEUP.md](WRITEUP.md) — the narrative behind each completed milestone
- [ARCHITECTURE.md](ARCHITECTURE.md) — current module structure and design decisions
- [RESULTS.md](RESULTS.md) — the numbers each milestone produced
