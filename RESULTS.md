# Evaluation Results

Offline IR evaluation of each ranking strategy against the WANDS relevance
judgments (480 queries, top-50 retrieved per query). Regenerate with `./scripts/run-eval.sh`.

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency (ms) |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.6720 | 0.8827 | 0.0615 | 0.2477 | 0.7933 | 3 |
| Semantic (bge-small-en-v1.5) | 0.6958 | 0.8851 | 0.0575 | 0.2289 | 0.7969 | 6 |
| Semantic HNSW (float, tuned) | 0.6973 | 0.8851 | 0.0579 | 0.2307 | 0.7979 | 6 |
| Hybrid (RRF, k=40) | 0.7322 | 0.9244 | 0.0632 | 0.2503 | 0.8429 | 8 |
| Hybrid + Cross-Encoder Rerank | 0.7447 | 0.9043 | 0.0641 | 0.2503 | 0.8367 | 224 |
| Neural Rerank (MLP) | 0.7730 | 0.9361 | 0.0687 | 0.2723 | 0.9073 | 10 |
| Learned Tower (fine-tuned, shared) | 0.6561 | 0.8597 | 0.0558 | 0.2265 | 0.7687 | 8 |

_Neural Rerank and Learned Tower each train on 80% of the 480 WANDS queries and are scored above on only the held-out 20% (see `search-eval/src/main/resources/neural-reranker-eval-queries.txt`) — their rows aren't on the same query count as the others. Learned Tower is the shared-tower mode, which won a head-to-head against a two-tower model on this split — see TRAINING.md._

## Same-query comparison (held-out 96 queries)

Every strategy on the same held-out queries, so the trained strategies can be compared with the others directly. These queries score higher than the full set for every strategy, so compare rows within this table, not with the table above.

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 |
|---|---|---|---|---|---|
| Lexical (BM25) | 0.6845 | 0.8968 | 0.0623 | 0.2666 | 0.8115 |
| Semantic (bge-small-en-v1.5) | 0.7291 | 0.9129 | 0.0625 | 0.2486 | 0.8385 |
| Semantic HNSW (float, tuned) | 0.7254 | 0.9083 | 0.0632 | 0.2516 | 0.8354 |
| Hybrid (RRF, k=40) | 0.7468 | 0.9330 | 0.0661 | 0.2723 | 0.8688 |
| Hybrid + Cross-Encoder Rerank | 0.7798 | 0.9407 | 0.0687 | 0.2723 | 0.8896 |
| Neural Rerank (MLP) | 0.7730 | 0.9361 | 0.0687 | 0.2723 | 0.9073 |
| Learned Tower (fine-tuned, shared) | 0.6561 | 0.8597 | 0.0558 | 0.2265 | 0.7687 |

## RRF constant sweep

Fusion of the same lexical/semantic candidate lists at each k (see `results/rrf-sweep.csv`); the Hybrid row above uses whichever k scored highest on nDCG@10.

| k | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 |
|---|---|---|---|---|---|
| 10 | 0.7232 | 0.9266 | 0.0622 | 0.2503 | 0.8258 |
| 20 | 0.7298 | 0.9260 | 0.0630 | 0.2503 | 0.8356 |
| 40 | 0.7322 | 0.9244 | 0.0632 | 0.2503 | 0.8429 |
| 60 | 0.7319 | 0.9245 | 0.0631 | 0.2503 | 0.8425 |
| 100 | 0.7318 | 0.9242 | 0.0632 | 0.2503 | 0.8429 |
| 200 | 0.7317 | 0.9238 | 0.0632 | 0.2503 | 0.8429 |
