# Evaluation Results

Offline IR evaluation of each ranking strategy against the WANDS relevance
judgments (480 queries, top-50 retrieved per query). Regenerate with `./scripts/run-eval.sh`.

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency (ms) |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.6694 | 0.8794 | 0.0613 | 0.2482 | 0.7923 | 3 |
| Semantic (bge-small-en-v1.5) | 0.6982 | 0.8900 | 0.0579 | 0.2288 | 0.7990 | 6 |
| Hybrid (RRF, k=40) | 0.7295 | 0.9202 | 0.0639 | 0.2502 | 0.8450 | 9 |
| Hybrid + Cross-Encoder Rerank | 0.7447 | 0.9037 | 0.0642 | 0.2502 | 0.8362 | 1519 |
| Neural Rerank (MLP) | 0.6809 | 0.8555 | 0.0613 | 0.2731 | 0.8021 | 33 |
| Learned Tower (fine-tuned, shared) | 0.6603 | 0.8559 | 0.0559 | 0.2284 | 0.7729 | 10 |

_Neural Rerank and Learned Tower each train on 80% of the 480 WANDS queries and are scored above on only the held-out 20% (see `search-eval/src/main/resources/neural-reranker-eval-queries.txt`) — their rows aren't on the same query count as the other four. Learned Tower is the shared-tower mode, which won a head-to-head against a two-tower model on this split — see TRAINING.md._

## RRF constant sweep

Fusion of the same lexical/semantic candidate lists at each k (see `results/rrf-sweep.csv`); the Hybrid row above uses whichever k scored highest on nDCG@10.

| k | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 |
|---|---|---|---|---|---|
| 10 | 0.7210 | 0.9236 | 0.0627 | 0.2502 | 0.8283 |
| 20 | 0.7277 | 0.9230 | 0.0637 | 0.2502 | 0.8388 |
| 40 | 0.7295 | 0.9203 | 0.0639 | 0.2502 | 0.8450 |
| 60 | 0.7293 | 0.9204 | 0.0639 | 0.2502 | 0.8450 |
| 100 | 0.7290 | 0.9198 | 0.0640 | 0.2502 | 0.8454 |
| 200 | 0.7282 | 0.9207 | 0.0639 | 0.2502 | 0.8450 |
