# Model training

This project trains two of its own models on WANDS, on top of the
off-the-shelf `bge-small-en-v1.5` (embedding) and `ms-marco-MiniLM-L-6-v2`
(cross-encoder reranker) models that `scripts/download-models.sh` fetches
pre-exported. Both training pipelines live under `training/`. Python is used
only here, offline; all serving runs in Java through ONNX Runtime.

- **Track A** — a small MLP reranker over cheap, cached features, trained
  with a LambdaRank loss (`NeuralRerankStrategy`).
- **Track B** — `bge-small-en-v1.5` fine-tuned on WANDS, comparing a shared
  encoder against a two-tower (separate query/product encoders)
  architecture (`LearnedTowerSearchStrategy`).

Both are wired into `search-eval`, `search-api`, and the demo. See
[ARCHITECTURE.md](ARCHITECTURE.md#reading-the-results) for what the results
mean and [HOWTO.md](HOWTO.md) for where training fits into local setup.

## Setup

Both tracks share one Python virtual environment. Each wrapper script
creates `training/.venv` if it doesn't exist, runs `pip install -r
training/requirements.txt`, and invokes the corresponding Python script:

```bash
./scripts/train-neural-reranker.sh          # Track A
./scripts/train-embedding-tower.sh shared   # Track B
```

`training/.venv` and everything under `models/` are gitignored — generated
artifacts, not source.

`training/requirements.txt`:

```
numpy>=2.1
torch>=2.6
onnx>=1.17
onnxruntime>=1.20
tokenizers>=0.20
transformers>=4.46,<5
huggingface_hub>=0.26
```

Version constraints:
- **`transformers` is pinned below 5.** In 5.x, `BertModel.forward()` is
  not compatible with the legacy TorchScript ONNX tracer
  (`torch.onnx.export(..., dynamo=False)`); export fails with
  `got multiple values for argument 'use_cache'`.
- **`torch.onnx.export(..., dynamo=False)`** is used on both tracks. The
  newer `torch.export`-based exporter needs the optional `onnxscript`
  package, which is not in requirements; the legacy tracer is enough for
  these small, control-flow-free forward passes.

Prerequisites:
- Elasticsearch running with the full catalog ingested (see
  [HOWTO.md](HOWTO.md) steps 0 and 3). Track A exports its features from
  the live index; Track B's runbook ingests and evaluates against it.
- For Track B, **network access** to download the `BAAI/bge-small-en-v1.5`
  PyTorch checkpoint from Hugging Face. This is separate from the ONNX
  export in `models/`: ONNX Runtime is inference-only, and fine-tuning needs
  real weights to backpropagate through.
- Stop other ONNX/JVM processes (for example a running `search-api`) before
  a CPU-bound training run. They compete for CPU and make training appear to
  hang.

## Track A: feature-based neural reranker

### What it trains

A small MLP — `Linear(17, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 1)`
— that reorders Hybrid's top-50 candidates from 17 cheap, mostly-cached
features per (query, candidate) pair. It replaces a cross-encoder forward
pass with a few small matrix multiplications.

### One feature implementation

`RerankFeatureBuilder.java` is the only implementation of the features.
`RerankFeatureExportCli` runs Hybrid for every WANDS query and writes each
top-50 candidate's feature vector and grade to
`models/neural-reranker/training-features.tsv`, through the same code
`NeuralRerankStrategy` serves with. Python reads that file and only trains.
So the training pools, the RRF constant, and every feature value are the
ones served. `NeuralRerankStrategy` checks at startup that the model's
input width equals `RerankFeatureBuilder.FEATURE_COUNT`.

Features are relative to the candidate pool where raw values are not
comparable between queries (BM25 scores, for example):

| # | Feature | Meaning |
|---|---|---|
| 0 | `rrf_score_relative` | RRF score ÷ the pool's best RRF score |
| 1 | `hybrid_reciprocal_rank` | 1 ÷ position in Hybrid's order |
| 2 | `lexical_reciprocal_rank` | 1 ÷ BM25 rank (0 if BM25 did not retrieve it) |
| 3 | `lexical_score_relative` | BM25 score ÷ the pool's best BM25 score |
| 4 | `semantic_reciprocal_rank` | 1 ÷ kNN rank (0 if kNN did not retrieve it) |
| 5 | `cosine_similarity` | Query · product embedding (both L2-normalized) |
| 6 | `cosine_gap_to_best` | Cosine minus the pool's best cosine |
| 7–10 | `name_overlap`, `class_overlap`, `category_overlap`, `description_overlap` | Fraction of query tokens found in each field |
| 11 | `query_coverage` | Fraction of query tokens found in any field |
| 12 | `name_contains_query` | 1 if the query tokens appear in order in the name |
| 13 | `query_token_count` | Number of query tokens |
| 14 | `average_rating` | Defaults to `4.5301` (catalog mean) when missing |
| 15 | `log1p_review_count` | 0 when missing |
| 16 | `has_rating` | 1 if the product has a rating |

Text features compare lowercased tokens with simple plural folding
("chairs" → "chair").

### Loss: LambdaRank

For each pair of candidates in one query's pool where one has a higher
grade, the loss is the pairwise logistic loss `log(1 + exp(-(s_i - s_j)))`,
weighted by how much swapping the two would change that query's nDCG
(`|ΔnDCG_ij|`). This trains the ranking order directly, and puts the most
weight on mistakes near the top of the list. `--loss mse` trains a
pointwise regression on the grade instead, for comparison.

### Process

1. `RerankFeatureExportCli` writes 24,000 rows (480 queries × 50
   candidates). Unlabeled candidates get grade 0, the same gain the eval
   gives them. Each row is tagged `train` or `held_out` from the shared
   query split (`training/query_split.py`, seed 42); the trainer checks
   that the split matches the eval's held-out file.
2. The 384 training queries are split again: 327 to fit, 57 for
   validation. Validation nDCG@10 picks the epoch (early stopping). The
   96 held-out queries are never used for any choice.
3. Feature mean and standard deviation from the fit rows are stored in the
   model, so the exported graph standardizes its own input.
4. Adam (lr `1e-3`, weight decay `1e-4`), 16 queries per step, up to 300
   epochs with patience 40.
5. Export to `models/neural-reranker/model.onnx`, then check that the ONNX
   output matches PyTorch (max difference ~5e-7).
6. Write `training-metadata.json`: features, split sizes, best epoch,
   validation scores, and permutation feature importance.

### How to run

```bash
./scripts/train-neural-reranker.sh             # LambdaRank (served)
./scripts/train-neural-reranker.sh --loss mse  # pointwise baseline, for comparison
```

**Required** before `search-api` starts — its `NeuralRerankerService` bean
fails without `models/neural-reranker/model.onnx`, like the
`EmbeddingService`/`RerankerService` beans fail without their model dirs.
The whole run takes about a minute on a laptop CPU.

### Track A results

All rows are nDCG@10 on the same 96 held-out queries (see the same-query
table in [RESULTS.md](RESULTS.md)):

| Ranking | nDCG@10 | p95 latency |
|---|---|---|
| Hybrid order (no rerank) | 0.7468 | 8 ms |
| MLP, pointwise MSE loss | 0.7639 | — |
| **MLP, LambdaRank loss (served)** | **0.7730** | **10 ms** |
| Cross-encoder rerank | 0.7798 | 224 ms |

The MLP gets 79% of the cross-encoder's nDCG@10 gain over Hybrid
(+0.0262 of +0.0330) at 1/22 of its latency, and the best Precision@10 of
any strategy (0.9073). The Python training script and the Java eval report
the same held-out nDCG@10 (0.7730), which confirms that training and
serving use identical features.

Most of the gain comes from the features and training data, not the loss:
the MSE model already reaches 0.7639. LambdaRank adds +0.0091.

Permutation importance (validation nDCG@10 drop when one feature is
shuffled) ranks `name_overlap` (0.051), `query_coverage` (0.025), and
`cosine_gap_to_best` (0.019) highest. Text match against the product name
and full query coverage carry most of the signal that RRF's rank-only
fusion loses.

## Track B: fine-tuned two-tower vs. shared-tower embeddings

### What it trains

`SemanticSearchStrategy` uses `bge-small-en-v1.5` exactly as published.
Track B fine-tunes it on WANDS's own relevance judgments and compares two
architectures: **separate** encoders for queries and products (two-tower)
against one **shared** encoder for both. Both candidates are scored through
the real `search-eval`/`TowerComparisonCli` code, not a Python-side proxy
metric.

Both modes fine-tune the pretrained `BAAI/bge-small-en-v1.5` transformer
with a pairwise margin-ranking loss:

```
query text   → tokenize → query tower   → CLS-pool + L2-normalize → query embedding
product text → tokenize → product tower → CLS-pool + L2-normalize → product embedding
```

- `--mode shared`: query tower and product tower are the *same* model
  instance/weights.
- `--mode two-tower`: two independently trained copies of the pretrained
  model, one per role.

Loss: for each `(query, higher-graded product, lower-graded product)`
triplet built from WANDS's within-query grade differences (Exact vs.
Partial vs. Irrelevant), `MarginRankingLoss` pushes
`cosine(query, higher)` above `cosine(query, lower)` by a margin of 0.1.

### No new Java inference code

A fine-tuned tower is exported to ONNX with the **same tensor contract**
(`input_ids`/`attention_mask`/`token_type_ids` → `last_hidden_state`,
CLS-pooled + L2-normalized by the *caller*, not in the graph) that
`EmbeddingService.java` expects. So a fine-tuned tower is just another
model directory `EmbeddingService` loads. Two-tower mode uses two
`EmbeddingService` instances (one per model dir); shared-tower mode uses one
instance for both `embedQuery`/`embedDocument`.

### Process (`training/train_embedding_towers.py --mode {shared,two-tower}`)

1. Load `dataset/product.csv` + `dataset/query.csv` + `dataset/label.csv`.
   Product text is built the same way `EmbeddingTextBuilder.java` builds it
   (name + class + normalized category hierarchy + truncated description),
   replicated in Python.
2. Use the same 80/20 query split as Track A (`training/query_split.py`,
   shared by both tracks, so both hold out the *same* queries).
3. Build `(query, higher-graded product, lower-graded product)` triplets
   from each training query's labeled products, capped at 15 per query
   (5,404 total from 384 training queries).
4. Load `BAAI/bge-small-en-v1.5` via `transformers.AutoModel.from_pretrained`.
   Tokenize with the local `models/bge-small-en-v1.5/tokenizer.json` via
   the `tokenizers` package (no second tokenizer download).
5. Train with `AdamW` (lr `2e-5`), batch size 16, 2 epochs, full
   fine-tuning (all weights, no frozen layers).
6. Export: `--mode shared` → `models/learned-shared-tower/{model.onnx,
   tokenizer.json}`. `--mode two-tower` → `models/learned-query-tower/`
   and `models/learned-product-tower/`. Same export-then-verify check as
   Track A (ONNX output matches torch output to within `2e-6`).

CPU-only fine-tuning is slow (low single-digit seconds per step). Smoke-test
timing on a small run first:

```bash
python3 training/train_embedding_towers.py --mode shared \
  --max-triplets 32 --batch-size 8 --epochs 1
```

### Comparison runbook

`products-mapping.json` has one `learned_embedding` `dense_vector` field.
Each candidate's embeddings are ingested into that field, evaluated, and
compared one candidate at a time, so the index holds no field for the
candidate that is not served:

```bash
# 1. Train a candidate
./scripts/train-embedding-tower.sh shared        # or: two-tower

# 2. Ingest its product-tower embeddings into learned_embedding
#    (overwrites the previously ingested candidate)
java -jar search-ingestion/target/search-ingestion.jar \
  --learned-model-dir models/learned-shared-tower    # or models/learned-product-tower

# 3. Score it on the held-out split
java -cp search-eval/target/search-eval.jar \
  com.avanti.search.eval.TowerComparisonCli models/learned-shared-tower "Learned Tower (shared)"
```

`IngestionCli --learned-model-dir` re-embeds and partial-updates (`_update`,
not a full reindex) only the `learned_embedding` field of every product in
the index. The pass is **resumable**: it checkpoints progress after every
batch to `.learned-embedding-checkpoint-<model-dir-name>.txt` (gitignored),
and a re-run continues from the checkpoint. The served index holds the
shared-tower embeddings, which `LearnedTowerSearchStrategy` queries.

### Track B results

| Candidate | nDCG@10 (held-out 96) | MRR | Recall@10 | Recall@50 | Precision@10 |
|---|---|---|---|---|---|
Selection run (`TowerComparisonCli`, each candidate's embeddings ingested
in turn):

| Candidate | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 |
|---|---|---|---|---|---|
| **Shared tower** | **0.6594** | 0.8753 | 0.0559 | 0.2279 | 0.7719 |
| Two-tower | 0.6468 | 0.8532 | 0.0551 | 0.2121 | 0.7667 |

Shared-tower outscores two-tower on every metric, and is served as
`LearnedTowerSearchStrategy` (`StrategyType.LEARNED_TOWER`). In the
current eval ([RESULTS.md](RESULTS.md), same-query table) it scores
0.6561 nDCG@10 against 0.7291 for the off-the-shelf model on the same
held-out queries, so fine-tuning did not beat the base model.
