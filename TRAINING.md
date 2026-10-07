# Model training

This project trains two of its own models on WANDS, on top of the
off-the-shelf `bge-small-en-v1.5` (embedding) and `ms-marco-MiniLM-L-6-v2`
(cross-encoder reranker) models `scripts/download-models.sh` fetches
pre-exported. Both training tracks live under `training/` (Python — the
only place this Java-first project uses Python, and only offline/one-time,
never for serving) and are otherwise unrelated:

- **Track A** — a small MLP reranker trained on cheap, cached features.
  Done, wired into `search-eval`/`search-api`/the demo.
- **Track B** — fine-tuning `bge-small-en-v1.5` itself on WANDS, comparing
  a shared encoder against a true two-tower (separate query/product
  encoders) architecture. In progress — see [Results](#track-b-results).

See [WRITEUP.md](WRITEUP.md) for the narrative (why these were tried, what
the numbers mean) and [HOWTO.md](HOWTO.md) for where these steps fit into
overall local setup. This document is the how-it-actually-works reference
for both training pipelines.

## Setup

Both tracks share one Python virtual environment, created and populated
automatically the first time you run either wrapper script:

```bash
./scripts/train-neural-reranker.sh          # Track A
./scripts/train-embedding-tower.sh shared   # Track B
```

Under the hood, each wrapper is a thin shell script
(`scripts/train-neural-reranker.sh`, `scripts/train-embedding-tower.sh`)
that creates `training/.venv` if it doesn't exist, `pip install -r
training/requirements.txt`, and invokes the corresponding Python script.
`training/.venv` and everything under `models/` are gitignored — generated
artifacts, not source, same as the two downloaded models.

`training/requirements.txt`:

```
requests>=2.32
numpy>=2.1
torch>=2.6
onnx>=1.17
onnxruntime>=1.20
tokenizers>=0.20
transformers>=4.46,<5
huggingface_hub>=0.26
```

Version notes (real problems hit while building this, not hypothetical):
- **`transformers` is pinned below 5** deliberately. `transformers` 5.x's
  `BertModel.forward()` signature collides with the legacy TorchScript
  ONNX tracer (`torch.onnx.export(..., dynamo=False)`) — see [Track B
  troubleshooting](#track-b-troubleshooting). 4.x works cleanly.
- **`torch.onnx.export(..., dynamo=False)`** is required on both tracks'
  export step. Recent `torch` defaults to the newer `torch.export`-based
  ONNX exporter, which pulls in the optional `onnxscript` package that
  isn't in requirements — the legacy tracer avoids that dependency and is
  plenty for these small, control-flow-free forward passes.
- Prerequisites: Elasticsearch running with the full catalog ingested
  (both tracks compute features/embeddings against the live index —
  see [HOWTO.md](HOWTO.md) steps 0 and 3), and for Track B specifically,
  **network access** to download the `BAAI/bge-small-en-v1.5` PyTorch
  checkpoint from Hugging Face (a separate download from the ONNX export
  already in `models/` — ONNX Runtime is inference-only, fine-tuning needs
  real weights to backprop through).

## Track A: feature-based neural reranker

### What it trains

A tiny MLP — `Linear(6, 8) → ReLU → Linear(8, 1)` — that reads 6 cheap,
mostly-cached features per (query, candidate) pair and outputs a single
relevance score, replacing a full cross-encoder forward pass for a much
faster (but, as it turns out, lower-quality — see
[Results](#track-a-results)) reranker.

**Feature order** (the contract between this script and
`RerankFeatureBuilder.java` — changing one side without the other silently
produces garbage scores):

| # | Feature | Source |
|---|---|---|
| 0 | `rrf_score` | The candidate's Hybrid RRF-fused score — already computed getting the candidate pool, zero extra cost |
| 1 | `cosine_similarity` | Dot product of query embedding · product embedding (both L2-normalized) — RRF fusion discards this real-valued signal (rank-only), so it's new information |
| 2 | `exact_term_overlap_fraction` | Fraction of query tokens found in `product_name` |
| 3 | `category_match` | 1.0 if any query token appears in `category_hierarchy`/`product_class` |
| 4 | `average_rating` | Defaults to `4.5301` (dataset-wide mean) when null |
| 5 | `log1p_review_count` | Defaults to 0 when null |

### Process (`training/train_neural_reranker.py`)

1. Load `dataset/query.csv` + `dataset/label.csv`; map WANDS grades
   (Exact/Partial/Irrelevant → 2/1/0).
2. Split query ids 80/20 (`training/query_split.py`, seed 42) — the
   held-out 20% is written to
   `search-eval/src/main/resources/neural-reranker-eval-queries.txt`, so
   `EvalCli` scores this strategy only on queries it never trained on.
3. For each training-split labeled (query, product) pair, compute the same
   6 features against the **live Elasticsearch index** — real BM25/kNN
   scores via the same queries `LexicalSearchStrategy`/
   `SemanticSearchStrategy` issue, RRF fusion via the same formula as
   `RrfFusionService`, and query embeddings via `onnxruntime` +
   `tokenizers` loading `models/bge-small-en-v1.5` directly (replicating
   `EmbeddingService`'s CLS-pool + L2-normalize + BGE query-prefix
   exactly — the one place training and serving logic aren't literally
   shared code, and so the one place a silent mismatch could hide).
4. Train with Adam, MSE loss regressing to the 0/1/2 label, 200 epochs,
   batch size 512, on the resulting ~195K labeled pairs.
5. Export via `torch.onnx.export` to `models/neural-reranker/model.onnx`
   (input `features` `[batch, 6]` → output `score` `[batch, 1]`) and
   verify the exported graph's output matches the in-memory torch model's
   output on a sample batch before trusting it.
6. Write `models/neural-reranker/training-metadata.json` (feature names,
   split sizes, example count) for future reference.

### How to run

```bash
./scripts/train-neural-reranker.sh
```

**Required**, not optional, once you want `search-api` to start — its
`NeuralRerankerService` bean fails without `models/neural-reranker/model.onnx`,
exactly like the `EmbeddingService`/`RerankerService` beans fail without
their model dirs.

### Track A results

Real run, from `training-metadata.json` and `RESULTS.md`:

| | |
|---|---|
| Training queries | 384 |
| Held-out queries | 96 |
| Training examples | 194,925 |
| Training loss (MSE) | 0.3641 → 0.2431 over 200 epochs |

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency |
|---|---|---|---|---|---|---|
| Hybrid (RRF, k=60) — all 480 queries | 0.7308 | 0.9226 | 0.0638 | 0.2506 | 0.8431 | 8ms |
| Hybrid + Cross-Encoder Rerank — all 480 | 0.7456 | 0.9015 | 0.0642 | 0.2506 | 0.8358 | 1288ms |
| **Neural Rerank (MLP) — held-out 96** | **0.6789** | 0.8645 | 0.0622 | 0.2729 | 0.8052 | **15ms** |

**Honest result: it doesn't beat Hybrid.** ~85x faster than the
cross-encoder and about as fast as Hybrid itself, but its nDCG@10 is
*below* Hybrid's — meaning its reordering of Hybrid's own candidate pool
is actively worse than leaving Hybrid's RRF order alone. Kept as a
documented latency/quality tradeoff point (a 5th `SearchStrategy`), not a
replacement for the cross-encoder. See [WRITEUP.md](WRITEUP.md) for the
full discussion of likely causes (pointwise MSE vs. a ranking-aware loss,
feature ceiling, label imbalance, held-out sample size).

### Track A troubleshooting

- **`torch==2.5.1` not installable**: the pinned version in an earlier
  draft of `requirements.txt` predated the installed Python (3.13);
  switched to `>=` minimum-version constraints throughout.
- **`ModuleNotFoundError: No module named 'onnxscript'`** on export: see
  the `dynamo=False` note in [Setup](#setup).

## Track B: fine-tuned two-tower vs. shared-tower embeddings

### What it trains

`SemanticSearchStrategy` uses `bge-small-en-v1.5` exactly as published —
never fine-tuned on WANDS. This asks: does fine-tuning it on WANDS's own
relevance judgments help, and does giving queries and products **separate**
encoders (a true two-tower model) beat one **shared** encoder used for
both? Both get trained and scored through the real `search-eval`/
`TowerComparisonCli` machinery (not a Python-side proxy metric), and
whichever wins ships as a new standalone `SearchStrategy`.

Both modes fine-tune the actual pretrained `BAAI/bge-small-en-v1.5`
transformer (not a from-scratch encoder) with a pairwise margin-ranking
loss:

```
query text   → tokenize → query tower  → CLS-pool + L2-normalize → query embedding
product text → tokenize → product tower → CLS-pool + L2-normalize → product embedding
```

- `--mode shared`: query tower and product tower are the *same* model
  instance/weights.
- `--mode two-tower`: two independently-initialized, independently-trained
  copies of the pretrained model — one specializing per role.

Loss: for each `(query, higher-graded product, lower-graded product)`
triplet built from WANDS's within-query grade differences (Exact vs.
Partial vs. Irrelevant), `MarginRankingLoss` pushes
`cosine(query, higher)` above `cosine(query, lower)` by a margin of 0.1.

### The key fact that keeps this simple on the Java side

Fine-tuning still starts from `bge-small-en-v1.5`'s architecture and gets
exported to ONNX with the **exact same tensor contract**
(`input_ids`/`attention_mask`/`token_type_ids` → `last_hidden_state`,
CLS-pooled + L2-normalized by the *caller*, not baked into the graph) that
`EmbeddingService.java` already expects. **No new Java inference class was
needed** — a fine-tuned tower is just another model directory
`EmbeddingService` can load. Two-tower mode means two `EmbeddingService`
instances (one per model dir); shared-tower mode means one instance reused
for both `embedQuery`/`embedDocument`.

### Process (`training/train_embedding_towers.py --mode {shared,two-tower}`)

1. Load `dataset/product.csv` (for product text — a from-scratch fine-tune
   can't reuse bge's own precomputed embeddings) + `dataset/query.csv` +
   `dataset/label.csv`. Product text is built the same way
   `EmbeddingTextBuilder.java` does (name + class + normalized category
   hierarchy + truncated description) — replicated carefully in Python,
   flagged as a risk area same as Track A's query-embedding replication.
2. Same 80/20 query split as Track A (`training/query_split.py` — a small
   shared module both tracks import, so every training track in this repo
   holds out the *same* queries; Track A's script was retrofitted to
   import it too, rather than duplicating the split logic).
3. Build `(query, higher-graded product, lower-graded product)` triplets
   from each training query's labeled products, capped at 15 per query
   (5,404 total from 384 training queries).
4. Load `BAAI/bge-small-en-v1.5` via `transformers.AutoModel.from_pretrained`
   — a genuinely separate download from the ONNX export already in
   `models/` (see [Setup](#setup)). Tokenize with the *local*
   `models/bge-small-en-v1.5/tokenizer.json` via the `tokenizers` package
   directly (avoids a second tokenizer download).
5. Train with `AdamW` (lr `2e-5`), batch size 16, 2 epochs, full
   fine-tuning (all weights, no frozen layers).
6. Export: `--mode shared` → `models/learned-shared-tower/{model.onnx,
   tokenizer.json}`. `--mode two-tower` → `models/learned-query-tower/`
   and `models/learned-product-tower/`. Same export-then-verify pattern as
   Track A (checks the ONNX graph's output against the in-memory torch
   model's output — both runs so far matched to within `2e-6`).

Useful flags for smoke-testing before committing to a full run (CPU-only
fine-tuning of even a "small" transformer is genuinely slow — normal
per-step time is low single-digit seconds, not milliseconds):

```bash
python3 training/train_embedding_towers.py --mode shared \
  --max-triplets 32 --batch-size 8 --epochs 1
```

### Comparison runbook (manual, staged)

One `learned_embedding` `dense_vector` field was added to
`products-mapping.json` (and the live index, via `PUT
products/_mapping`, since adding a new field to an existing Elasticsearch
index doesn't require a reindex). Each candidate's embeddings are ingested
into that one field, evaluated, and compared — one candidate at a time,
deliberately avoiding a second permanent field for "the loser" since only
one candidate ships:

```bash
# 1. Train a candidate
./scripts/train-embedding-tower.sh shared        # or: two-tower

# 2. Ingest its product-tower embeddings into learned_embedding
#    (overwrites whatever candidate was ingested before)
java -jar search-ingestion/target/search-ingestion.jar \
  --learned-model-dir models/learned-shared-tower    # or models/learned-product-tower

# 3. Score it on the same held-out split via the standalone comparison
#    driver (kept separate from EvalCli's stable 5-strategy flow until a
#    winner is chosen — see search-eval's TowerComparisonCli)
java -cp search-eval/target/search-eval.jar \
  com.avanti.search.eval.TowerComparisonCli models/learned-shared-tower "Learned Tower (shared)"
```

`IngestionCli --learned-model-dir` re-embeds and partial-updates
(`_update`, not a full reindex) just the `learned_embedding` field for
every product already in the index — a separate pass from normal
ingestion, used only for this comparison and the eventual final
deployment.

### Track B results

| Candidate | nDCG@10 (held-out 96) | MRR | Recall@10 | Recall@50 | Precision@10 |
|---|---|---|---|---|---|
| `Semantic (bge-small-en-v1.5)` off-the-shelf — *all 480, for reference* | 0.6990 | 0.8872 | 0.0580 | 0.2303 | 0.7988 |
| **Shared tower (winner)** | **0.6594** | 0.8753 | 0.0559 | 0.2279 | 0.7719 |
| Two-tower | 0.6468 | 0.8532 | 0.0551 | 0.2121 | 0.7667 |

Two honest findings, not one:

1. **Neither fine-tuned candidate beats the off-the-shelf model** —
   plausibly because 5,404 triplets over 2 epochs is a small amount of
   task-specific fine-tuning next to `bge-small-en-v1.5`'s original
   large-scale pretraining, which a short fine-tune can erode faster than
   it adds task-specific signal.
2. **Shared-tower clearly beat two-tower** — a clean sweep across every
   metric, not a close call. Two-tower doubles the trainable parameters
   (two independent encoders vs. one) without doubling the training data,
   so each encoder gets less supervision per parameter — the kind of gap
   that plausibly narrows or flips with more training data, which this
   comparison didn't have.

Shared-tower shipped as the 6th `SearchStrategy`
(`LearnedTowerSearchStrategy` / `StrategyType.LEARNED_TOWER`), wired into
`EvalCli`, `search-api`, and the demo exactly like the other five. See
[WRITEUP.md](WRITEUP.md) for the full narrative version of this result.

**Comparison mechanics, for reproducibility:** both candidates' embeddings
were ingested into the *same* `learned_embedding` field one at a time (see
the runbook above) — two-tower's product-tower embeddings were ingested
and scored first (0.6468), then overwritten by re-ingesting shared-tower's
(0.6594), which is what the field holds now, matching the winner
`LearnedTowerSearchStrategy` actually queries against.

### Track B troubleshooting

Real problems hit building this pipeline, in the order encountered:

1. **`TypeError: BertModel.forward() got multiple values for argument
   'use_cache'`** on ONNX export, even after setting
   `model.config.use_cache = False`. Root cause turned out to be
   `transformers` 5.x itself (installed via unpinned `>=4.46`) — its
   `BertModel.forward()` signature isn't compatible with the legacy
   TorchScript tracer regardless of the `use_cache` config value. Fixed by
   pinning `transformers>=4.46,<5` (see [Setup](#setup)) rather than
   fighting the newer major version's internal wrapper mechanism.
2. **A "full" training run appeared to hang** — 36 minutes with not even
   the first 50-step progress print. Root cause was CPU contention from
   an idle-but-still-loaded `search-api` process (with its own loaded
   ONNX models) left running from earlier in the session, not the
   training itself — killing it and re-running showed real per-step
   timing was ~1-4s/step, entirely reasonable. Lesson: stop other
   ONNX/JVM processes before a CPU-bound training run, and always
   smoke-test timing (`--max-triplets` etc.) before trusting a "full run
   estimate" from a first attempt.
3. **Full-catalog `learned_embedding` ingestion (~43K products) took much
   longer in wall-clock time than its actual CPU time** (~2 hours
   wall-clock vs. ~51 minutes of actual CPU work on one run) — the
   process was repeatedly suspended by environment-level interruptions
   outside the training/ingestion code itself, not a performance problem
   in `IngestionCli`.
4. **The interruptions above recurred** (a second full-catalog pass for
   the two-tower candidate was killed and restarted roughly a dozen times,
   each surviving only a few minutes before being stopped) — restarting
   from scratch each time would have made this pass essentially never
   finish. Fixed by making `IngestionCli --learned-model-dir` resumable:
   progress checkpoints to a local file (`.learned-embedding-checkpoint-
   <model-dir-name>.txt`, gitignored) after every batch, and a re-run picks
   up from there instead of recomputing from product 1. A permanent,
   generically useful change to `IngestionCli`, not a one-off workaround —
   any long-running `--learned-model-dir` pass benefits from it,
   independent of why an interruption happens.
