# How to run this locally

Step-by-step instructions to get Elasticsearch, the WANDS catalog, and
`search-api` running on your machine, plus how to confirm each piece
actually came up before moving to the next step.

## Prerequisites

Installed via Homebrew: JDK 26, Maven, Colima + Docker + Docker Compose
plugin.

Colima's default VM allocation (2 CPUs) is too little once `search-api`'s
reranker is under any load — a single cross-encoder forward pass can occupy
the one active core long enough that the *same container's* outbound call to
Elasticsearch misses its 1s connect timeout, surfacing as a flaky 500 that
looks like a networking bug but is actually CPU starvation.

```bash
colima start --cpu 6 --memory 8
```

(or edit `~/.colima/default/colima.yaml`).

Homebrew's `openjdk` formula is keg-only, so `java`/`mvn` may not be on
`PATH` even after `brew install openjdk` — if they can't find a runtime, set:

```bash
export JAVA_HOME="$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
```

Note this only needs to be exported in shells that don't already source it
from your shell profile — IntelliJ's integrated terminal and its run
configurations pick it up from your interactive shell's profile
automatically; a plain non-interactive script invocation may not.

## Step-by-step

Run these in order — each step depends on the one before it. Step numbers
match the IntelliJ run configs in the table further down.

### 0. Start Elasticsearch + Kibana

```bash
docker compose up -d
```

**Verify:** `curl -sf http://localhost:9200/_cluster/health` should return
JSON with `"status":"green"` or `"yellow"` (not connection-refused, and not
`"status":"red"`).

### 1. Fetch the WANDS dataset

```bash
./scripts/download-dataset.sh
```

Downloads `product.csv`, `query.csv`, `label.csv` into `dataset/`.

**Verify:** `wc -l dataset/*.csv` — expect ~42,995 products, 480 queries,
233,448 labels (plus one header row each).

### 2. Fetch the embedding + reranker models

```bash
./scripts/download-models.sh
```

Downloads ONNX weights + tokenizers for `bge-small-en-v1.5` (embedding) and
`ms-marco-MiniLM-L-6-v2` (reranker) into `models/`.

**Verify:** `models/` should contain a subdirectory per model, each with a
`model.onnx` and `tokenizer.json`.

### 3. Ingest the catalog into Elasticsearch

```bash
mvn -q -pl search-ingestion -am package -DskipTests
java -jar search-ingestion/target/search-ingestion.jar --recreate
```

This embeds and bulk-indexes all ~43K products — the slow, CPU-bound step
(ONNX inference, not I/O).

**Verify:**

```bash
curl -sf http://localhost:9200/products/_count
```

`count` should match the product row count from step 1's verification
(~42,995).

Each product's embedding is indexed twice: `embedding` (1-bit BBQ HNSW,
Elasticsearch's default for 384 dims) and `embedding_hnsw` (float HNSW,
m=32, ef_construction=200, used by the "Semantic HNSW (float, tuned)"
strategy). An index built before `embedding_hnsw` was in the mapping can
get the field without re-embedding:

```bash
curl -XPUT localhost:9200/products/_mapping -H 'Content-Type: application/json' \
  -d '{"properties":{"embedding_hnsw":{"type":"dense_vector","dims":384,"index":true,"similarity":"cosine","index_options":{"type":"hnsw","m":32,"ef_construction":200}}}}'
curl -XPOST 'localhost:9200/products/_update_by_query?wait_for_completion=false&slices=auto' \
  -H 'Content-Type: application/json' \
  -d '{"query":{"bool":{"must_not":{"exists":{"field":"embedding_hnsw"}}}},"script":{"source":"ctx._source.embedding_hnsw = ctx._source.embedding"}}'
```

### 4. Train the neural reranker

```bash
./scripts/train-neural-reranker.sh
```

**Required** — `search-api`'s `NeuralRerankerService` bean fails to start
without `models/neural-reranker/model.onnx`, like the
`EmbeddingService`/`RerankerService` beans fail without their model dirs.
One-time (unless you want to retrain), about a minute. It runs two steps:

1. `RerankFeatureExportCli` (Java) runs Hybrid for every WANDS query and
   writes each top-50 candidate's feature vector and grade to
   `models/neural-reranker/training-features.tsv`. It uses the same
   feature code the strategy serves with.
2. `training/train_neural_reranker.py` trains a small MLP with a
   LambdaRank loss on the training split and exports it to ONNX. The
   held-out queries in
   `search-eval/src/main/resources/neural-reranker-eval-queries.txt` are
   never used for training or model selection.

Needs Elasticsearch running with the catalog already ingested (step 3).

**Verify:** `models/neural-reranker/model.onnx` exists (a few KB — 17
inputs → 32 → 16 → 1 output). See [TRAINING.md](TRAINING.md) for the
complete process and results.

### 5. Train the embedding tower

```bash
./scripts/train-embedding-tower.sh shared
```

**Required** — `search-api`'s `learnedTowerEmbeddingService` bean fails to
start without `models/learned-shared-tower/model.onnx`. Fine-tunes
`bge-small-en-v1.5` on WANDS (one shared encoder for both queries and
products — this mode outscores the two-tower alternative, see
[TRAINING.md](TRAINING.md)), then populates every
product's `learned_embedding` field via
`java -jar search-ingestion/target/search-ingestion.jar --learned-model-dir
models/learned-shared-tower` (the wrapper script does this for you). Needs
network access on first run (a separate download of the PyTorch checkpoint
from Hugging Face, since ONNX Runtime is inference-only) and Elasticsearch
running with the catalog already ingested (step 3).

**Verify:** `models/learned-shared-tower/model.onnx` exists, and
`curl -sf "http://localhost:9200/products/_count"` still matches step 3's
count (the ingestion pass only updates a field, it doesn't add/remove
documents).

### 6. Run the offline eval harness (optional at this point)

```bash
./scripts/run-eval.sh
```

Writes `RESULTS.md` and per-strategy CSVs under `results/`. Add
`--baseline-file ci/eval-baseline.json` to fail if any strategy's nDCG@10
drops below its regression floor (see [CI.md](CI.md) — this eval runs
locally, not in CI).

**Verify:** exits 0 and prints `All 7 strategies met their baseline floor`
when given `--baseline-file`; otherwise just check `RESULTS.md` was written.

To re-tune the HNSW strategy (optional, about 2 minutes):

```bash
java -cp search-eval/target/search-eval.jar com.avanti.search.eval.HnswTuningCli
```

It builds a scratch index per vector type and graph setting, compares each
`num_candidates` value against exact kNN, writes `results/hnsw-sweep.csv`,
and logs the recommended setting. Apply a new graph setting in
`products-mapping.json` (then rebuild the field) and a new
`num_candidates` in `application.yml` (`search.semantic-hnsw`) and
`EvalCli`.

### 7. Run search-api

```bash
mvn -q -pl search-api -am package -DskipTests
java -jar search-api/target/search-api.jar
```

Or run `com.avanti.search.api.SearchApiApplication`'s `main` directly from
your IDE instead of building a jar first.

**Verify the process is actually up and pollable on `localhost:8080`:**

```bash
./scripts/wait-for-search-api.sh
```

Polls `http://localhost:8080/` until it gets *any* HTTP response (a 404 on
an unmapped path still counts — the point is confirming the process is
listening, not that a specific route exists) or times out after 120s. Then
confirm the real endpoints work end to end:

```bash
curl -sf "http://localhost:8080/api/search?query=modern+oak+dining+table&strategy=LEARNED_TOWER" | head -c 500
```

Or visit in a browser:

- `http://localhost:8080/index.html` — the search-comparison demo page
- `http://localhost:8080/swagger-ui.html` — call `/api/search` and
  `/api/search/compare` directly

## Running search-api in Docker instead of locally

```bash
docker compose up -d --build search-api
```

Build stage matches the local toolchain (Maven 3.9.16 + JDK 26); runtime is
a glibc-based JRE image, since ONNX Runtime's and the tokenizer's native
libraries aren't musl-compatible. This mounts the already-downloaded
`models/` directory read-only into the container rather than baking the
weights into the image, and points `SEARCH_ELASTICSEARCH_HOST` at the
`elasticsearch` compose service by name — no extra configuration needed
beyond having already completed steps 2, 4, and 5 above (`models/`
contains `bge-small-en-v1.5`, `ms-marco-MiniLM-L-6-v2`, `neural-reranker`,
and `learned-shared-tower`, all four required by the strategies —
`docker-compose.yml`'s `search-api` service sets a
`SEARCH_MODELS_*_DIR` env var per model dir, same names `application.yml`
uses locally).

**Verify:** same as step 7 above — `./scripts/wait-for-search-api.sh`, or
`docker compose logs -f search-api` and watch for the Spring Boot startup
banner.

## Sanity-checking the whole build

```bash
mvn test
```

Not part of the numbered flow above (no step depends on it, and it has no
dedicated IntelliJ config) — run it any time to build and test every
module. Tests that need Elasticsearch or the models skip gracefully
(`Assumptions.assumeTrue`) if steps 0–2 (or 4/5, for the neural reranker
and tower tests) haven't been done yet.

## Training the two-tower alternative (optional)

```bash
./scripts/train-embedding-tower.sh two-tower
```

Fine-tunes the other Track B candidate — separate query/product encoders
instead of step 5's shared one. Nothing in `search-api` depends on this
(step 5's shared tower is what's served); use it to reproduce the
comparison or experiment further. See [TRAINING.md](TRAINING.md)
for the full runbook, exact hyperparameters, and both candidates' numbers.

## Running from IntelliJ

Shared run configurations live under `.idea/runConfigurations/` (note:
`.idea/` itself is gitignored in this repo, so these are local-only unless
you deliberately commit that subfolder — see the repo's `.gitignore`):

| # | Config | Runs |
|---|---|---|
| 0 | Start Elasticsearch | `scripts/start-elasticsearch.sh` — `docker compose up -d` for ES + Kibana, then polls `_cluster/health` |
| 1 | Download WANDS dataset | `scripts/download-dataset.sh` |
| 2 | Download ONNX models | `scripts/download-models.sh` |
| 3 | Ingest catalog | `scripts/ingest-catalog.sh` — builds `search-ingestion` and runs it with `--recreate` |
| 4 | Train neural reranker | `scripts/train-neural-reranker.sh` — required before `search-api` will start |
| 5 | Train embedding tower | `scripts/train-embedding-tower.sh shared` — also required before `search-api` will start |
| 6 | Run eval (vs CI baseline) | `scripts/run-eval.sh --baseline-file ci/eval-baseline.json` |
| 7 | Capture demo snapshots | `scripts/capture-demo-snapshots.sh` — hits a running `search-api` to regenerate the GitHub Pages demo data |
| 8 | search-api (app) | `SearchApiApplication.main` directly, no jar build |
| 9 | Wait for search-api :8080 | `scripts/wait-for-search-api.sh` — polls until the app responds |
| 10 | search-api (start + wait until ready) | Compound: runs 8 + 9 together |
| 11 | Stop search-api | `scripts/stop-search-api.sh` — stops it whether it's running as a docker-compose container or a plain local process on :8080 |

Configs 0–6 mirror the numbered steps above one-to-one. Config 7 is a
follow-on task that needs `search-api` already running (see step 7). Configs
8–10 are all ways to run step 7 itself — use 10 when you just want to launch
`search-api` and know as soon as it's actually reachable, without watching
the console for the Spring Boot startup banner yourself. Config 11 stops it
again — useful mainly for the Docker path (`docker compose up -d --build
search-api` has no equivalent "stop" button in the IDE the way a foreground
run configuration does); for config 8/10, IntelliJ's own Stop button on that
running configuration works too.

Note config 5's train-embedding-tower.sh takes `shared` or `two-tower` as
an argument — the config is fixed to `shared` (the served mode); there's
no dedicated config for the optional `two-tower` mode (see the section
above).

After step 7, `scripts/extract-product-details.py` pulls the full WANDS
`product.csv` record (description, structured attributes, rating) for
every product referenced in the freshly-captured `docs/data/q*.json`,
writing `docs/data/product-details.json` — this is what backs the
per-result "why was this shown" detail modal on the demo page. No
run config for it; it's a plain read of `dataset/product.csv` plus the
`docs/data/q*.json` files already on disk, no running services needed.

## Elasticsearch security note

Elasticsearch runs with security disabled (`xpack.security.enabled=false`)
for local dev simplicity — not for production use.

## Where to go next

See [RESULTS.md](RESULTS.md) for the evaluation numbers and
[ARCHITECTURE.md](ARCHITECTURE.md) for module layout and key decisions, and
[CI.md](CI.md) for the pipeline/security posture. See
[TRAINING.md](TRAINING.md) for complete setup, process, and results for
this project's own trained models (the neural reranker and the fine-tuned
embedding towers).
