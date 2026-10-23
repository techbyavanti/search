# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## Project

A product search engine over the WANDS furniture catalog. It compares seven
ranking strategies behind one `SearchStrategy` interface. Java 26 and
Spring Boot serve the API. All models run in-process through ONNX Runtime.
Python is used only for offline training under `training/`.

| Module | Responsibility |
|---|---|
| `search-common` | WANDS data types and the CSV loader |
| `search-inference` | ONNX model wrappers (embedding, cross-encoder, MLP reranker) |
| `search-retrieval` | The seven `SearchStrategy` implementations, RRF fusion, caches, rerank features |
| `search-ingestion` | Builds the Elasticsearch index |
| `search-api` | Spring Boot REST API and wiring |
| `search-eval` | Offline evaluation (`RESULTS.md`), the neural reranker feature export, and HNSW tuning |

See [ARCHITECTURE.md](ARCHITECTURE.md), [HOWTO.md](HOWTO.md), and
[TRAINING.md](TRAINING.md).

## Commands

```bash
export JAVA_HOME="/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
mvn -q test                          # all tests; model-dependent tests skip when models/ is empty
./scripts/sonar-scan.sh              # SonarQube quality gate (throwaway server on :9001; --keep to browse)
./scripts/scan-cves.sh               # Trivy: dependency CVEs, Dockerfile misconfigurations, secrets
mvn -q package -DskipTests           # build all jars
./scripts/run-eval.sh --baseline-file ci/eval-baseline.json   # full eval; rewrites RESULTS.md and results/
./scripts/train-neural-reranker.sh   # export features (Java), then train (Python)
java -cp search-eval/target/search-eval.jar com.avanti.search.eval.HnswTuningCli   # HNSW sweep → results/hnsw-sweep.csv
```

The API runs on port 8090 locally (`--server.port=8090`), because 8080 is
used by another service on this machine.

## Clean code guidelines

### Design

- **One responsibility per class.** A strategy ranks; a service runs one
  model; a builder builds features. Do not mix I/O, model calls, and
  ranking logic in one method.
- **Depend on the narrowest type.** Take `SearchStrategy` unless you need
  more (for example, `NeuralRerankStrategy` takes
  `HybridRrfSearchStrategy` because it needs `searchWithSignals`).
- **Inject dependencies through constructors.** No field injection, no
  static singletons. Pass a `Function` instead of a client when that makes
  a class testable without Elasticsearch (see `ProductFeatureCache`).
- **Make classes `final` and fields `private final`.** Use `record` for
  data carriers (`ScoredResult`, `HybridCandidate`).
- **One implementation of each rule.** Do not copy logic between Java and
  Python or between modules. Rerank features exist only in
  `RerankFeatureBuilder`; Python reads the exported file.
- **Fail fast on broken contracts.** Check invariants at construction time
  (chunk size ≥ 1, model input width = `FEATURE_COUNT`) with a message
  that says how to fix it.

### Functions and naming

- Keep methods short and at one level of abstraction. Extract a well-named
  private method instead of adding a comment that explains a block.
- Use names that state intent: `fetchFeaturesByIds`, `scorableIds`,
  `heldOutQueryIds`. No abbreviations other than standard ones (`id`,
  `k`, `ms`).
- Name constants; do not leave magic numbers in logic
  (`DEFAULT_CHUNK_SIZE`, `NDCG_K`, `FEATURE_CACHE_MAX_SIZE`).
- Prefer immutable results (`List.of`, `.toList()`); do not return
  mutable internal state.

### Comments

- Javadoc says **why**, and the contracts a caller must know (order,
  units, null meaning, thread safety). Do not restate the code.
- Do not write history in code or docs ("previously", "now", "we changed").
  Describe what exists.
- Keep comment density close to the surrounding code.

### Errors and concurrency

- Wrap checked exceptions with context (`UncheckedIOException("Lexical
  search failed for query: " + query, e)`). Never swallow an exception.
- On interruption, restore the interrupt flag and cancel outstanding work.
- Use virtual threads for I/O fan-out. Bound CPU-heavy model work (one
  intra-op thread per pass, a semaphore sized to the core count).
- Keep `synchronized` sections short; do slow work (model calls, network)
  outside the lock.

### ML and evaluation

- Training and serving use the same code path for features, candidate
  pools, and RRF `k`. Any change to `RerankFeatureBuilder.FEATURE_NAMES`
  needs a retrain.
- Never tune or select on held-out queries
  (`neural-reranker-eval-queries.txt`). Use a validation slice of the
  training split.
- Compare models on the same queries and the same index. Use an A/B run
  to isolate one change.
- Measure latency serially; disable caches that would hide model cost.
- Verify every ONNX export against the PyTorch output.
- Report negative results honestly, with the likely cause.

### Static analysis

- The SonarQube quality gate (`scripts/sonar-scan.sh`, CI job `sonarqube`)
  must pass: A ratings for reliability, security, and maintainability,
  hotspots reviewed, coverage at or above the floor, ≤ 3% duplication.
  Run it before pushing a non-trivial change.
- Fix the finding; do not lower the gate. Suppress a rule only when the
  finding is a deliberate design choice, with
  `@SuppressWarnings("java:Sxxxx") // reason` on the narrowest element.
- When coverage rises, raise `COVERAGE_FLOOR` in `scripts/sonar-scan.sh`
  and the number in `CI.md`.
- Trivy (`scripts/scan-cves.sh`) must report no CRITICAL, HIGH, or MEDIUM
  vulnerability, misconfiguration, or secret.
- Close every `AutoCloseable` (ONNX `SessionOptions`, tensors, results)
  with try-with-resources. Avoid regexes that backtrack on long input;
  prefer a split or a single pass.

### Tests

- JUnit 5 and AssertJ. One behavior per test; the test name states it.
- Unit-test pure logic with fakes (no Elasticsearch). Tests that need a
  model use `assumeTrue` to skip when the model is absent.
- Pin contracts in tests: feature order, ranking order, cache keys.

### Python (training only)

- Type-stable, small functions; constants at module top; `argparse` for
  options; deterministic seeds.
- No pandas or new dependencies without a reason; keep
  `training/requirements.txt` minimal.

## Keep docs and comments in sync

Every feature change is complete only when the docs and code comments
describe the new behavior. In the same change:

1. **Code comments and Javadoc.** Update every comment that names the
   changed class, method, constant, feature, or number. Search for the old
   name and value (`git grep`), including tests and scripts.
2. **Docs.** Check and update each file that describes the area:

   | Change | Files to check |
   |---|---|
   | Any ranking, model, or metric change | `README.md`, `ARCHITECTURE.md` (text and Mermaid diagrams), `RESULTS.md` (regenerate with the eval, do not hand-edit), `docs/index.html` |
   | Training or features | `TRAINING.md`, `HOWTO.md`, `training/requirements.txt` |
   | Setup, scripts, or config | `HOWTO.md`, `README.md`, `.idea/runConfigurations/` |
   | CI or regression floors | `CI.md`, `ci/eval-baseline.json` (entries are matched by position, in `EvalCli` order), `COVERAGE_FLOOR` in `scripts/sonar-scan.sh` |
   | Index mapping or vector settings | `products-mapping.json`, `HOWTO.md` step 3, `ARCHITECTURE.md` (HNSW tuning), rerun `HnswTuningCli`; retrain the neural reranker if Hybrid's pools change |
   | Demo results | Recapture `docs/data/` (`scripts/capture-demo-snapshots.sh`, then `scripts/extract-product-details.py`) and re-embed both data blocks in `docs/index.html` |

3. **Numbers.** Quote metrics only from the latest `RESULTS.md` run. When a
   number changes, search all docs for the old value and replace each one.
4. **No history.** Describe what exists; do not write "previously",
   "now", or "changed from".
5. **Verify.** Mermaid diagrams still render, links resolve, and
   `git grep` finds no stale names or values.

## Git

- Commit as `techbyavanti <60168355+techbyavanti@users.noreply.github.com>`
  (repo-local config).
- Do not add `Co-Authored-By` or "Generated with" lines to commits or pull
  requests.
- Commit only when asked.
