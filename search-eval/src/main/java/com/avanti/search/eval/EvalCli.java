package com.avanti.search.eval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.common.RelevanceGrade;
import com.avanti.search.common.WandsCsvLoader;
import com.avanti.search.common.WandsDataset;
import com.avanti.search.common.WandsQuery;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.inference.NeuralRerankerService;
import com.avanti.search.inference.RerankerService;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.HybridRerankStrategy;
import com.avanti.search.retrieval.HybridRrfSearchStrategy;
import com.avanti.search.retrieval.LearnedTowerSearchStrategy;
import com.avanti.search.retrieval.LexicalSearchStrategy;
import com.avanti.search.retrieval.NeuralRerankStrategy;
import com.avanti.search.retrieval.ProductFeatureCache;
import com.avanti.search.retrieval.ProductLookup;
import com.avanti.search.retrieval.RrfFusionService;
import com.avanti.search.retrieval.ScoredResult;
import com.avanti.search.retrieval.SearchStrategy;
import com.avanti.search.retrieval.SemanticSearchStrategy;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Runs every currently-implemented SearchStrategy against all 480 WANDS
 * queries and reports nDCG@10 / MRR / Recall@10 / Recall@50 / Precision@10,
 * writing a fresh RESULTS.md each run. Strategies are added to STRATEGY
 * count as later milestones implement them (M2: semantic, M3: hybrid,
 * M4: hybrid+rerank) — this file is re-run in full each time rather than
 * patching individual rows, so RESULTS.md can never go stale relative to
 * the strategies that actually exist.
 *
 * <p>Metrics (nDCG/MRR/recall/precision) are computed concurrently across
 * all 480 queries — correct regardless of concurrency, and much faster.
 * Latency is measured separately, serially, over a fixed-size sample —
 * firing all 480 queries as one simultaneous burst would measure queueing
 * delay behind that burst rather than realistic single-query serving
 * latency, a gap that's small for the cheaper strategies but enormous for
 * the reranker (a few seconds of genuine per-query cross-encoder work
 * compounds into minutes of queueing once hundreds of queries pile up on
 * one CPU at once).
 */
@Command(name = "search-eval", mixinStandardHelpOptions = true,
        description = "Evaluates all implemented ranking strategies against WANDS relevance judgments.")
public class EvalCli implements Callable<Integer> {

    private static final Logger log = LoggerFactory.getLogger(EvalCli.class);

    private static final int TOP_K = 50;
    private static final int NDCG_K = 10;
    private static final int PRECISION_K = 10;
    private static final int RECALL_SMALL_K = 10;
    private static final List<Integer> RRF_K_CANDIDATES = List.of(10, 20, 40, 60, 100, 200);
    private static final int LATENCY_SAMPLE_SIZE = 50;

    @Option(names = "--dataset-dir", defaultValue = "dataset")
    private Path datasetDir;

    @Option(names = "--host", defaultValue = "http://localhost:9200")
    private String host;

    @Option(names = "--results-dir", defaultValue = "results")
    private Path resultsDir;

    @Option(names = "--results-md", defaultValue = "RESULTS.md")
    private Path resultsMdPath;

    @Option(names = "--model-dir", description = "Directory containing the embedding model's model.onnx/tokenizer.json", defaultValue = "models/bge-small-en-v1.5")
    private Path modelDir;

    @Option(names = "--reranker-model-dir", description = "Directory containing the reranker's model.onnx/tokenizer.json", defaultValue = "models/ms-marco-MiniLM-L-6-v2")
    private Path rerankerModelDir;

    @Option(names = "--neural-reranker-model-dir", description = "Directory containing the neural reranker's model.onnx (see scripts/train-neural-reranker.sh)", defaultValue = "models/neural-reranker")
    private Path neuralRerankerModelDir;

    @Option(names = "--learned-tower-model-dir", description = "Directory containing the winning fine-tuned tower's model.onnx (see scripts/train-embedding-tower.sh and TRAINING.md)", defaultValue = "models/learned-shared-tower")
    private Path learnedTowerModelDir;

    @Option(names = "--baseline-file", description = "Optional regression floor (see ci/eval-baseline.json) — if given, exits non-zero when any strategy's nDCG@10 falls below its floor")
    private Path baselineFile;

    private static final int FEATURE_CACHE_MAX_SIZE = 50_000;
    private static final String HELD_OUT_QUERIES_RESOURCE = "/neural-reranker-eval-queries.txt";

    public static void main(String[] args) {
        int exitCode = new CommandLine(new EvalCli()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        WandsDataset dataset = WandsCsvLoader.load(datasetDir);
        log.info("Loaded {} queries, {} labels", dataset.queries().size(), dataset.labels().size());

        Files.createDirectories(resultsDir);

        List<StrategySummary> summaries;
        try (ElasticsearchClient client = ElasticsearchClients.create(host);
             EmbeddingService embeddingService = new EmbeddingService(modelDir);
             RerankerService rerankerService = new RerankerService(rerankerModelDir);
             NeuralRerankerService neuralRerankerService = new NeuralRerankerService(neuralRerankerModelDir);
             EmbeddingService learnedTowerEmbeddingService = new EmbeddingService(learnedTowerModelDir)) {
            ProductFeatureCache featureCache = new ProductFeatureCache(
                    ids -> ProductLookup.fetchByIds(client, ids), FEATURE_CACHE_MAX_SIZE);
            LexicalSearchStrategy lexical = new LexicalSearchStrategy(client);
            SemanticSearchStrategy semantic = new SemanticSearchStrategy(client, embeddingService);
            List<WandsQuery> latencySample = selectLatencySample(dataset.queries(), LATENCY_SAMPLE_SIZE);

            summaries = new ArrayList<>();
            summaries.add(evaluate(lexical, dataset, latencySample));
            summaries.add(evaluate(semantic, dataset, latencySample));

            List<RrfSweepRow> sweep = sweepRrfConstant(lexical, semantic, dataset);
            writeSweepCsv(sweep);
            int bestK = sweep.stream()
                    .max(Comparator.comparingDouble(RrfSweepRow::ndcgAt10))
                    .orElseThrow()
                    .k();
            log.info("RRF sweep complete; best k={} by nDCG@10", bestK);

            HybridRrfSearchStrategy hybrid = new HybridRrfSearchStrategy(lexical, semantic, TOP_K, bestK);
            summaries.add(evaluate(hybrid, dataset, latencySample));

            HybridRerankStrategy rerank = new HybridRerankStrategy(hybrid, featureCache, rerankerService, TOP_K);
            summaries.add(evaluate(rerank, dataset, latencySample));

            // Scored on the held-out 20% of queries only (see scripts/train-neural-reranker.sh
            // and scripts/train-embedding-tower.sh) — both these strategies' training data
            // comes from the other 80%, unlike the four strategies above.
            WandsDataset heldOutDataset = restrictToHeldOutQueries(dataset);
            List<WandsQuery> heldOutLatencySample = selectLatencySample(heldOutDataset.queries(), LATENCY_SAMPLE_SIZE);

            NeuralRerankStrategy neuralRerank = new NeuralRerankStrategy(
                    hybrid, client, embeddingService, neuralRerankerService, featureCache, TOP_K);
            summaries.add(evaluate(neuralRerank, heldOutDataset, heldOutLatencySample));

            // The shared-tower mode won Track B's head-to-head against a two-tower model on
            // this same held-out split (0.6594 vs. 0.6468 nDCG@10 — see TRAINING.md) — this is
            // that winner, wired the same way as any other strategy from here on.
            LearnedTowerSearchStrategy learnedTower = new LearnedTowerSearchStrategy(client, learnedTowerEmbeddingService);
            summaries.add(evaluate(learnedTower, heldOutDataset, heldOutLatencySample));

            writeResultsMarkdown(summaries, sweep);
        }

        if (baselineFile != null) {
            EvalBaseline baseline = EvalBaseline.load(baselineFile);
            List<String> failures = BaselineChecker.checkRegressions(summaries, baseline);
            if (!failures.isEmpty()) {
                failures.forEach(f -> log.error("REGRESSION: {}", f));
                return 1;
            }
            log.info("All {} strategies met their baseline floor in {}", summaries.size(), baselineFile);
        }

        return 0;
    }

    private StrategySummary evaluate(SearchStrategy strategy, WandsDataset dataset, List<WandsQuery> latencySample)
            throws InterruptedException, IOException {
        log.info("Evaluating strategy: {}", strategy.name());
        List<QueryMetrics> perQuery = evaluateMetrics(strategy, dataset);
        writePerQueryCsv(strategy.name(), perQuery);

        List<Long> latenciesMs = measureLatencies(strategy, latencySample);
        return StrategySummary.aggregate(strategy.name(), perQuery, latenciesMs);
    }

    private List<QueryMetrics> evaluateMetrics(SearchStrategy strategy, WandsDataset dataset) throws InterruptedException {
        List<WandsQuery> queries = dataset.queries();
        List<QueryMetrics> perQuery = new ArrayList<>();

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<QueryMetrics>> futures = new ArrayList<>();
            for (WandsQuery query : queries) {
                futures.add(executor.submit(() -> runOneQuery(strategy, query, dataset)));
            }
            for (Future<QueryMetrics> future : futures) {
                try {
                    perQuery.add(future.get());
                } catch (ExecutionException e) {
                    throw new RuntimeException("Query evaluation failed for strategy " + strategy.name(), e.getCause());
                }
            }
        }
        return perQuery;
    }

    private QueryMetrics runOneQuery(SearchStrategy strategy, WandsQuery query, WandsDataset dataset) {
        Map<String, RelevanceGrade> judgments = dataset.judgmentsFor(query.queryId());
        List<ScoredResult> results = strategy.search(query.queryText(), TOP_K);
        return toMetrics(query.queryId(), results, judgments);
    }

    /**
     * Serial (one query at a time), unlike evaluateMetrics — see class
     * javadoc for why concurrent measurement would be misleading here.
     */
    private List<Long> measureLatencies(SearchStrategy strategy, List<WandsQuery> sampleQueries) {
        strategy.search(sampleQueries.get(0).queryText(), TOP_K); // warm up, excluded from the sample

        List<Long> latenciesMs = new ArrayList<>(sampleQueries.size());
        for (WandsQuery query : sampleQueries) {
            long start = System.nanoTime();
            strategy.search(query.queryText(), TOP_K);
            latenciesMs.add((System.nanoTime() - start) / 1_000_000);
        }
        return latenciesMs;
    }

    /**
     * Filters to the held-out query ids written by
     * scripts/train-neural-reranker.sh — NeuralRerankStrategy trains on the
     * other 80%, so scoring it on all 480 (like the other strategies) would
     * partly measure how well it memorized its own training queries.
     */
    private WandsDataset restrictToHeldOutQueries(WandsDataset dataset) {
        Set<String> heldOutQueryIds = new HashSet<>();
        try (InputStream in = EvalCli.class.getResourceAsStream(HELD_OUT_QUERIES_RESOURCE)) {
            if (in == null) {
                throw new IllegalStateException("Missing " + HELD_OUT_QUERIES_RESOURCE
                        + " on the classpath — run scripts/train-neural-reranker.sh first");
            }
            for (String line : new String(in.readAllBytes(), StandardCharsets.UTF_8).split("\n")) {
                if (!line.isBlank()) {
                    heldOutQueryIds.add(line.trim());
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read " + HELD_OUT_QUERIES_RESOURCE, e);
        }

        List<WandsQuery> heldOutQueries = dataset.queries().stream()
                .filter(q -> heldOutQueryIds.contains(q.queryId()))
                .toList();
        return new WandsDataset(dataset.products(), heldOutQueries, dataset.labels(), dataset.judgmentsByQuery());
    }

    private static List<WandsQuery> selectLatencySample(List<WandsQuery> queries, int sampleSize) {
        if (queries.size() <= sampleSize) {
            return queries;
        }
        int step = queries.size() / sampleSize;
        List<WandsQuery> sample = new ArrayList<>(sampleSize);
        for (int i = 0; i < queries.size() && sample.size() < sampleSize; i += step) {
            sample.add(queries.get(i));
        }
        return sample;
    }

    private QueryMetrics toMetrics(String queryId, List<ScoredResult> results, Map<String, RelevanceGrade> judgments) {
        List<String> rankedIds = results.stream().map(ScoredResult::productId).toList();
        return new QueryMetrics(
                queryId,
                MetricsCalculator.ndcgAtK(rankedIds, judgments, NDCG_K),
                MetricsCalculator.reciprocalRank(rankedIds, judgments),
                MetricsCalculator.recallAtK(rankedIds, judgments, RECALL_SMALL_K),
                MetricsCalculator.recallAtK(rankedIds, judgments, TOP_K),
                MetricsCalculator.precisionAtK(rankedIds, judgments, PRECISION_K)
        );
    }

    /**
     * Sweeps candidate RRF constants against the same lexical/semantic
     * candidate lists rather than re-querying Elasticsearch/ONNX per
     * candidate — fusion is pure in-memory arithmetic, so this makes the
     * sweep cheap regardless of how many k values are tried.
     */
    private List<RrfSweepRow> sweepRrfConstant(SearchStrategy lexical, SearchStrategy semantic, WandsDataset dataset)
            throws InterruptedException {
        log.info("Sweeping RRF constant candidates: {}", RRF_K_CANDIDATES);
        List<WandsQuery> queries = dataset.queries();
        List<QueryCandidates> perQueryCandidates = new ArrayList<>();

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<QueryCandidates>> futures = new ArrayList<>();
            for (WandsQuery query : queries) {
                futures.add(executor.submit(() -> new QueryCandidates(
                        query.queryId(),
                        lexical.search(query.queryText(), TOP_K),
                        semantic.search(query.queryText(), TOP_K),
                        dataset.judgmentsFor(query.queryId()))));
            }
            for (Future<QueryCandidates> future : futures) {
                try {
                    perQueryCandidates.add(future.get());
                } catch (ExecutionException e) {
                    throw new RuntimeException("Candidate generation failed during RRF sweep", e.getCause());
                }
            }
        }

        List<RrfSweepRow> sweep = new ArrayList<>();
        for (int k : RRF_K_CANDIDATES) {
            List<QueryMetrics> perQuery = new ArrayList<>();
            for (QueryCandidates candidates : perQueryCandidates) {
                List<ScoredResult> fused = RrfFusionService.fuse(
                        List.of(candidates.lexicalResults(), candidates.semanticResults()), k);
                perQuery.add(toMetrics(candidates.queryId(), fused, candidates.judgments()));
            }
            StrategySummary summary = StrategySummary.aggregate("k=" + k, perQuery, List.of());
            sweep.add(new RrfSweepRow(k, summary.ndcgAt10(), summary.mrr(), summary.recallAt10(),
                    summary.recallAt50(), summary.precisionAt10()));
        }
        return sweep;
    }

    private record QueryCandidates(String queryId, List<ScoredResult> lexicalResults,
                                    List<ScoredResult> semanticResults, Map<String, RelevanceGrade> judgments) {
    }

    private record RrfSweepRow(int k, double ndcgAt10, double mrr, double recallAt10,
                                double recallAt50, double precisionAt10) {
    }

    private void writeSweepCsv(List<RrfSweepRow> sweep) throws IOException {
        Path path = resultsDir.resolve("rrf-sweep.csv");
        CSVFormat format = CSVFormat.Builder.create(CSVFormat.DEFAULT)
                .setHeader("rrf_k", "ndcg_at_10", "mrr", "recall_at_10", "recall_at_50", "precision_at_10")
                .build();

        try (Writer writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
             CSVPrinter printer = new CSVPrinter(writer, format)) {
            for (RrfSweepRow row : sweep) {
                printer.printRecord(row.k(), row.ndcgAt10(), row.mrr(), row.recallAt10(), row.recallAt50(), row.precisionAt10());
            }
        }
        log.info("Wrote RRF sweep results to {}", path);
    }

    private void writePerQueryCsv(String strategyName, List<QueryMetrics> perQuery) throws IOException {
        String fileName = strategyName.toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]+", "-")
                .replaceAll("(^-|-$)", "") + ".csv";
        Path path = resultsDir.resolve(fileName);

        CSVFormat format = CSVFormat.Builder.create(CSVFormat.DEFAULT)
                .setHeader("query_id", "ndcg_at_10", "mrr", "recall_at_10", "recall_at_50", "precision_at_10")
                .build();

        try (Writer writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
             CSVPrinter printer = new CSVPrinter(writer, format)) {
            for (QueryMetrics m : perQuery) {
                printer.printRecord(m.queryId(), m.ndcgAt10(), m.mrr(), m.recallAt10(), m.recallAt50(), m.precisionAt10());
            }
        }
        log.info("Wrote per-query metrics to {}", path);
    }

    private void writeResultsMarkdown(List<StrategySummary> summaries, List<RrfSweepRow> sweep) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("# Evaluation Results\n\n");
        sb.append("Offline IR evaluation of each ranking strategy against the WANDS relevance\n");
        sb.append("judgments (480 queries, top-").append(TOP_K).append(" retrieved per query). ");
        sb.append("Regenerate with `./scripts/run-eval.sh`.\n\n");
        sb.append("| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency (ms) |\n");
        sb.append("|---|---|---|---|---|---|---|\n");
        for (StrategySummary s : summaries) {
            sb.append("| ").append(s.strategyName())
                    .append(" | ").append(format(s.ndcgAt10()))
                    .append(" | ").append(format(s.mrr()))
                    .append(" | ").append(format(s.recallAt10()))
                    .append(" | ").append(format(s.recallAt50()))
                    .append(" | ").append(format(s.precisionAt10()))
                    .append(" | ").append(s.p95LatencyMs())
                    .append(" |\n");
        }
        sb.append("\n_Neural Rerank and Learned Tower each train on 80% of the 480 WANDS queries and are scored ");
        sb.append("above on only the held-out 20% (see `search-eval/src/main/resources/neural-reranker-eval-queries.txt`) ");
        sb.append("— their rows aren't on the same query count as the other four. Learned Tower is the shared-tower ");
        sb.append("mode, which won a head-to-head against a two-tower model on this split — see TRAINING.md._\n");

        sb.append("\n## RRF constant sweep\n\n");
        sb.append("Fusion of the same lexical/semantic candidate lists at each k (see `results/rrf-sweep.csv`); ");
        sb.append("the Hybrid row above uses whichever k scored highest on nDCG@10.\n\n");
        sb.append("| k | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 |\n");
        sb.append("|---|---|---|---|---|---|\n");
        for (RrfSweepRow row : sweep) {
            sb.append("| ").append(row.k())
                    .append(" | ").append(format(row.ndcgAt10()))
                    .append(" | ").append(format(row.mrr()))
                    .append(" | ").append(format(row.recallAt10()))
                    .append(" | ").append(format(row.recallAt50()))
                    .append(" | ").append(format(row.precisionAt10()))
                    .append(" |\n");
        }

        Files.writeString(resultsMdPath, sb.toString());
        log.info("Wrote {}", resultsMdPath);
    }

    private static String format(double value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }
}
