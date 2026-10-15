package com.avanti.search.eval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.common.RelevanceGrade;
import com.avanti.search.common.WandsCsvLoader;
import com.avanti.search.common.WandsDataset;
import com.avanti.search.common.WandsQuery;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.inference.QueryEncoder;
import com.avanti.search.retrieval.CachingQueryEncoder;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.HybridCandidate;
import com.avanti.search.retrieval.HybridRrfSearchStrategy;
import com.avanti.search.retrieval.LexicalSearchStrategy;
import com.avanti.search.retrieval.NeuralRerankStrategy;
import com.avanti.search.retrieval.ProductFeatureCache;
import com.avanti.search.retrieval.ProductLookup;
import com.avanti.search.retrieval.RerankFeatureBuilder;
import com.avanti.search.retrieval.SemanticSearchStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Writes the neural reranker's training data: for every WANDS query, the
 * hybrid candidate pool NeuralRerankStrategy would rerank, with each
 * candidate's feature vector (built by the same RerankFeatureBuilder code
 * that serves) and its relevance grade. Unlabeled candidates get grade 0,
 * the same gain the eval gives them. Each row is tagged train or held_out
 * from the shared query split, so the trainer never fits on held-out
 * queries.
 *
 * <p>Usage: java -cp search-eval/target/search-eval.jar
 * com.avanti.search.eval.RerankFeatureExportCli [--rrf-k 40] [--output ...]
 */
@Command(name = "rerank-feature-export", mixinStandardHelpOptions = true,
        description = "Exports neural reranker training features for every WANDS query's hybrid candidate pool.")
public final class RerankFeatureExportCli implements Callable<Integer> {

    private static final Logger log = LoggerFactory.getLogger(RerankFeatureExportCli.class);
    private static final int FEATURE_CACHE_MAX_SIZE = 50_000;
    private static final int QUERY_CACHE_SIZE = 1_000;

    @Option(names = "--dataset-dir", defaultValue = "dataset")
    private Path datasetDir;

    @Option(names = "--host", defaultValue = "http://localhost:9200")
    private String host;

    @Option(names = "--model-dir", defaultValue = "models/bge-small-en-v1.5")
    private Path modelDir;

    @Option(names = "--held-out-file", defaultValue = "search-eval/src/main/resources/neural-reranker-eval-queries.txt")
    private Path heldOutFile;

    @Option(names = "--rrf-k", description = "Must match the k the served Hybrid strategy uses", defaultValue = "40")
    private int rrfK;

    @Option(names = "--pool-size", defaultValue = "50")
    private int poolSize;

    @Option(names = "--output", defaultValue = "models/neural-reranker/training-features.tsv")
    private Path output;

    public static void main(String[] args) {
        System.exit(new CommandLine(new RerankFeatureExportCli()).execute(args));
    }

    @Override
    public Integer call() throws Exception {
        WandsDataset dataset = WandsCsvLoader.load(datasetDir);
        Set<String> heldOutIds = new HashSet<>();
        for (String line : Files.readAllLines(heldOutFile, StandardCharsets.UTF_8)) {
            if (!line.isBlank()) {
                heldOutIds.add(line.trim());
            }
        }

        List<List<String>> rowsByQuery = new ArrayList<>();
        try (ElasticsearchClient client = ElasticsearchClients.create(host);
             EmbeddingService embeddingService = new EmbeddingService(modelDir);
             ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            ProductFeatureCache featureCache = new ProductFeatureCache(
                    ids -> ProductLookup.fetchFeaturesByIds(client, ids), FEATURE_CACHE_MAX_SIZE);
            QueryEncoder queryEncoder = new CachingQueryEncoder(embeddingService, QUERY_CACHE_SIZE);
            HybridRrfSearchStrategy hybrid = new HybridRrfSearchStrategy(
                    new LexicalSearchStrategy(client), new SemanticSearchStrategy(client, queryEncoder), poolSize, rrfK);

            List<Future<List<String>>> futures = new ArrayList<>();
            for (WandsQuery query : dataset.queries()) {
                String split = heldOutIds.contains(query.queryId()) ? "held_out" : "train";
                futures.add(executor.submit(() -> rowsFor(query, split, hybrid, featureCache, queryEncoder, dataset)));
            }
            for (Future<List<String>> future : futures) {
                rowsByQuery.add(future.get());
            }
        }

        Files.createDirectories(output.toAbsolutePath().getParent());
        long rowCount = 0;
        try (BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            writer.write("query_id\tsplit\tproduct_id\tgrade\t" + String.join("\t", RerankFeatureBuilder.FEATURE_NAMES));
            writer.newLine();
            for (List<String> rows : rowsByQuery) {
                for (String row : rows) {
                    writer.write(row);
                    writer.newLine();
                    rowCount++;
                }
            }
        }
        log.info("Wrote {} candidate rows for {} queries (rrf k={}, pool {}) to {}",
                rowCount, rowsByQuery.size(), rrfK, poolSize, output);
        return 0;
    }

    private List<String> rowsFor(WandsQuery query, String split, HybridRrfSearchStrategy hybrid,
                                 ProductFeatureCache featureCache, QueryEncoder queryEncoder, WandsDataset dataset) {
        // The same pool cut NeuralRerankStrategy serves: the top poolSize fused candidates.
        List<HybridCandidate> candidates = hybrid.searchWithSignals(query.queryText(), poolSize);
        List<float[]> vectors = NeuralRerankStrategy.buildFeatures(featureCache, queryEncoder, query.queryText(), candidates);
        Map<String, RelevanceGrade> judgments = dataset.judgmentsFor(query.queryId());

        List<String> rows = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            float[] vector = vectors.get(i);
            if (vector == null) {
                continue;
            }
            String productId = candidates.get(i).productId();
            RelevanceGrade grade = judgments.get(productId);
            StringBuilder row = new StringBuilder()
                    .append(query.queryId()).append('\t').append(split).append('\t').append(productId).append('\t')
                    .append(grade == null ? 0 : grade.grade());
            for (float value : vector) {
                row.append('\t').append(String.format(Locale.ROOT, "%.6g", value));
            }
            rows.add(row.toString());
        }
        return rows;
    }
}
