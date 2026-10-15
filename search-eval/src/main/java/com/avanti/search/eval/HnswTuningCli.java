package com.avanti.search.eval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.FieldValue;
import co.elastic.clients.elasticsearch._types.SortOrder;
import co.elastic.clients.elasticsearch.core.BulkRequest;
import co.elastic.clients.elasticsearch.core.BulkResponse;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.json.JsonData;
import com.avanti.search.common.RelevanceGrade;
import com.avanti.search.common.WandsCsvLoader;
import com.avanti.search.common.WandsDataset;
import com.avanti.search.common.WandsQuery;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.ProductDocument;
import com.avanti.search.retrieval.SearchConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.ByteArrayInputStream;
import java.io.IOException;
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
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

/**
 * Tunes approximate kNN (HNSW) for the product embeddings. For each vector
 * storage type and graph setting, it builds a scratch index holding only the
 * vectors, force-merges it to one segment, and measures each query-time
 * {@code num_candidates} setting against ground truth: the exact top-K
 * neighbors, computed by brute force in Java. Reports recall against exact
 * search, nDCG@10 against the WANDS labels, serial search latency, build
 * time, and index size. It recommends the setting that reaches the recall
 * target with the fewest {@code num_candidates} (the graph work per query),
 * breaking ties by build time: at this catalog size every setting answers
 * in a few milliseconds, mostly request overhead, so latency alone is too
 * noisy to choose by. It also times exact kNN in Elasticsearch
 * ({@code script_score}) as the reference ANN is compared with.
 *
 * <p>Latency is the Elasticsearch search call only (queries are embedded
 * up front), one query at a time.
 *
 * <p>Usage: java -cp search-eval/target/search-eval.jar com.avanti.search.eval.HnswTuningCli
 */
@Command(name = "hnsw-tuning", mixinStandardHelpOptions = true,
        description = "Sweeps HNSW index and query parameters against exact kNN.")
public final class HnswTuningCli implements Callable<Integer> {

    private static final Logger log = LoggerFactory.getLogger(HnswTuningCli.class);

    private static final String TUNING_INDEX_PREFIX = "hnsw-tuning-";
    private static final String VECTOR_FIELD = "v";
    private static final int TOP_K = 50;
    private static final int RECALL_SMALL_K = 10;
    private static final int NDCG_K = 10;
    private static final int BULK_BATCH_SIZE = 2_000;
    private static final int SCAN_PAGE_SIZE = 5_000;
    private static final int WARMUP_QUERIES = 20;
    private static final int EXACT_LATENCY_QUERIES = 50;
    private static final int SIZE_POLL_ATTEMPTS = 30;
    private static final long SIZE_POLL_INTERVAL_MILLIS = 500;
    /** Raw float vectors alone are ~66 MB (42,994 × 384 × 4 bytes); anything far below is a stale reading. */
    private static final long MIN_PLAUSIBLE_INDEX_BYTES = 10_000_000;

    @Option(names = "--dataset-dir", defaultValue = "dataset")
    private Path datasetDir;

    @Option(names = "--host", defaultValue = "http://localhost:9200")
    private String host;

    @Option(names = "--model-dir", defaultValue = "models/bge-small-en-v1.5")
    private Path modelDir;

    @Option(names = "--configs", split = ",", description = "type:m:ef_construction entries",
            defaultValue = "bbq_hnsw:16:100,int8_hnsw:16:100,hnsw:16:100,hnsw:32:200")
    private List<String> configs;

    @Option(names = "--multipliers", split = ",", description = "num_candidates = k × multiplier",
            defaultValue = "1,2,4,10,20")
    private List<Integer> multipliers;

    @Option(names = "--recall-target", description = "Minimum Recall@50 against exact kNN for the recommendation",
            defaultValue = "0.98")
    private double recallTarget;

    @Option(names = "--output", defaultValue = "results/hnsw-sweep.csv")
    private Path output;

    record HnswConfig(String type, int m, int efConstruction) {
        static HnswConfig parse(String spec) {
            String[] parts = spec.split(":");
            return new HnswConfig(parts[0], Integer.parseInt(parts[1]), Integer.parseInt(parts[2]));
        }

        String label() {
            return type + " m=" + m + " ef=" + efConstruction;
        }

        /** One scratch index per config, so a deleted index's stats never leak into the next measurement. */
        String indexName() {
            return TUNING_INDEX_PREFIX + type.replace('_', '-') + "-m" + m + "-ef" + efConstruction;
        }
    }

    record SweepRow(HnswConfig config, int numCandidates, double recallAt10, double recallAt50, double ndcgAt10,
                    long p50Micros, long p95Micros, long buildMillis, long sizeBytes) {
    }

    private record EncodedQuery(WandsQuery query, float[] vector, List<String> exactTopK) {
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new HnswTuningCli()).execute(args));
    }

    @Override
    public Integer call() throws Exception {
        WandsDataset dataset = WandsCsvLoader.load(datasetDir);
        List<SweepRow> rows = new ArrayList<>();

        try (ElasticsearchClient client = ElasticsearchClients.create(host);
             EmbeddingService embeddingService = new EmbeddingService(modelDir)) {
            List<ProductDocument> products = loadProductVectors(client);
            log.info("Loaded {} product vectors", products.size());

            List<EncodedQuery> queries = encodeWithExactNeighbors(dataset.queries(), embeddingService, products);
            log.info("Computed exact top-{} for {} queries", TOP_K, queries.size());

            long exactP95Micros = -1;
            for (String spec : configs) {
                HnswConfig config = HnswConfig.parse(spec);
                String index = config.indexName();
                long buildMillis = buildTuningIndex(client, index, config, products);
                long sizeBytes = indexSizeBytes(client, index);
                log.info("{}: built in {} ms, {} MB", config.label(), buildMillis, sizeBytes / 1_000_000);
                for (int multiplier : multipliers) {
                    SweepRow row = measure(client, index, config, TOP_K * multiplier, queries, dataset, buildMillis, sizeBytes);
                    rows.add(row);
                    log.info("  num_candidates={}: recall@10={} recall@50={} nDCG@10={} p50={} µs p95={} µs",
                            row.numCandidates(), format(row.recallAt10()), format(row.recallAt50()),
                            format(row.ndcgAt10()), row.p50Micros(), row.p95Micros());
                }
                if (exactP95Micros < 0) {
                    exactP95Micros = exactKnnP95Micros(client, index, queries);
                }
                client.indices().delete(d -> d.index(index));
            }
            double exactNdcg = queries.stream()
                    .mapToDouble(q -> MetricsCalculator.ndcgAtK(q.exactTopK(), dataset.judgmentsFor(q.query().queryId()), NDCG_K))
                    .average().orElse(0.0);
            log.info("Exact kNN: nDCG@10 {}, p95 {} µs (script_score brute force)", format(exactNdcg), exactP95Micros);

            writeCsv(rows);
            recommend(rows, exactP95Micros);
        }
        return 0;
    }

    private static List<ProductDocument> loadProductVectors(ElasticsearchClient client) throws IOException {
        List<ProductDocument> products = new ArrayList<>();
        List<FieldValue> searchAfter = null;
        while (true) {
            List<FieldValue> after = searchAfter;
            SearchResponse<ProductDocument> page = client.search(s -> {
                s.index(SearchConstants.PRODUCTS_INDEX)
                        .size(SCAN_PAGE_SIZE)
                        .sort(o -> o.field(f -> f.field("product_id").order(SortOrder.Asc)))
                        .source(src -> src.filter(f -> f.includes("product_id", "embedding")));
                if (after != null) {
                    s.searchAfter(after);
                }
                return s;
            }, ProductDocument.class);
            List<Hit<ProductDocument>> hits = page.hits().hits();
            if (hits.isEmpty()) {
                return products;
            }
            for (Hit<ProductDocument> hit : hits) {
                if (hit.source() != null && hit.source().embedding() != null) {
                    products.add(hit.source());
                }
            }
            searchAfter = hits.getLast().sort();
        }
    }

    private static List<EncodedQuery> encodeWithExactNeighbors(List<WandsQuery> queries, EmbeddingService embeddingService,
                                                               List<ProductDocument> products) {
        List<float[]> vectors = queries.stream().map(q -> embeddingService.embedQuery(q.queryText())).toList();
        return IntStream.range(0, queries.size()).parallel()
                .mapToObj(i -> new EncodedQuery(queries.get(i), vectors.get(i), exactTopK(vectors.get(i), products)))
                .toList();
    }

    /** Brute-force top-K by dot product; vectors are L2-normalized, so this is cosine order. */
    static List<String> exactTopK(float[] query, List<ProductDocument> products) {
        PriorityQueue<Map.Entry<String, Float>> best = new PriorityQueue<>(Map.Entry.comparingByValue());
        for (ProductDocument product : products) {
            float score = dot(query, product.embedding());
            if (best.size() < TOP_K) {
                best.add(Map.entry(product.productId(), score));
            } else if (score > best.peek().getValue()) {
                best.poll();
                best.add(Map.entry(product.productId(), score));
            }
        }
        return best.stream()
                .sorted(Map.Entry.<String, Float>comparingByValue().reversed())
                .map(Map.Entry::getKey)
                .toList();
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static long buildTuningIndex(ElasticsearchClient client, String index, HnswConfig config,
                                         List<ProductDocument> products) throws IOException {
        if (client.indices().exists(e -> e.index(index)).value()) {
            client.indices().delete(d -> d.index(index));
        }
        String mapping = String.format(Locale.ROOT, """
                {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}},
                 "mappings": {"properties": {
                   "product_id": {"type": "keyword"},
                   "%s": {"type": "dense_vector", "dims": %d, "index": true, "similarity": "cosine",
                          "index_options": {"type": "%s", "m": %d, "ef_construction": %d}}}}}""",
                VECTOR_FIELD, EmbeddingService.DIMENSIONS, config.type(), config.m(), config.efConstruction());
        client.indices().create(c -> c.index(index)
                .withJson(new ByteArrayInputStream(mapping.getBytes(StandardCharsets.UTF_8))));

        long start = System.nanoTime();
        for (int from = 0; from < products.size(); from += BULK_BATCH_SIZE) {
            List<ProductDocument> batch = products.subList(from, Math.min(from + BULK_BATCH_SIZE, products.size()));
            BulkRequest.Builder bulk = new BulkRequest.Builder();
            for (ProductDocument product : batch) {
                Map<String, Object> doc = Map.of("product_id", product.productId(), VECTOR_FIELD, product.embedding());
                bulk.operations(op -> op.index(i -> i.index(index).id(product.productId()).document(doc)));
            }
            BulkResponse response = client.bulk(bulk.build());
            if (response.errors()) {
                throw new IllegalStateException("Bulk load into " + index + " reported errors");
            }
        }
        client.indices().refresh(r -> r.index(index));
        client.indices().forcemerge(f -> f.index(index).maxNumSegments(1L));
        long buildMillis = (System.nanoTime() - start) / 1_000_000;
        // Make the merged segment visible to index stats before indexSizeBytes reads them.
        client.indices().flush(f -> f.index(index).waitIfOngoing(true));
        client.indices().refresh(r -> r.index(index));
        return buildMillis;
    }

    /** Store stats can lag a just-finished merge, so poll until they report the merged segment. */
    private static long indexSizeBytes(ElasticsearchClient client, String index) throws IOException, InterruptedException {
        long size = 0;
        for (int attempt = 0; attempt < SIZE_POLL_ATTEMPTS; attempt++) {
            size = client.indices().stats(s -> s.index(index)).all().primaries().store().sizeInBytes();
            if (size >= MIN_PLAUSIBLE_INDEX_BYTES) {
                return size;
            }
            Thread.sleep(SIZE_POLL_INTERVAL_MILLIS);
        }
        return size;
    }

    private static SweepRow measure(ElasticsearchClient client, String index, HnswConfig config, int numCandidates,
                                    List<EncodedQuery> queries, WandsDataset dataset, long buildMillis, long sizeBytes) {
        for (int i = 0; i < WARMUP_QUERIES; i++) {
            approximateSearch(client, index, queries.get(i).vector(), numCandidates);
        }

        double recall10 = 0, recall50 = 0, ndcg = 0;
        List<Long> latenciesMicros = new ArrayList<>(queries.size());
        for (EncodedQuery query : queries) {
            long start = System.nanoTime();
            List<String> ids = approximateSearch(client, index, query.vector(), numCandidates);
            latenciesMicros.add((System.nanoTime() - start) / 1_000);

            recall10 += overlap(ids, query.exactTopK(), RECALL_SMALL_K);
            recall50 += overlap(ids, query.exactTopK(), TOP_K);
            Map<String, RelevanceGrade> judgments = dataset.judgmentsFor(query.query().queryId());
            ndcg += MetricsCalculator.ndcgAtK(ids, judgments, NDCG_K);
        }
        int n = queries.size();
        List<Long> sorted = latenciesMicros.stream().sorted().toList();
        return new SweepRow(config, numCandidates, recall10 / n, recall50 / n, ndcg / n,
                percentile(sorted, 0.50), percentile(sorted, 0.95), buildMillis, sizeBytes);
    }

    private static List<String> approximateSearch(ElasticsearchClient client, String index, float[] vector, int numCandidates) {
        List<Float> queryVector = toList(vector);
        try {
            SearchResponse<Void> response = client.search(s -> s
                    .index(index)
                    .size(TOP_K)
                    .source(src -> src.fetch(false))
                    .knn(k -> k.field(VECTOR_FIELD).queryVector(queryVector).k(TOP_K).numCandidates(numCandidates)),
                    Void.class);
            return response.hits().hits().stream().map(Hit::id).toList();
        } catch (IOException e) {
            throw new UncheckedIOException("kNN search failed on " + index, e);
        }
    }

    /** p95 of exact brute-force kNN inside Elasticsearch, over the first queries. */
    private static long exactKnnP95Micros(ElasticsearchClient client, String index, List<EncodedQuery> queries)
            throws IOException {
        List<Long> latencies = new ArrayList<>();
        for (EncodedQuery query : queries.subList(0, Math.min(EXACT_LATENCY_QUERIES, queries.size()))) {
            List<Float> queryVector = toList(query.vector());
            long start = System.nanoTime();
            client.search(s -> s
                    .index(index)
                    .size(TOP_K)
                    .source(src -> src.fetch(false))
                    .query(q -> q.scriptScore(ss -> ss
                            .query(m -> m.matchAll(a -> a))
                            .script(sc -> sc
                                    .source(src -> src.scriptString("cosineSimilarity(params.q, '" + VECTOR_FIELD + "') + 1.0"))
                                    .params("q", JsonData.of(queryVector))))),
                    Void.class);
            latencies.add((System.nanoTime() - start) / 1_000);
        }
        return percentile(latencies.stream().sorted().toList(), 0.95);
    }

    private static List<Float> toList(float[] vector) {
        List<Float> list = new ArrayList<>(vector.length);
        for (float value : vector) {
            list.add(value);
        }
        return list;
    }

    static double overlap(List<String> approximate, List<String> exact, int k) {
        Set<String> truth = new HashSet<>(exact.subList(0, Math.min(k, exact.size())));
        long found = approximate.stream().limit(k).filter(truth::contains).count();
        return (double) found / k;
    }

    private static long percentile(List<Long> sorted, double p) {
        return sorted.get((int) Math.min(sorted.size() - 1, Math.ceil(sorted.size() * p) - 1));
    }

    private void writeCsv(List<SweepRow> rows) throws IOException {
        Files.createDirectories(output.toAbsolutePath().getParent());
        try (Writer writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            writer.write("type,m,ef_construction,num_candidates,recall_at_10,recall_at_50,ndcg_at_10,"
                    + "p50_us,p95_us,build_ms,size_bytes\n");
            for (SweepRow r : rows) {
                writer.write(String.format(Locale.ROOT, "%s,%d,%d,%d,%.4f,%.4f,%.4f,%d,%d,%d,%d%n",
                        r.config().type(), r.config().m(), r.config().efConstruction(), r.numCandidates(),
                        r.recallAt10(), r.recallAt50(), r.ndcgAt10(), r.p50Micros(), r.p95Micros(),
                        r.buildMillis(), r.sizeBytes()));
            }
        }
        log.info("Wrote {}", output);
    }

    private void recommend(List<SweepRow> rows, long exactP95Micros) {
        rows.stream()
                .filter(r -> r.recallAt50() >= recallTarget)
                .min(Comparator.comparingInt(SweepRow::numCandidates).thenComparingLong(SweepRow::buildMillis))
                .ifPresentOrElse(
                        r -> log.info("Recommended: {} num_candidates={} (recall@50 {}, nDCG@10 {}, p95 {} µs vs exact {} µs)",
                                r.config().label(), r.numCandidates(), format(r.recallAt50()), format(r.ndcgAt10()),
                                r.p95Micros(), exactP95Micros),
                        () -> log.warn("No setting reached recall@50 {}", recallTarget));
    }

    private static String format(double value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }
}
