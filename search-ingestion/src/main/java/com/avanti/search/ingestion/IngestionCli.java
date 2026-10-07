package com.avanti.search.ingestion;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.BulkRequest;
import co.elastic.clients.elasticsearch.core.BulkResponse;
import co.elastic.clients.elasticsearch.core.bulk.BulkResponseItem;
import co.elastic.clients.elasticsearch.indices.CreateIndexRequest;
import com.avanti.search.common.EmbeddingTextBuilder;
import com.avanti.search.common.WandsCsvLoader;
import com.avanti.search.common.WandsDataset;
import com.avanti.search.common.WandsProduct;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.ProductDocument;
import com.avanti.search.retrieval.SearchConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;

@Command(name = "search-ingestion", mixinStandardHelpOptions = true,
        description = "Creates the products index and bulk-indexes the WANDS catalog.")
public class IngestionCli implements Callable<Integer> {

    private static final Logger log = LoggerFactory.getLogger(IngestionCli.class);
    private static final String MAPPING_RESOURCE = "elasticsearch/products-mapping.json";

    @Option(names = "--dataset-dir", description = "Directory containing product.csv/query.csv/label.csv", defaultValue = "dataset")
    private Path datasetDir;

    @Option(names = "--host", description = "Elasticsearch host URL", defaultValue = SearchConstants.DEFAULT_ELASTICSEARCH_HOST)
    private String host;

    @Option(names = "--batch-size", description = "Documents per bulk request", defaultValue = "500")
    private int batchSize;

    @Option(names = "--recreate", description = "Delete and recreate the index if it already exists", defaultValue = "false")
    private boolean recreate;

    @Option(names = "--model-dir", description = "Directory containing the embedding model's model.onnx/tokenizer.json", defaultValue = "models/bge-small-en-v1.5")
    private Path modelDir;

    @Option(names = "--learned-model-dir", description = "If given, skips normal ingestion and instead computes learned_embedding for every product already in the index, using this model dir (see Track B's fine-tuned tower comparison)")
    private Path learnedModelDir;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new IngestionCli()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws IOException {
        log.info("Loading WANDS dataset from {}", datasetDir);
        WandsDataset dataset = WandsCsvLoader.load(datasetDir);
        log.info("Loaded {} products", dataset.products().size());

        try (ElasticsearchClient client = ElasticsearchClients.create(host)) {
            if (learnedModelDir != null) {
                try (EmbeddingService learnedEmbeddingService = new EmbeddingService(learnedModelDir)) {
                    updateLearnedEmbeddings(client, learnedEmbeddingService, dataset.products());
                }
                return 0;
            }

            try (EmbeddingService embeddingService = new EmbeddingService(modelDir)) {
                createIndex(client);
                indexProducts(client, embeddingService, dataset.products());
            }
            client.indices().refresh(r -> r.index(SearchConstants.PRODUCTS_INDEX));

            long count = client.count(c -> c.index(SearchConstants.PRODUCTS_INDEX)).count();
            log.info("Index now contains {} documents", count);
        }

        return 0;
    }

    private void createIndex(ElasticsearchClient client) throws IOException {
        boolean exists = client.indices().exists(e -> e.index(SearchConstants.PRODUCTS_INDEX)).value();
        if (exists) {
            if (!recreate) {
                log.info("Index '{}' already exists, skipping creation (pass --recreate to rebuild)", SearchConstants.PRODUCTS_INDEX);
                return;
            }
            log.info("Deleting existing index '{}'", SearchConstants.PRODUCTS_INDEX);
            client.indices().delete(d -> d.index(SearchConstants.PRODUCTS_INDEX));
        }

        try (InputStream mapping = getClass().getClassLoader().getResourceAsStream(MAPPING_RESOURCE)) {
            if (mapping == null) {
                throw new IllegalStateException("Could not find mapping resource: " + MAPPING_RESOURCE);
            }
            CreateIndexRequest request = CreateIndexRequest.of(b -> b
                    .index(SearchConstants.PRODUCTS_INDEX)
                    .withJson(mapping));
            boolean acknowledged = client.indices().create(request).acknowledged();
            log.info("Created index '{}' (acknowledged={})", SearchConstants.PRODUCTS_INDEX, acknowledged);
        }
    }

    private void indexProducts(ElasticsearchClient client, EmbeddingService embeddingService, List<WandsProduct> products) throws IOException {
        int total = products.size();
        int indexed = 0;

        for (int start = 0; start < total; start += batchSize) {
            int end = Math.min(start + batchSize, total);
            List<WandsProduct> batch = products.subList(start, end);

            List<String> embeddingTexts = batch.stream().map(EmbeddingTextBuilder::build).toList();
            List<float[]> embeddings = embeddingService.embedDocuments(embeddingTexts);

            BulkRequest.Builder bulkRequest = new BulkRequest.Builder();
            for (int i = 0; i < batch.size(); i++) {
                ProductDocument doc = ProductDocument.from(batch.get(i), embeddings.get(i));
                bulkRequest.operations(op -> op
                        .index(idx -> idx
                                .index(SearchConstants.PRODUCTS_INDEX)
                                .id(doc.productId())
                                .document(doc)));
            }

            BulkResponse response = client.bulk(bulkRequest.build());
            if (response.errors()) {
                for (BulkResponseItem item : response.items()) {
                    if (item.error() != null) {
                        log.error("Failed to index product {}: {}", item.id(), item.error().reason());
                    }
                }
            }

            indexed = end;
            log.info("Indexed {}/{} products", indexed, total);
        }
    }

    /**
     * Partial-updates just the learned_embedding field on already-indexed
     * documents — see --learned-model-dir. Resumable: this pass can take a
     * long time over the full ~43K catalog, so progress is checkpointed to
     * a local file (keyed by model dir, so switching candidates doesn't
     * cross-contaminate resume state) and re-running after an interruption
     * picks up where it left off instead of recomputing from scratch.
     */
    private void updateLearnedEmbeddings(ElasticsearchClient client, EmbeddingService learnedEmbeddingService, List<WandsProduct> products) throws IOException {
        int total = products.size();
        Path checkpointFile = checkpointFileFor(learnedModelDir);
        int resumeFrom = readCheckpoint(checkpointFile);
        if (resumeFrom > 0) {
            log.info("Resuming from checkpoint: {}/{} products already updated", resumeFrom, total);
        }

        for (int start = resumeFrom; start < total; start += batchSize) {
            int end = Math.min(start + batchSize, total);
            List<WandsProduct> batch = products.subList(start, end);

            List<String> embeddingTexts = batch.stream().map(EmbeddingTextBuilder::build).toList();
            List<float[]> embeddings = learnedEmbeddingService.embedDocuments(embeddingTexts);

            BulkRequest.Builder bulkRequest = new BulkRequest.Builder();
            for (int i = 0; i < batch.size(); i++) {
                String productId = batch.get(i).productId();
                float[] learnedEmbedding = embeddings.get(i);
                bulkRequest.operations(op -> op
                        .update(u -> u
                                .index(SearchConstants.PRODUCTS_INDEX)
                                .id(productId)
                                .action(a -> a.doc(java.util.Map.of("learned_embedding", learnedEmbedding)))));
            }

            BulkResponse response = client.bulk(bulkRequest.build());
            if (response.errors()) {
                for (BulkResponseItem item : response.items()) {
                    if (item.error() != null) {
                        log.error("Failed to update learned_embedding for product {}: {}", item.id(), item.error().reason());
                    }
                }
            }

            writeCheckpoint(checkpointFile, end);
            log.info("Updated learned_embedding for {}/{} products", end, total);
        }

        client.indices().refresh(r -> r.index(SearchConstants.PRODUCTS_INDEX));
        Files.deleteIfExists(checkpointFile);
        log.info("Done updating learned_embedding for {} products", total);
    }

    private static Path checkpointFileFor(Path learnedModelDir) {
        String key = learnedModelDir.getFileName().toString().replaceAll("[^a-zA-Z0-9-]", "_");
        return Path.of(".learned-embedding-checkpoint-" + key + ".txt");
    }

    private static int readCheckpoint(Path checkpointFile) throws IOException {
        if (!Files.exists(checkpointFile)) {
            return 0;
        }
        return Integer.parseInt(Files.readString(checkpointFile, StandardCharsets.UTF_8).trim());
    }

    private static void writeCheckpoint(Path checkpointFile, int progress) throws IOException {
        Files.writeString(checkpointFile, Integer.toString(progress), StandardCharsets.UTF_8);
    }
}
