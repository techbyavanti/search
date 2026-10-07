package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import com.avanti.search.inference.EmbeddingService;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Dense kNN search over a WANDS-fine-tuned tower's embeddings (stored in
 * the {@code learned_embedding} field, populated via IngestionCli's
 * --learned-model-dir pass — see training/train_embedding_towers.py),
 * structurally identical to SemanticSearchStrategy's off-the-shelf
 * bge-small-en-v1.5 search but against a different field/model.
 *
 * <p>The shared-tower mode (one encoder for both queries and products) won
 * the head-to-head comparison against a true two-tower model on the held-out
 * split — nDCG@10 0.6594 vs. 0.6468, a clean sweep across every metric (see
 * TRAINING.md) — plausibly because two-tower's extra capacity (two
 * independently-trained encoders) is less data-efficient than one shared
 * encoder when fine-tuning on only ~5.4K triplets. So the
 * {@code queryTowerEmbeddingService} wired in here is the shared tower —
 * the same model/instance used to embed products at ingestion time.
 */
public final class LearnedTowerSearchStrategy implements SearchStrategy {

    private static final String LEARNED_EMBEDDING_FIELD = "learned_embedding";
    private static final int NUM_CANDIDATES_MULTIPLIER = 10;

    private final ElasticsearchClient client;
    private final EmbeddingService queryTowerEmbeddingService;

    public LearnedTowerSearchStrategy(ElasticsearchClient client, EmbeddingService queryTowerEmbeddingService) {
        this.client = client;
        this.queryTowerEmbeddingService = queryTowerEmbeddingService;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        float[] queryEmbedding = queryTowerEmbeddingService.embedQuery(query);
        List<Float> queryVector = new ArrayList<>(queryEmbedding.length);
        for (float value : queryEmbedding) {
            queryVector.add(value);
        }

        try {
            SearchResponse<Void> response = client.search(s -> s
                            .index(SearchConstants.PRODUCTS_INDEX)
                            .size(topK)
                            .source(src -> src.fetch(false))
                            .knn(k -> k
                                    .field(LEARNED_EMBEDDING_FIELD)
                                    .queryVector(queryVector)
                                    .k(topK)
                                    .numCandidates(topK * NUM_CANDIDATES_MULTIPLIER)
                            ),
                    Void.class
            );

            return response.hits().hits().stream()
                    .map(hit -> new ScoredResult(hit.id(), hit.score() == null ? 0.0 : hit.score()))
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException("Learned-tower search failed for query: " + query, e);
        }
    }

    @Override
    public String name() {
        return "Learned Tower (fine-tuned, shared)";
    }
}
