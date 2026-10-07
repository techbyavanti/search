package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import com.avanti.search.inference.EmbeddingService;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

/** Dense kNN search over bge-small-en-v1.5 embeddings. */
public final class SemanticSearchStrategy implements SearchStrategy {

    private static final String EMBEDDING_FIELD = "embedding";
    private static final int NUM_CANDIDATES_MULTIPLIER = 10;

    private final ElasticsearchClient client;
    private final EmbeddingService embeddingService;

    public SemanticSearchStrategy(ElasticsearchClient client, EmbeddingService embeddingService) {
        this.client = client;
        this.embeddingService = embeddingService;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        float[] queryEmbedding = embeddingService.embedQuery(query);
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
                                    .field(EMBEDDING_FIELD)
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
            throw new UncheckedIOException("Semantic search failed for query: " + query, e);
        }
    }

    @Override
    public String name() {
        return "Semantic (bge-small-en-v1.5)";
    }
}
