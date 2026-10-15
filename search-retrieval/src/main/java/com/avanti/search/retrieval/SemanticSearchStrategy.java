package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import com.avanti.search.inference.QueryEncoder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Approximate kNN search over bge-small-en-v1.5 embeddings. Elasticsearch
 * answers it from an HNSW graph; how the field's vectors are stored (float,
 * int8, or 1-bit BBQ) and the graph's {@code m}/{@code ef_construction} are
 * set in products-mapping.json. {@code num_candidates} — how many
 * candidates the graph search keeps, the query-time recall/latency knob — is
 * fixed rather than scaled with topK, because recall depends on it, not on
 * how many results are returned.
 *
 * <p>Two instances are served: the default {@code embedding} field (1-bit
 * BBQ HNSW) and {@link #TUNED_HNSW_FIELD} (float HNSW, m=32,
 * ef_construction=200), with parameters chosen by {@code HnswTuningCli}.
 */
public final class SemanticSearchStrategy implements SearchStrategy {

    public static final String DEFAULT_FIELD = "embedding";
    public static final String TUNED_HNSW_FIELD = "embedding_hnsw";
    public static final int DEFAULT_NUM_CANDIDATES = 500;

    private final ElasticsearchClient client;
    private final QueryEncoder queryEncoder;
    private final String vectorField;
    private final int numCandidates;
    private final String name;

    public SemanticSearchStrategy(ElasticsearchClient client, QueryEncoder queryEncoder) {
        this(client, queryEncoder, DEFAULT_FIELD, DEFAULT_NUM_CANDIDATES, "Semantic (bge-small-en-v1.5)");
    }

    /** Semantic search over {@link #TUNED_HNSW_FIELD}. */
    public static SemanticSearchStrategy tunedHnsw(ElasticsearchClient client, QueryEncoder queryEncoder, int numCandidates) {
        return new SemanticSearchStrategy(client, queryEncoder, TUNED_HNSW_FIELD, numCandidates, "Semantic HNSW (float, tuned)");
    }

    public SemanticSearchStrategy(ElasticsearchClient client, QueryEncoder queryEncoder, String vectorField,
                                  int numCandidates, String name) {
        if (numCandidates < 1) {
            throw new IllegalArgumentException("numCandidates must be at least 1, got " + numCandidates);
        }
        this.client = client;
        this.queryEncoder = queryEncoder;
        this.vectorField = vectorField;
        this.numCandidates = numCandidates;
        this.name = name;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        float[] queryEmbedding = queryEncoder.embedQuery(query);
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
                                    .field(vectorField)
                                    .queryVector(queryVector)
                                    .k(topK)
                                    .numCandidates(Math.max(topK, numCandidates))
                            ),
                    Void.class
            );

            return response.hits().hits().stream()
                    .map(hit -> new ScoredResult(hit.id(), hit.score() == null ? 0.0 : hit.score()))
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException(name + " search failed for query: " + query, e);
        }
    }

    @Override
    public String name() {
        return name;
    }
}
