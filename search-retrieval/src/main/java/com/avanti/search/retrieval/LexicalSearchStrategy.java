package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.query_dsl.TextQueryType;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;

/** Pure BM25 full-text search via a boosted multi_match query. */
public final class LexicalSearchStrategy implements SearchStrategy {

    private static final List<String> SEARCH_FIELDS = List.of(
            "product_name^3",
            "product_class^2",
            "category_hierarchy",
            "product_description",
            "product_features"
    );

    private final ElasticsearchClient client;

    public LexicalSearchStrategy(ElasticsearchClient client) {
        this.client = client;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        try {
            SearchResponse<Void> response = client.search(s -> s
                            .index(SearchConstants.PRODUCTS_INDEX)
                            .size(topK)
                            .source(src -> src.fetch(false))
                            .query(q -> q
                                    .multiMatch(m -> m
                                            .query(query)
                                            .fields(SEARCH_FIELDS)
                                            .type(TextQueryType.BestFields)
                                    )
                            ),
                    Void.class
            );

            return response.hits().hits().stream()
                    .map(hit -> new ScoredResult(hit.id(), hit.score() == null ? 0.0 : hit.score()))
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException("Lexical search failed for query: " + query, e);
        }
    }

    @Override
    public String name() {
        return "Lexical (BM25)";
    }
}
