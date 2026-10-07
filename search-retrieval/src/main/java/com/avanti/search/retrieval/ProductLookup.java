package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fetches product content by id — the shared foundation for the two
 * callers that need document content rather than just an id/score:
 * reranking (ProductTextFetcher) and the API's display layer.
 */
public final class ProductLookup {

    private static final List<String> DISPLAY_FIELDS = List.of(
            "product_name", "product_class", "category_hierarchy", "product_description",
            "rating_count", "average_rating", "review_count");

    private ProductLookup() {
    }

    public static Map<String, ProductDocument> fetchByIds(ElasticsearchClient client, List<String> productIds) {
        if (productIds.isEmpty()) {
            return Map.of();
        }

        try {
            SearchResponse<ProductDocument> response = client.search(s -> s
                            .index(SearchConstants.PRODUCTS_INDEX)
                            .size(productIds.size())
                            .query(q -> q.ids(i -> i.values(productIds)))
                            .source(src -> src.filter(f -> f.includes(DISPLAY_FIELDS))),
                    ProductDocument.class);

            Map<String, ProductDocument> documents = new HashMap<>();
            for (Hit<ProductDocument> hit : response.hits().hits()) {
                if (hit.source() != null) {
                    documents.put(hit.id(), hit.source());
                }
            }
            return documents;
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to fetch products by id", e);
        }
    }
}
