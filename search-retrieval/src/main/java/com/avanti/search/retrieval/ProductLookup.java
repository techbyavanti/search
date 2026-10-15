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

    /** Display fields plus the product embedding, which rerank features need for cosine similarity. */
    private static final List<String> FEATURE_FIELDS = List.of(
            "product_name", "product_class", "category_hierarchy", "product_description",
            "rating_count", "average_rating", "review_count", "embedding");

    private ProductLookup() {
    }

    public static Map<String, ProductDocument> fetchByIds(ElasticsearchClient client, List<String> productIds) {
        return fetch(client, productIds, DISPLAY_FIELDS);
    }

    /** For {@link ProductFeatureCache}: also returns the embedding, which display callers don't need. */
    public static Map<String, ProductDocument> fetchFeaturesByIds(ElasticsearchClient client, List<String> productIds) {
        return fetch(client, productIds, FEATURE_FIELDS);
    }

    private static Map<String, ProductDocument> fetch(ElasticsearchClient client, List<String> productIds, List<String> fields) {
        if (productIds.isEmpty()) {
            return Map.of();
        }

        try {
            SearchResponse<ProductDocument> response = client.search(s -> s
                            .index(SearchConstants.PRODUCTS_INDEX)
                            .size(productIds.size())
                            .query(q -> q.ids(i -> i.values(productIds)))
                            .source(src -> src.filter(f -> f.includes(fields))),
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
