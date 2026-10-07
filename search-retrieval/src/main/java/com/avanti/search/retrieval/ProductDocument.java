package com.avanti.search.retrieval;

import com.avanti.search.common.WandsProduct;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * The Elasticsearch document shape for a product — field names match
 * products-mapping.json, which is why this doesn't just reuse WandsProduct
 * (whose accessor names are camelCase Java convention, not the mapping's
 * snake_case). Ranking strategies fetch just the doc id and score (see
 * SearchConstants / LexicalSearchStrategy) since that's all a ranking
 * decision needs; ProductLookup deserializes this for the few callers that
 * need product content instead — reranking and the API's display layer.
 */
public record ProductDocument(
        @JsonProperty("product_id") String productId,
        @JsonProperty("product_name") String productName,
        @JsonProperty("product_class") String productClass,
        @JsonProperty("category_hierarchy") String categoryHierarchy,
        @JsonProperty("product_description") String productDescription,
        @JsonProperty("product_features") String productFeatures,
        @JsonProperty("rating_count") Integer ratingCount,
        @JsonProperty("average_rating") Double averageRating,
        @JsonProperty("review_count") Integer reviewCount,
        @JsonProperty("embedding") float[] embedding,
        @JsonProperty("learned_embedding") float[] learnedEmbedding
) {
    public static ProductDocument from(WandsProduct product, float[] embedding) {
        return new ProductDocument(
                product.productId(),
                product.productName(),
                product.productClass(),
                product.categoryHierarchy(),
                product.productDescription(),
                product.productFeatures(),
                product.ratingCount(),
                product.averageRating(),
                product.reviewCount(),
                embedding,
                null
        );
    }

    /** Used only by the optional --learned-model-dir ingestion pass (see Track B's tower comparison). */
    public ProductDocument withLearnedEmbedding(float[] learnedEmbedding) {
        return new ProductDocument(productId, productName, productClass, categoryHierarchy, productDescription,
                productFeatures, ratingCount, averageRating, reviewCount, embedding, learnedEmbedding);
    }
}
