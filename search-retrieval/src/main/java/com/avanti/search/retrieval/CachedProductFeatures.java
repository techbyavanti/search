package com.avanti.search.retrieval;

/**
 * The subset of a product's fields needed to build rerank feature vectors
 * ({@link RerankFeatureBuilder}) and cross-encoder rerank text
 * (EmbeddingTextBuilder), cached by {@link ProductFeatureCache} so repeat
 * requests for the same product don't refetch it from Elasticsearch.
 */
public record CachedProductFeatures(
        String productId,
        float[] embedding,
        String productName,
        String productClass,
        String categoryHierarchy,
        String productDescription,
        Double averageRating,
        Integer reviewCount
) {
    public static CachedProductFeatures from(ProductDocument document) {
        return new CachedProductFeatures(
                document.productId(),
                document.embedding(),
                document.productName(),
                document.productClass(),
                document.categoryHierarchy(),
                document.productDescription(),
                document.averageRating(),
                document.reviewCount()
        );
    }
}
