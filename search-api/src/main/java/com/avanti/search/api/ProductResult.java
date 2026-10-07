package com.avanti.search.api;

public record ProductResult(
        String productId,
        double score,
        String productName,
        String productClass,
        String categoryHierarchy,
        Double averageRating,
        Integer ratingCount
) {
}
