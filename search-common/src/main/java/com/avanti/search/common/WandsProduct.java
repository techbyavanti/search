package com.avanti.search.common;

/**
 * A row from WANDS product.csv. {@code productClass}, {@code categoryHierarchy},
 * {@code productDescription}, and the rating fields are frequently blank in the
 * source data and are left null rather than defaulted.
 */
public record WandsProduct(
        String productId,
        String productName,
        String productClass,
        String categoryHierarchy,
        String productDescription,
        String productFeatures,
        Integer ratingCount,
        Double averageRating,
        Integer reviewCount
) {
}
