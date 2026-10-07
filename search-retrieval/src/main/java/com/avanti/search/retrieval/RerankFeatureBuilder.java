package com.avanti.search.retrieval;

import java.util.List;
import java.util.regex.Pattern;

/**
 * Builds the fixed-order feature vector {@link NeuralRerankStrategy} feeds
 * to the neural reranker's ONNX model. This ordering is a contract with the
 * Python training script (training/train_neural_reranker.py) — changing it
 * here without retraining the model silently produces garbage scores, so
 * {@code RerankFeatureBuilderTest} pins the order.
 *
 * <p>Feature 5 (average rating) defaults to {@link #DEFAULT_AVERAGE_RATING}
 * — the WANDS catalog's mean {@code average_rating} across all rated
 * products — when a candidate has no rating, rather than 0, which would
 * otherwise look like a real (terrible) rating to the model.
 */
public final class RerankFeatureBuilder {

    public static final int FEATURE_COUNT = 6;
    public static final double DEFAULT_AVERAGE_RATING = 4.5301;

    private static final Pattern TOKEN_SPLIT = Pattern.compile("[^a-z0-9]+");

    private RerankFeatureBuilder() {
    }

    /** Lowercased, non-alphanumeric-delimited tokens — used for both the query and product-side text below. */
    public static List<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        return List.of(TOKEN_SPLIT.split(text.toLowerCase(java.util.Locale.ROOT))).stream()
                .filter(token -> !token.isBlank())
                .toList();
    }

    public static float[] build(double rrfScore, float[] queryEmbedding, List<String> queryTokens, CachedProductFeatures product) {
        float[] features = new float[FEATURE_COUNT];
        features[0] = (float) rrfScore;
        features[1] = cosineSimilarity(queryEmbedding, product.embedding());
        features[2] = (float) exactTermOverlapFraction(queryTokens, product.productName());
        features[3] = categoryMatch(queryTokens, product.categoryHierarchy(), product.productClass()) ? 1.0f : 0.0f;
        features[4] = (float) (product.averageRating() != null ? product.averageRating() : DEFAULT_AVERAGE_RATING);
        features[5] = (float) Math.log1p(product.reviewCount() != null ? product.reviewCount() : 0);
        return features;
    }

    private static float cosineSimilarity(float[] queryEmbedding, float[] productEmbedding) {
        if (queryEmbedding == null || productEmbedding == null) {
            return 0.0f;
        }
        float dot = 0.0f;
        int length = Math.min(queryEmbedding.length, productEmbedding.length);
        for (int i = 0; i < length; i++) {
            dot += queryEmbedding[i] * productEmbedding[i];
        }
        return dot;
    }

    private static double exactTermOverlapFraction(List<String> queryTokens, String productName) {
        if (queryTokens.isEmpty()) {
            return 0.0;
        }
        List<String> nameTokens = tokenize(productName);
        long overlap = queryTokens.stream().filter(nameTokens::contains).count();
        return (double) overlap / queryTokens.size();
    }

    private static boolean categoryMatch(List<String> queryTokens, String categoryHierarchy, String productClass) {
        List<String> categoryTokens = tokenize((categoryHierarchy == null ? "" : categoryHierarchy) + " " + (productClass == null ? "" : productClass));
        return queryTokens.stream().anyMatch(categoryTokens::contains);
    }
}
