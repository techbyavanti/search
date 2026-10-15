package com.avanti.search.retrieval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Builds the fixed-order feature vectors {@link NeuralRerankStrategy} feeds
 * to the neural reranker's ONNX model. This class is the only
 * implementation of the features: training data comes from
 * {@code RerankFeatureExportCli}, which calls this same code over the same
 * hybrid candidate pools the strategy serves, so training and serving
 * cannot drift apart. {@link #FEATURE_NAMES} is the order; changing it
 * means retraining, and {@code RerankFeatureBuilderTest} pins it.
 *
 * <p>Features are built per candidate pool, not per product, because some
 * are relative to the pool (score over the pool's best, cosine gap to the
 * pool's best). That puts every query on the same scale, which matters
 * since raw BM25 scores vary a lot between queries.
 *
 * <p>Text features compare light-stemmed tokens ({@link #stem}), so "chairs"
 * matches "chair". A missing average rating defaults to
 * {@link #DEFAULT_AVERAGE_RATING} (the catalog mean), with
 * {@code has_rating} telling the model it was missing.
 */
public final class RerankFeatureBuilder {

    public static final List<String> FEATURE_NAMES = List.of(
            "rrf_score_relative",
            "hybrid_reciprocal_rank",
            "lexical_reciprocal_rank",
            "lexical_score_relative",
            "semantic_reciprocal_rank",
            "cosine_similarity",
            "cosine_gap_to_best",
            "name_overlap",
            "class_overlap",
            "category_overlap",
            "description_overlap",
            "query_coverage",
            "name_contains_query",
            "query_token_count",
            "average_rating",
            "log1p_review_count",
            "has_rating"
    );
    public static final int FEATURE_COUNT = FEATURE_NAMES.size();
    public static final double DEFAULT_AVERAGE_RATING = 4.5301;

    private static final Pattern TOKEN_SPLIT = Pattern.compile("[^a-z0-9]+");

    private RerankFeatureBuilder() {
    }

    /** Lowercased, non-alphanumeric-delimited tokens. */
    public static List<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        return List.of(TOKEN_SPLIT.split(text.toLowerCase(Locale.ROOT))).stream()
                .filter(token -> !token.isBlank())
                .toList();
    }

    /** Minimal plural folding: "lamps" → "lamp", "accessories" → "accessory", "glass" stays. */
    static String stem(String token) {
        if (token.length() > 4 && token.endsWith("ies")) {
            return token.substring(0, token.length() - 3) + "y";
        }
        if (token.length() > 3 && token.endsWith("s") && !token.endsWith("ss")) {
            return token.substring(0, token.length() - 1);
        }
        return token;
    }

    static List<String> stemmedTokens(String text) {
        return tokenize(text).stream().map(RerankFeatureBuilder::stem).toList();
    }

    /**
     * One feature vector per candidate, in candidate order. An entry is
     * null when the candidate has no cached product data.
     */
    public static List<float[]> buildPool(List<HybridCandidate> candidates, String query, float[] queryEmbedding,
                                          Map<String, CachedProductFeatures> products) {
        List<String> queryTokens = stemmedTokens(query);
        Float[] cosines = new Float[candidates.size()];
        double maxRrf = 0.0;
        double maxLexical = 0.0;
        double maxCosine = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < candidates.size(); i++) {
            HybridCandidate candidate = candidates.get(i);
            maxRrf = Math.max(maxRrf, candidate.rrfScore());
            maxLexical = Math.max(maxLexical, candidate.lexicalScore());
            CachedProductFeatures product = products.get(candidate.productId());
            if (product != null && product.embedding() != null) {
                cosines[i] = cosineSimilarity(queryEmbedding, product.embedding());
                maxCosine = Math.max(maxCosine, cosines[i]);
            }
        }
        PoolMaxima maxima = new PoolMaxima(maxRrf, maxLexical, maxCosine);

        List<float[]> vectors = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            CachedProductFeatures product = products.get(candidates.get(i).productId());
            vectors.add(product == null ? null : build(candidates.get(i), queryTokens, product, cosines[i], maxima));
        }
        return vectors;
    }

    /** Pool-wide maxima that the relative features divide by or subtract from. */
    private record PoolMaxima(double rrfScore, double lexicalScore, double cosine) {
    }

    /** {@code cosine} is null when the product has no embedding. */
    private static float[] build(HybridCandidate candidate, List<String> queryTokens, CachedProductFeatures product,
                                 Float cosine, PoolMaxima max) {
        List<String> nameTokens = stemmedTokens(product.productName());
        List<String> classTokens = stemmedTokens(product.productClass());
        List<String> categoryTokens = stemmedTokens(product.categoryHierarchy());
        List<String> descriptionTokens = stemmedTokens(product.productDescription());

        float[] f = new float[FEATURE_COUNT];
        f[0] = (float) (max.rrfScore() > 0 ? candidate.rrfScore() / max.rrfScore() : 0.0);
        f[1] = 1.0f / candidate.hybridRank();
        f[2] = candidate.lexicalRank() > 0 ? 1.0f / candidate.lexicalRank() : 0.0f;
        f[3] = (float) (max.lexicalScore() > 0 ? candidate.lexicalScore() / max.lexicalScore() : 0.0);
        f[4] = candidate.semanticRank() > 0 ? 1.0f / candidate.semanticRank() : 0.0f;
        f[5] = cosine != null ? cosine : 0.0f;
        f[6] = cosine != null ? (float) (cosine - max.cosine()) : -1.0f;
        f[7] = overlapFraction(queryTokens, nameTokens);
        f[8] = overlapFraction(queryTokens, classTokens);
        f[9] = overlapFraction(queryTokens, categoryTokens);
        f[10] = overlapFraction(queryTokens, descriptionTokens);
        f[11] = coverage(queryTokens, nameTokens, classTokens, categoryTokens, descriptionTokens);
        f[12] = !queryTokens.isEmpty() && Collections.indexOfSubList(nameTokens, queryTokens) >= 0 ? 1.0f : 0.0f;
        f[13] = queryTokens.size();
        f[14] = (float) (product.averageRating() != null ? product.averageRating() : DEFAULT_AVERAGE_RATING);
        f[15] = (float) Math.log1p(product.reviewCount() != null ? product.reviewCount() : 0);
        f[16] = product.averageRating() != null ? 1.0f : 0.0f;
        return f;
    }

    private static float cosineSimilarity(float[] queryEmbedding, float[] productEmbedding) {
        float dot = 0.0f;
        int length = Math.min(queryEmbedding.length, productEmbedding.length);
        for (int i = 0; i < length; i++) {
            dot += queryEmbedding[i] * productEmbedding[i];
        }
        return dot;
    }

    private static float overlapFraction(List<String> queryTokens, List<String> fieldTokens) {
        if (queryTokens.isEmpty()) {
            return 0.0f;
        }
        long overlap = queryTokens.stream().filter(fieldTokens::contains).count();
        return (float) overlap / queryTokens.size();
    }

    @SafeVarargs
    private static float coverage(List<String> queryTokens, List<String>... fields) {
        if (queryTokens.isEmpty()) {
            return 0.0f;
        }
        long covered = queryTokens.stream()
                .filter(token -> Stream.of(fields).anyMatch(field -> field.contains(token)))
                .count();
        return (float) covered / queryTokens.size();
    }
}
