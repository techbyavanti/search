package com.avanti.search.retrieval;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

class RerankFeatureBuilderTest {

    private static final float[] QUERY_EMBEDDING = {1.0f, 0.0f, 0.0f};

    private static int index(String featureName) {
        return RerankFeatureBuilder.FEATURE_NAMES.indexOf(featureName);
    }

    @Test
    void featureOrderIsTheTrainedModelsContract() {
        assertThat(RerankFeatureBuilder.FEATURE_NAMES).containsExactly(
                "rrf_score_relative", "hybrid_reciprocal_rank", "lexical_reciprocal_rank", "lexical_score_relative",
                "semantic_reciprocal_rank", "cosine_similarity", "cosine_gap_to_best", "name_overlap",
                "class_overlap", "category_overlap", "description_overlap", "query_coverage",
                "name_contains_query", "query_token_count", "average_rating", "log1p_review_count", "has_rating");
        assertThat(RerankFeatureBuilder.FEATURE_COUNT).isEqualTo(17);
    }

    @Test
    void buildsRetrieverSignalsRelativeToThePool() {
        List<HybridCandidate> pool = List.of(
                new HybridCandidate("a", 1, 0.04, 1, 20.0, 2, 0.9),
                new HybridCandidate("b", 2, 0.02, 0, 0.0, 1, 0.95));
        Map<String, CachedProductFeatures> products = Map.of(
                "a", product("a", new float[]{0.8f, 0.6f, 0.0f}, "Oak Dining Table"),
                "b", product("b", new float[]{0.6f, 0.8f, 0.0f}, "Pine Desk"));

        List<float[]> vectors = RerankFeatureBuilder.buildPool(pool, "oak table", QUERY_EMBEDDING, products);
        float[] a = vectors.get(0);
        float[] b = vectors.get(1);

        assertThat(a).hasSize(RerankFeatureBuilder.FEATURE_COUNT);
        assertThat(a[index("rrf_score_relative")]).isEqualTo(1.0f);
        assertThat(b[index("rrf_score_relative")]).isEqualTo(0.5f);
        assertThat(b[index("hybrid_reciprocal_rank")]).isEqualTo(0.5f);
        assertThat(a[index("lexical_reciprocal_rank")]).isEqualTo(1.0f);
        assertThat(b[index("lexical_reciprocal_rank")]).isEqualTo(0.0f);   // lexical did not retrieve b
        assertThat(a[index("lexical_score_relative")]).isEqualTo(1.0f);
        assertThat(a[index("semantic_reciprocal_rank")]).isEqualTo(0.5f);
        assertThat(a[index("cosine_similarity")]).isCloseTo(0.8f, offset(1e-6f));
        assertThat(a[index("cosine_gap_to_best")]).isCloseTo(0.0f, offset(1e-6f));
        assertThat(b[index("cosine_gap_to_best")]).isCloseTo(-0.2f, offset(1e-6f));
    }

    @Test
    void textFeaturesMatchAcrossFieldsWithPluralFolding() {
        List<HybridCandidate> pool = List.of(new HybridCandidate("p", 1, 0.03, 1, 10.0, 1, 0.9));
        CachedProductFeatures product = new CachedProductFeatures("p", new float[]{1.0f, 0.0f, 0.0f},
                "Modern Dining Chairs", "Dining Chairs", "Furniture / Kitchen & Dining", "Solid oak frame.", 4.2, 150);

        float[] f = RerankFeatureBuilder.buildPool(pool, "oak dining chair", QUERY_EMBEDDING, Map.of("p", product)).get(0);

        assertThat(f[index("name_overlap")]).isCloseTo(2f / 3, offset(1e-6f));        // dining, chair(s)
        assertThat(f[index("class_overlap")]).isCloseTo(2f / 3, offset(1e-6f));
        assertThat(f[index("category_overlap")]).isCloseTo(1f / 3, offset(1e-6f));    // dining
        assertThat(f[index("description_overlap")]).isCloseTo(1f / 3, offset(1e-6f)); // oak
        assertThat(f[index("query_coverage")]).isEqualTo(1.0f);
        assertThat(f[index("name_contains_query")]).isEqualTo(0.0f);                   // "oak" not in the name
        assertThat(f[index("query_token_count")]).isEqualTo(3.0f);
        assertThat(f[index("average_rating")]).isEqualTo(4.2f);
        assertThat(f[index("log1p_review_count")]).isCloseTo((float) Math.log1p(150), offset(1e-6f));
        assertThat(f[index("has_rating")]).isEqualTo(1.0f);
    }

    @Test
    void nameContainsQueryNeedsTheTokensInOrder() {
        List<HybridCandidate> pool = List.of(new HybridCandidate("p", 1, 0.03, 1, 10.0, 1, 0.9));
        Map<String, CachedProductFeatures> products = Map.of("p", product("p", QUERY_EMBEDDING, "Rustic Coffee Tables Set"));

        assertThat(RerankFeatureBuilder.buildPool(pool, "coffee table", QUERY_EMBEDDING, products).get(0)
                [index("name_contains_query")]).isEqualTo(1.0f);
        assertThat(RerankFeatureBuilder.buildPool(pool, "table coffee", QUERY_EMBEDDING, products).get(0)
                [index("name_contains_query")]).isEqualTo(0.0f);
    }

    @Test
    void defaultsMissingRatingAndFlagsIt() {
        List<HybridCandidate> pool = List.of(new HybridCandidate("p", 1, 0.03, 1, 10.0, 1, 0.9));
        CachedProductFeatures product = new CachedProductFeatures("p", QUERY_EMBEDDING, "Chair", null, null, null, null, null);

        float[] f = RerankFeatureBuilder.buildPool(pool, "chair", QUERY_EMBEDDING, Map.of("p", product)).get(0);

        assertThat(f[index("average_rating")]).isEqualTo((float) RerankFeatureBuilder.DEFAULT_AVERAGE_RATING);
        assertThat(f[index("has_rating")]).isEqualTo(0.0f);
        assertThat(f[index("log1p_review_count")]).isEqualTo(0.0f);
    }

    @Test
    void candidatesWithoutProductDataGetNoVector() {
        List<HybridCandidate> pool = List.of(new HybridCandidate("missing", 1, 0.03, 1, 10.0, 0, 0.0));

        assertThat(RerankFeatureBuilder.buildPool(pool, "chair", QUERY_EMBEDDING, Map.of())).hasSize(1).containsOnlyNulls();
    }

    @Test
    void stemFoldsSimplePlurals() {
        assertThat(RerankFeatureBuilder.stem("lamps")).isEqualTo("lamp");
        assertThat(RerankFeatureBuilder.stem("accessories")).isEqualTo("accessory");
        assertThat(RerankFeatureBuilder.stem("glass")).isEqualTo("glass");
        assertThat(RerankFeatureBuilder.stem("bus")).isEqualTo("bus");
    }

    @Test
    void tokenizeLowercasesAndSplitsOnNonAlphanumerics() {
        assertThat(RerankFeatureBuilder.tokenize("Modern Oak Dining-Table!"))
                .containsExactly("modern", "oak", "dining", "table");
        assertThat(RerankFeatureBuilder.tokenize(null)).isEmpty();
        assertThat(RerankFeatureBuilder.tokenize("  ")).isEmpty();
    }

    private static CachedProductFeatures product(String id, float[] embedding, String name) {
        return new CachedProductFeatures(id, embedding, name, null, null, null, 4.5, 10);
    }
}
