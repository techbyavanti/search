package com.avanti.search.retrieval;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

class RerankFeatureBuilderTest {

    @Test
    void buildsFeaturesInTheFixedContractOrder() {
        float[] queryEmbedding = {1.0f, 0.0f, 0.0f};
        CachedProductFeatures product = new CachedProductFeatures(
                "p1", new float[]{0.5f, 0.5f, 0.0f}, "Oak Dining Table", "Tables",
                "Furniture / Dining Room", "A sturdy oak table.", 4.2, 150);

        float[] features = RerankFeatureBuilder.build(0.031, queryEmbedding, List.of("oak", "table"), product);

        assertThat(features).hasSize(RerankFeatureBuilder.FEATURE_COUNT);
        assertThat(features[0]).isEqualTo(0.031f);                    // rrfScore, passed through as-is
        assertThat(features[1]).isCloseTo(0.5f, offset(1e-6f));       // cosineSimilarity (dot product of normalized vectors)
        assertThat(features[2]).isEqualTo(1.0f);                      // exactTermOverlapFraction: "oak" and "table" both in the name
        assertThat(features[3]).isEqualTo(0.0f);                      // categoryMatch: neither token in "Furniture / Dining Room" / "Tables"
        assertThat(features[4]).isEqualTo(4.2f);                      // averageRating, present
        assertThat(features[5]).isCloseTo((float) Math.log1p(150), offset(1e-6f)); // log1p(reviewCount)
    }

    @Test
    void defaultsAverageRatingWhenNull() {
        CachedProductFeatures product = new CachedProductFeatures(
                "p1", new float[]{1.0f}, "Chair", "Chairs", "Furniture / Chairs", null, null, null);

        float[] features = RerankFeatureBuilder.build(0.0, new float[]{1.0f}, List.of("chair"), product);

        assertThat(features[4]).isEqualTo((float) RerankFeatureBuilder.DEFAULT_AVERAGE_RATING);
        assertThat(features[5]).isEqualTo(0.0f);
    }

    @Test
    void categoryMatchLooksAtBothCategoryHierarchyAndProductClass() {
        CachedProductFeatures product = new CachedProductFeatures(
                "p1", new float[]{1.0f}, "Widget", "Lamps", "Furniture / Lighting", null, null, null);

        float[] byClass = RerankFeatureBuilder.build(0.0, new float[]{1.0f}, List.of("lamps"), product);
        float[] byHierarchy = RerankFeatureBuilder.build(0.0, new float[]{1.0f}, List.of("lighting"), product);
        float[] noMatch = RerankFeatureBuilder.build(0.0, new float[]{1.0f}, List.of("nonexistent"), product);

        assertThat(byClass[3]).isEqualTo(1.0f);
        assertThat(byHierarchy[3]).isEqualTo(1.0f);
        assertThat(noMatch[3]).isEqualTo(0.0f);
    }

    @Test
    void tokenizeLowercasesAndSplitsOnNonAlphanumerics() {
        assertThat(RerankFeatureBuilder.tokenize("Modern Oak Dining-Table!"))
                .containsExactly("modern", "oak", "dining", "table");
        assertThat(RerankFeatureBuilder.tokenize(null)).isEmpty();
        assertThat(RerankFeatureBuilder.tokenize("  ")).isEmpty();
    }
}
