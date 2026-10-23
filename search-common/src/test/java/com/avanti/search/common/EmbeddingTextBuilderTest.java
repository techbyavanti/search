package com.avanti.search.common;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class EmbeddingTextBuilderTest {

    @Test
    void categoryHierarchyDropsWhitespaceAroundEachSlash() {
        assertThat(EmbeddingTextBuilder.formatCategoryHierarchy("Furniture / Beds")).isEqualTo("Furniture > Beds");
        assertThat(EmbeddingTextBuilder.formatCategoryHierarchy("Furniture/Beds")).isEqualTo("Furniture > Beds");
    }

    @Test
    void categoryHierarchyKeepsWhitespaceAwayFromSlashes() {
        assertThat(EmbeddingTextBuilder.formatCategoryHierarchy(" Home Decor /Wall Art ")).isEqualTo(" Home Decor > Wall Art ");
    }

    @Test
    void categoryHierarchyKeepsEmptyLevels() {
        assertThat(EmbeddingTextBuilder.formatCategoryHierarchy("a/ /b/")).isEqualTo("a >  > b > ");
    }

    @Test
    void buildJoinsFieldsWithFormattedHierarchy() {
        assertThat(EmbeddingTextBuilder.build("Oak Bed", "Beds", "Furniture / Beds", null))
                .isEqualTo("Oak Bed. Beds. Furniture > Beds");
    }
}
