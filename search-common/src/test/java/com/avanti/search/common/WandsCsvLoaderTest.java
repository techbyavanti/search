package com.avanti.search.common;

import org.junit.jupiter.api.Test;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class WandsCsvLoaderTest {

    @Test
    void loadsProductsQueriesAndLabelsFromSampleFixture() throws URISyntaxException {
        WandsDataset dataset = WandsCsvLoader.load(sampleDir());

        assertThat(dataset.products()).hasSize(3);
        assertThat(dataset.queries()).hasSize(2);
        assertThat(dataset.labels()).hasSize(3);
    }

    @Test
    void parsesProductFieldsIncludingBlankOnes() throws URISyntaxException {
        WandsDataset dataset = WandsCsvLoader.load(sampleDir());

        WandsProduct bed = dataset.products().stream()
                .filter(p -> p.productId().equals("0"))
                .findFirst()
                .orElseThrow();
        assertThat(bed.productName()).isEqualTo("solid wood platform bed");
        assertThat(bed.categoryHierarchy()).isEqualTo("Furniture / Bedroom Furniture / Beds & Headboards / Beds / Twin Beds");
        assertThat(bed.ratingCount()).isEqualTo(15);
        assertThat(bed.averageRating()).isEqualTo(4.5);

        WandsProduct table = dataset.products().stream()
                .filter(p -> p.productId().equals("1"))
                .findFirst()
                .orElseThrow();
        assertThat(table.productDescription()).isNull();
        assertThat(table.ratingCount()).isNull();
        assertThat(table.averageRating()).isNull();
    }

    @Test
    void indexesJudgmentsByQueryAndProduct() throws URISyntaxException {
        WandsDataset dataset = WandsCsvLoader.load(sampleDir());

        Map<String, RelevanceGrade> judgmentsForQuery0 = dataset.judgmentsFor("0");
        assertThat(judgmentsForQuery0).containsEntry("0", RelevanceGrade.EXACT);
        assertThat(judgmentsForQuery0).containsEntry("1", RelevanceGrade.IRRELEVANT);

        assertThat(dataset.judgmentsFor("1")).containsEntry("1", RelevanceGrade.PARTIAL);
        assertThat(dataset.judgmentsFor("nonexistent-query")).isEmpty();
    }

    @Test
    void relevanceGradeOrdersByGradeAndFlagsRelevance() {
        assertThat(RelevanceGrade.EXACT.grade()).isEqualTo(2);
        assertThat(RelevanceGrade.PARTIAL.grade()).isEqualTo(1);
        assertThat(RelevanceGrade.IRRELEVANT.grade()).isEqualTo(0);

        assertThat(RelevanceGrade.EXACT.isRelevant()).isTrue();
        assertThat(RelevanceGrade.PARTIAL.isRelevant()).isTrue();
        assertThat(RelevanceGrade.IRRELEVANT.isRelevant()).isFalse();
    }

    @Test
    void buildsEmbeddingTextFromNameClassCategoryAndDescription() throws URISyntaxException {
        WandsDataset dataset = WandsCsvLoader.load(sampleDir());
        WandsProduct bed = dataset.products().stream()
                .filter(p -> p.productId().equals("0"))
                .findFirst()
                .orElseThrow();

        String text = EmbeddingTextBuilder.build(bed);

        assertThat(text).startsWith("solid wood platform bed. Beds. Furniture > Bedroom Furniture > Beds & Headboards > Beds > Twin Beds. A comfortable bed, built to last.");
        assertThat(text).doesNotContain("material:wood");
    }

    @Test
    void embeddingTextOmitsBlankFieldsGracefully() throws URISyntaxException {
        WandsDataset dataset = WandsCsvLoader.load(sampleDir());
        WandsProduct table = dataset.products().stream()
                .filter(p -> p.productId().equals("1"))
                .findFirst()
                .orElseThrow();

        String text = EmbeddingTextBuilder.build(table);

        assertThat(text).isEqualTo("modern glass coffee table. Coffee Tables. Furniture > Living Room Furniture > Coffee Tables");
    }

    private Path sampleDir() throws URISyntaxException {
        return Paths.get(getClass().getClassLoader().getResource("wands-sample").toURI());
    }
}
