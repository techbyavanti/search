package com.avanti.search.eval;

import com.avanti.search.retrieval.ProductDocument;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class HnswTuningCliTest {

    private static ProductDocument vector(String id, float x, float y) {
        return new ProductDocument(id, null, null, null, null, null, null, null, null, new float[]{x, y}, null, null);
    }

    @Test
    void exactTopKRanksByDotProductHighestFirst() {
        List<ProductDocument> products = List.of(vector("far", 0f, 1f), vector("near", 1f, 0f), vector("mid", 0.7f, 0.7f));

        assertThat(HnswTuningCli.exactTopK(new float[]{1f, 0f}, products)).containsExactly("near", "mid", "far");
    }

    @Test
    void overlapIsTheFractionOfExactTopKFound() {
        List<String> exact = List.of("a", "b", "c", "d");

        assertThat(HnswTuningCli.overlap(List.of("a", "x", "c", "y"), exact, 4)).isEqualTo(0.5);
        assertThat(HnswTuningCli.overlap(List.of("b", "a"), exact, 2)).isEqualTo(1.0);
    }
}
