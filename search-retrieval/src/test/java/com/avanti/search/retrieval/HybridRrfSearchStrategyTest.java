package com.avanti.search.retrieval;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class HybridRrfSearchStrategyTest {

    private static SearchStrategy fixed(List<ScoredResult> results) {
        return new SearchStrategy() {
            @Override
            public List<ScoredResult> search(String query, int topK) {
                return results;
            }

            @Override
            public String name() {
                return "fixed";
            }
        };
    }

    private final HybridRrfSearchStrategy hybrid = new HybridRrfSearchStrategy(
            fixed(List.of(new ScoredResult("a", 12.0), new ScoredResult("b", 9.0))),
            fixed(List.of(new ScoredResult("b", 0.95), new ScoredResult("c", 0.90))),
            50, 60);

    @Test
    void signalsKeepEachRetrieversRankAndScore() {
        List<HybridCandidate> candidates = hybrid.searchWithSignals("q", 10);

        assertThat(candidates).extracting(HybridCandidate::productId).containsExactly("b", "a", "c");
        HybridCandidate b = candidates.get(0);
        assertThat(b.hybridRank()).isEqualTo(1);
        assertThat(b.rrfScore()).isEqualTo(1.0 / 62 + 1.0 / 61);
        assertThat(b.lexicalRank()).isEqualTo(2);
        assertThat(b.lexicalScore()).isEqualTo(9.0);
        assertThat(b.semanticRank()).isEqualTo(1);
        assertThat(b.semanticScore()).isEqualTo(0.95);

        HybridCandidate c = candidates.get(2);
        assertThat(c.lexicalRank()).isZero();
        assertThat(c.lexicalScore()).isZero();
    }

    @Test
    void searchReturnsTheSameRankingAsTheSignals() {
        assertThat(hybrid.search("q", 2)).containsExactly(
                new ScoredResult("b", 1.0 / 62 + 1.0 / 61),
                new ScoredResult("a", 1.0 / 61));
    }
}
