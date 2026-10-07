package com.avanti.search.retrieval;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class HybridRrfSearchStrategyTest {

    @Test
    void queriesBothSubStrategiesWithTheCandidatePoolSizeNotTopK() {
        FixedSearchStrategy lexical = new FixedSearchStrategy(
                "Lexical", List.of(new ScoredResult("a", 10.0), new ScoredResult("b", 8.0)));
        FixedSearchStrategy semantic = new FixedSearchStrategy(
                "Semantic", List.of(new ScoredResult("b", 0.9), new ScoredResult("a", 0.7)));

        HybridRrfSearchStrategy hybrid = new HybridRrfSearchStrategy(lexical, semantic, 50, 60);
        hybrid.search("query", 10);

        assertThat(lexical.lastRequestedTopK).isEqualTo(50);
        assertThat(semantic.lastRequestedTopK).isEqualTo(50);
    }

    @Test
    void truncatesFusedResultsToRequestedTopK() {
        FixedSearchStrategy lexical = new FixedSearchStrategy("Lexical", List.of(
                new ScoredResult("a", 10.0), new ScoredResult("b", 9.0), new ScoredResult("c", 8.0)));
        FixedSearchStrategy semantic = new FixedSearchStrategy("Semantic", List.of());

        HybridRrfSearchStrategy hybrid = new HybridRrfSearchStrategy(lexical, semantic, 50, 60);
        List<ScoredResult> results = hybrid.search("query", 2);

        assertThat(results).hasSize(2);
        assertThat(results).extracting(ScoredResult::productId).containsExactly("a", "b");
    }

    @Test
    void nameIncludesTheConfiguredRrfConstant() {
        HybridRrfSearchStrategy hybrid = new HybridRrfSearchStrategy(
                new FixedSearchStrategy("Lexical", List.of()),
                new FixedSearchStrategy("Semantic", List.of()),
                50, 42);

        assertThat(hybrid.name()).isEqualTo("Hybrid (RRF, k=42)");
    }

    private static final class FixedSearchStrategy implements SearchStrategy {
        private final String name;
        private final List<ScoredResult> results;
        private int lastRequestedTopK = -1;

        private FixedSearchStrategy(String name, List<ScoredResult> results) {
            this.name = name;
            this.results = results;
        }

        @Override
        public List<ScoredResult> search(String query, int topK) {
            this.lastRequestedTopK = topK;
            return results;
        }

        @Override
        public String name() {
            return name;
        }
    }
}
