package com.avanti.search.retrieval;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;

class RerankScoreCacheTest {

    /** Fake scorer: scores each id by its length, and records every batch it was asked to score. */
    private static Function<List<String>, Map<String, Float>> recordingScorer(List<List<String>> calls) {
        return ids -> {
            calls.add(List.copyOf(ids));
            Map<String, Float> scores = new HashMap<>();
            for (String id : ids) {
                scores.put(id, (float) id.length());
            }
            return scores;
        };
    }

    @Test
    void scoresMissingIdsAndServesRepeatsFromCache() {
        List<List<String>> calls = new ArrayList<>();
        RerankScoreCache cache = new RerankScoreCache(100);

        Map<String, Float> first = cache.getOrScore("oak table", List.of("a", "bb"), recordingScorer(calls));
        Map<String, Float> second = cache.getOrScore("oak table", List.of("a", "bb"), recordingScorer(calls));

        assertThat(first).containsExactlyInAnyOrderEntriesOf(Map.of("a", 1f, "bb", 2f));
        assertThat(second).isEqualTo(first);
        assertThat(calls).containsExactly(List.of("a", "bb"));
    }

    @Test
    void onlyUncachedIdsReachTheScorer() {
        List<List<String>> calls = new ArrayList<>();
        RerankScoreCache cache = new RerankScoreCache(100);

        cache.getOrScore("oak table", List.of("a", "bb"), recordingScorer(calls));
        Map<String, Float> scores = cache.getOrScore("oak table", List.of("bb", "ccc"), recordingScorer(calls));

        assertThat(scores).containsOnlyKeys("bb", "ccc");
        assertThat(calls).containsExactly(List.of("a", "bb"), List.of("ccc"));
    }

    @Test
    void sameIdUnderADifferentQueryIsScoredAgain() {
        List<List<String>> calls = new ArrayList<>();
        RerankScoreCache cache = new RerankScoreCache(100);

        cache.getOrScore("oak table", List.of("a"), recordingScorer(calls));
        cache.getOrScore("pine table", List.of("a"), recordingScorer(calls));

        assertThat(calls).hasSize(2);
    }

    @Test
    void queriesDifferingOnlyInCaseAndWhitespaceShareEntries() {
        List<List<String>> calls = new ArrayList<>();
        RerankScoreCache cache = new RerankScoreCache(100);

        cache.getOrScore("Oak  Table", List.of("a"), recordingScorer(calls));
        cache.getOrScore("  oak table ", List.of("a"), recordingScorer(calls));

        assertThat(calls).hasSize(1);
    }

    @Test
    void evictsLeastRecentlyUsedEntriesBeyondMaxSize() {
        List<List<String>> calls = new ArrayList<>();
        RerankScoreCache cache = new RerankScoreCache(2);

        cache.getOrScore("q", List.of("a", "b"), recordingScorer(calls));
        cache.getOrScore("q", List.of("c"), recordingScorer(calls));
        assertThat(cache.size()).isEqualTo(2);

        cache.getOrScore("q", List.of("a"), recordingScorer(calls));
        assertThat(calls.getLast()).containsExactly("a");
    }

    @Test
    void disabledCacheAlwaysCallsTheScorerAndStoresNothing() {
        List<List<String>> calls = new ArrayList<>();
        RerankScoreCache cache = RerankScoreCache.disabled();

        cache.getOrScore("q", List.of("a"), recordingScorer(calls));
        cache.getOrScore("q", List.of("a"), recordingScorer(calls));

        assertThat(calls).hasSize(2);
        assertThat(cache.size()).isZero();
    }
}
