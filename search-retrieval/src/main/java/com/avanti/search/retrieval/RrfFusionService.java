package com.avanti.search.retrieval;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reciprocal Rank Fusion: combines multiple ranked lists into one by summing,
 * per document, 1/(k + rank) over every list it appears in (1-based rank; a
 * document absent from a list simply contributes nothing for that list).
 * Hand-rolled rather than using an engine's built-in fusion endpoint so the
 * eval harness can sweep k directly against offline metrics.
 */
public final class RrfFusionService {

    private RrfFusionService() {
    }

    public static List<ScoredResult> fuse(List<List<ScoredResult>> rankedLists, int k) {
        Map<String, Double> fusedScores = new LinkedHashMap<>();
        for (List<ScoredResult> rankedList : rankedLists) {
            for (int i = 0; i < rankedList.size(); i++) {
                String productId = rankedList.get(i).productId();
                double contribution = 1.0 / (k + i + 1);
                fusedScores.merge(productId, contribution, Double::sum);
            }
        }

        return fusedScores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .map(e -> new ScoredResult(e.getKey(), e.getValue()))
                .toList();
    }
}
