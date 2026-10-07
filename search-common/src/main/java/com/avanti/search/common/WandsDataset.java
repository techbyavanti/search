package com.avanti.search.common;

import java.util.List;
import java.util.Map;

/**
 * The full WANDS dataset, with labels pre-indexed by query for O(1) lookup
 * during evaluation.
 */
public record WandsDataset(
        List<WandsProduct> products,
        List<WandsQuery> queries,
        List<WandsLabel> labels,
        Map<String, Map<String, RelevanceGrade>> judgmentsByQuery
) {
    public Map<String, RelevanceGrade> judgmentsFor(String queryId) {
        return judgmentsByQuery.getOrDefault(queryId, Map.of());
    }
}
