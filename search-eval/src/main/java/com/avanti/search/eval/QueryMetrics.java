package com.avanti.search.eval;

public record QueryMetrics(
        String queryId,
        double ndcgAt10,
        double mrr,
        double recallAt10,
        double recallAt50,
        double precisionAt10
) {
}
