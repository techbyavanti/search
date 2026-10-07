package com.avanti.search.eval;

import java.util.List;

public record StrategySummary(
        String strategyName,
        double ndcgAt10,
        double mrr,
        double recallAt10,
        double recallAt50,
        double precisionAt10,
        long p95LatencyMs
) {
    public static StrategySummary aggregate(String strategyName, List<QueryMetrics> perQuery, List<Long> latenciesMs) {
        double ndcg = perQuery.stream().mapToDouble(QueryMetrics::ndcgAt10).average().orElse(0.0);
        double mrr = perQuery.stream().mapToDouble(QueryMetrics::mrr).average().orElse(0.0);
        double recall10 = perQuery.stream().mapToDouble(QueryMetrics::recallAt10).average().orElse(0.0);
        double recall50 = perQuery.stream().mapToDouble(QueryMetrics::recallAt50).average().orElse(0.0);
        double precision10 = perQuery.stream().mapToDouble(QueryMetrics::precisionAt10).average().orElse(0.0);

        List<Long> sorted = latenciesMs.stream().sorted().toList();
        long p95 = sorted.isEmpty() ? 0 : sorted.get((int) Math.min(sorted.size() - 1, Math.ceil(sorted.size() * 0.95) - 1));

        return new StrategySummary(strategyName, ndcg, mrr, recall10, recall50, precision10, p95);
    }
}
