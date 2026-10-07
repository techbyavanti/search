package com.avanti.search.eval;

import com.avanti.search.common.RelevanceGrade;

import java.util.List;
import java.util.Map;

/**
 * Standard IR metrics computed against WANDS graded relevance judgments.
 * Queries with no judged-relevant products contribute 0 to every metric
 * (consistent with trec_eval's default per-topic treatment) rather than
 * being excluded, so the aggregate mean isn't inflated by dropping hard cases.
 */
public final class MetricsCalculator {

    private MetricsCalculator() {
    }

    public static double ndcgAtK(List<String> rankedProductIds, Map<String, RelevanceGrade> judgments, int k) {
        double dcg = dcgAtK(rankedProductIds, judgments, k);

        List<String> idealOrder = judgments.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue().grade(), a.getValue().grade()))
                .map(Map.Entry::getKey)
                .toList();
        double idcg = dcgAtK(idealOrder, judgments, k);

        return idcg == 0.0 ? 0.0 : dcg / idcg;
    }

    private static double dcgAtK(List<String> rankedProductIds, Map<String, RelevanceGrade> judgments, int k) {
        double dcg = 0.0;
        int limit = Math.min(k, rankedProductIds.size());
        for (int i = 0; i < limit; i++) {
            RelevanceGrade grade = judgments.get(rankedProductIds.get(i));
            int relevance = grade == null ? 0 : grade.grade();
            dcg += (Math.pow(2, relevance) - 1) / (Math.log(i + 2) / Math.log(2));
        }
        return dcg;
    }

    public static double reciprocalRank(List<String> rankedProductIds, Map<String, RelevanceGrade> judgments) {
        for (int i = 0; i < rankedProductIds.size(); i++) {
            RelevanceGrade grade = judgments.get(rankedProductIds.get(i));
            if (grade != null && grade.isRelevant()) {
                return 1.0 / (i + 1);
            }
        }
        return 0.0;
    }

    public static double recallAtK(List<String> rankedProductIds, Map<String, RelevanceGrade> judgments, int k) {
        long totalRelevant = judgments.values().stream().filter(RelevanceGrade::isRelevant).count();
        if (totalRelevant == 0) {
            return 0.0;
        }
        long retrievedRelevant = rankedProductIds.stream()
                .limit(k)
                .filter(id -> isRelevant(judgments, id))
                .count();
        return (double) retrievedRelevant / totalRelevant;
    }

    public static double precisionAtK(List<String> rankedProductIds, Map<String, RelevanceGrade> judgments, int k) {
        int limit = Math.min(k, rankedProductIds.size());
        if (limit == 0) {
            return 0.0;
        }
        long retrievedRelevant = rankedProductIds.stream()
                .limit(limit)
                .filter(id -> isRelevant(judgments, id))
                .count();
        return (double) retrievedRelevant / limit;
    }

    private static boolean isRelevant(Map<String, RelevanceGrade> judgments, String productId) {
        RelevanceGrade grade = judgments.get(productId);
        return grade != null && grade.isRelevant();
    }
}
