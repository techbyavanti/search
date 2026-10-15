package com.avanti.search.retrieval;

import com.avanti.search.inference.RerankerService;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Rescores the hybrid strategy's candidate pool with the cross-encoder
 * reranker. Candidates without fetchable text (shouldn't happen against a
 * consistent index, but a strategy shouldn't throw over it) are scored as
 * -infinity so they sort last rather than breaking the batch. Product text
 * comes from the shared ProductFeatureCache (also used by
 * NeuralRerankStrategy) rather than a guaranteed-fresh Elasticsearch fetch
 * every request, so a product either strategy has already seen doesn't
 * need a second round trip. Scores come from a RerankScoreCache, so a
 * repeated query only runs the cross-encoder for candidates it hasn't
 * scored before for that query.
 */
public final class HybridRerankStrategy implements SearchStrategy {

    private final SearchStrategy hybrid;
    private final ProductFeatureCache featureCache;
    private final RerankerService rerankerService;
    private final RerankScoreCache scoreCache;
    private final int candidatePoolSize;

    public HybridRerankStrategy(SearchStrategy hybrid, ProductFeatureCache featureCache, RerankerService rerankerService,
                                RerankScoreCache scoreCache, int candidatePoolSize) {
        this.hybrid = hybrid;
        this.featureCache = featureCache;
        this.rerankerService = rerankerService;
        this.scoreCache = scoreCache;
        this.candidatePoolSize = candidatePoolSize;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        List<ScoredResult> candidates = hybrid.search(query, candidatePoolSize);
        if (candidates.isEmpty()) {
            return candidates;
        }

        List<String> productIds = candidates.stream().map(ScoredResult::productId).toList();
        Map<String, String> texts = ProductTextFetcher.fetchTexts(featureCache, productIds);

        List<String> scorableIds = productIds.stream().filter(texts::containsKey).toList();
        Map<String, Float> scores = scoreCache.getOrScore(query, scorableIds, ids -> {
            if (ids.isEmpty()) {
                return Map.of();
            }
            List<String> documents = ids.stream().map(texts::get).toList();
            List<Float> batchScores = rerankerService.scoreBatch(query, documents);
            Map<String, Float> byId = new HashMap<>();
            for (int i = 0; i < ids.size(); i++) {
                byId.put(ids.get(i), batchScores.get(i));
            }
            return byId;
        });

        List<ScoredResult> rescored = new ArrayList<>(candidates.size());
        for (ScoredResult candidate : candidates) {
            String productId = candidate.productId();
            float score = scores.getOrDefault(productId, Float.NEGATIVE_INFINITY);
            rescored.add(new ScoredResult(productId, score));
        }
        rescored.sort(Comparator.comparingDouble(ScoredResult::score).reversed());

        return rescored.size() > topK ? rescored.subList(0, topK) : rescored;
    }

    @Override
    public String name() {
        return "Hybrid + Cross-Encoder Rerank";
    }
}
