package com.avanti.search.retrieval;

import com.avanti.search.inference.RerankerService;

import java.util.ArrayList;
import java.util.Comparator;
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
 * need a second round trip.
 */
public final class HybridRerankStrategy implements SearchStrategy {

    private final SearchStrategy hybrid;
    private final ProductFeatureCache featureCache;
    private final RerankerService rerankerService;
    private final int candidatePoolSize;

    public HybridRerankStrategy(SearchStrategy hybrid, ProductFeatureCache featureCache, RerankerService rerankerService, int candidatePoolSize) {
        this.hybrid = hybrid;
        this.featureCache = featureCache;
        this.rerankerService = rerankerService;
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

        List<String> documents = new ArrayList<>(candidates.size());
        for (ScoredResult candidate : candidates) {
            documents.add(texts.getOrDefault(candidate.productId(), ""));
        }
        List<Float> scores = rerankerService.scoreBatch(query, documents);

        List<ScoredResult> rescored = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            String productId = candidates.get(i).productId();
            float score = texts.containsKey(productId) ? scores.get(i) : Float.NEGATIVE_INFINITY;
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
