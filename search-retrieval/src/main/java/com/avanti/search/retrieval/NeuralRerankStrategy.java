package com.avanti.search.retrieval;

import com.avanti.search.inference.QueryEncoder;
import com.avanti.search.inference.NeuralRerankerService;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Rescores the hybrid strategy's candidate pool with a small neural net
 * over cheap, mostly-cached features (see RerankFeatureBuilder) instead of
 * a transformer forward pass — the same quality-improving idea as
 * HybridRerankStrategy, at close to Hybrid's latency. The model is trained
 * with a LambdaRank objective on exactly these pools (see
 * training/train_neural_reranker.py). Candidates missing from the feature
 * cache/Elasticsearch (shouldn't happen against a consistent index, but
 * shouldn't break the batch either) score -infinity so they sort last.
 */
public final class NeuralRerankStrategy implements SearchStrategy {

    private final HybridRrfSearchStrategy hybrid;
    private final QueryEncoder queryEncoder;
    private final NeuralRerankerService neuralRerankerService;
    private final ProductFeatureCache featureCache;
    private final int candidatePoolSize;

    public NeuralRerankStrategy(HybridRrfSearchStrategy hybrid, QueryEncoder queryEncoder,
                                NeuralRerankerService neuralRerankerService, ProductFeatureCache featureCache,
                                int candidatePoolSize) {
        if (neuralRerankerService.featureCount() != RerankFeatureBuilder.FEATURE_COUNT) {
            throw new IllegalStateException("Neural reranker model expects " + neuralRerankerService.featureCount()
                    + " features but RerankFeatureBuilder builds " + RerankFeatureBuilder.FEATURE_COUNT
                    + "; retrain it with scripts/train-neural-reranker.sh");
        }
        this.hybrid = hybrid;
        this.queryEncoder = queryEncoder;
        this.neuralRerankerService = neuralRerankerService;
        this.featureCache = featureCache;
        this.candidatePoolSize = candidatePoolSize;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        List<HybridCandidate> candidates = hybrid.searchWithSignals(query, candidatePoolSize);
        if (candidates.isEmpty()) {
            return List.of();
        }

        List<float[]> vectors = buildFeatures(featureCache, queryEncoder, query, candidates);
        List<float[]> scorable = vectors.stream()
                .map(v -> v != null ? v : new float[RerankFeatureBuilder.FEATURE_COUNT])
                .toList();
        List<Float> scores = neuralRerankerService.scoreBatch(scorable);

        List<ScoredResult> rescored = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            float score = vectors.get(i) != null ? scores.get(i) : Float.NEGATIVE_INFINITY;
            rescored.add(new ScoredResult(candidates.get(i).productId(), score));
        }
        rescored.sort(Comparator.comparingDouble(ScoredResult::score).reversed());

        return rescored.size() > topK ? rescored.subList(0, topK) : rescored;
    }

    /**
     * The model's input for each candidate, in candidate order; null where a
     * product has no cached data. Shared with RerankFeatureExportCli, which
     * writes the training data.
     */
    public static List<float[]> buildFeatures(ProductFeatureCache featureCache, QueryEncoder queryEncoder,
                                              String query, List<HybridCandidate> candidates) {
        List<String> productIds = candidates.stream().map(HybridCandidate::productId).toList();
        Map<String, CachedProductFeatures> products = featureCache.getBatch(productIds);
        float[] queryEmbedding = queryEncoder.embedQuery(query);
        return RerankFeatureBuilder.buildPool(candidates, query, queryEmbedding, products);
    }

    @Override
    public String name() {
        return "Neural Rerank (MLP)";
    }
}
