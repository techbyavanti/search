package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.inference.NeuralRerankerService;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Rescores the hybrid strategy's candidate pool with a small neural net
 * over cheap, mostly-cached features (see RerankFeatureBuilder) instead of
 * a transformer forward pass — the same quality-improving idea as
 * HybridRerankStrategy, aimed at landing much closer to Hybrid's latency.
 * Candidates missing from the feature cache/Elasticsearch (shouldn't
 * happen against a consistent index, but shouldn't break the batch either)
 * score -infinity so they sort last.
 */
public final class NeuralRerankStrategy implements SearchStrategy {

    private final SearchStrategy hybrid;
    private final ElasticsearchClient client;
    private final EmbeddingService embeddingService;
    private final NeuralRerankerService neuralRerankerService;
    private final ProductFeatureCache featureCache;
    private final int candidatePoolSize;

    public NeuralRerankStrategy(SearchStrategy hybrid, ElasticsearchClient client, EmbeddingService embeddingService,
                                 NeuralRerankerService neuralRerankerService, ProductFeatureCache featureCache, int candidatePoolSize) {
        this.hybrid = hybrid;
        this.client = client;
        this.embeddingService = embeddingService;
        this.neuralRerankerService = neuralRerankerService;
        this.featureCache = featureCache;
        this.candidatePoolSize = candidatePoolSize;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        List<ScoredResult> candidates = hybrid.search(query, candidatePoolSize);
        if (candidates.isEmpty()) {
            return candidates;
        }

        List<String> productIds = candidates.stream().map(ScoredResult::productId).toList();
        Map<String, CachedProductFeatures> cachedFeatures = featureCache.getBatch(productIds);

        float[] queryEmbedding = embeddingService.embedQuery(query);
        List<String> queryTokens = RerankFeatureBuilder.tokenize(query);

        List<float[]> vectors = new ArrayList<>(candidates.size());
        List<Boolean> hasFeatures = new ArrayList<>(candidates.size());
        for (ScoredResult candidate : candidates) {
            CachedProductFeatures features = cachedFeatures.get(candidate.productId());
            if (features == null) {
                vectors.add(new float[RerankFeatureBuilder.FEATURE_COUNT]);
                hasFeatures.add(false);
            } else {
                vectors.add(RerankFeatureBuilder.build(candidate.score(), queryEmbedding, queryTokens, features));
                hasFeatures.add(true);
            }
        }
        List<Float> scores = neuralRerankerService.scoreBatch(vectors);

        List<ScoredResult> rescored = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            float score = hasFeatures.get(i) ? scores.get(i) : Float.NEGATIVE_INFINITY;
            rescored.add(new ScoredResult(candidates.get(i).productId(), score));
        }
        rescored.sort(Comparator.comparingDouble(ScoredResult::score).reversed());

        return rescored.size() > topK ? rescored.subList(0, topK) : rescored;
    }

    @Override
    public String name() {
        return "Neural Rerank (MLP)";
    }
}
