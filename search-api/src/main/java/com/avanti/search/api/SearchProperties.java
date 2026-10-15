package com.avanti.search.api;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.file.Path;

@ConfigurationProperties(prefix = "search")
public record SearchProperties(
        Elasticsearch elasticsearch,
        Models models,
        Hybrid hybrid,
        FeatureCache featureCache,
        QueryEmbeddingCache queryEmbeddingCache,
        SemanticHnsw semanticHnsw,
        Reranker reranker
) {
    public record Elasticsearch(String host) {
    }

    public record Models(Path embeddingDir, Path rerankerDir, Path neuralRerankerDir, Path learnedTowerDir) {
    }

    /** Bounds ProductFeatureCache's size — default comfortably covers the whole ~43K-product catalog. */
    public record FeatureCache(int maxSize) {
    }

    /** num_candidates for the tuned float-HNSW strategy; see results/hnsw-sweep.csv. */
    public record SemanticHnsw(int numCandidates) {
    }

    /** Bounds CachingQueryEncoder in queries (~1.5 KB per 384-dim vector). */
    public record QueryEmbeddingCache(int maxSize) {
    }

    /**
     * chunkSize: max (query, product) pairs per cross-encoder forward pass —
     * a 50-candidate pool becomes 10 parallel passes. scoreCacheMaxSize bounds
     * RerankScoreCache in (query, product) entries; 0 disables it.
     */
    public record Reranker(int chunkSize, int scoreCacheMaxSize) {
    }

    /**
     * rrfK=40 is the constant search-eval's offline sweep found best on the
     * WANDS eval set (see RESULTS.md's RRF sweep table — the curve is flat
     * across the swept range, so this isn't a fragile choice). The neural
     * reranker is trained on pools fused with this same k. A real deployment
     * would periodically re-sweep against fresh judgments rather than
     * trust a value picked once.
     */
    public record Hybrid(int rrfK, int candidatePoolSize) {
    }
}
