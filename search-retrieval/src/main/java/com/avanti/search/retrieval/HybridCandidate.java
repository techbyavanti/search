package com.avanti.search.retrieval;

/**
 * One fused hybrid candidate plus the per-retriever signals RRF reduces it
 * to. {@link RerankFeatureBuilder} turns these into the neural reranker's
 * input, so the model sees which retriever found the candidate and how
 * strongly, not only the fused score. Ranks are 1-based; a rank of 0 means
 * that retriever did not return the candidate (its score is then 0).
 */
public record HybridCandidate(
        String productId,
        int hybridRank,
        double rrfScore,
        int lexicalRank,
        double lexicalScore,
        int semanticRank,
        double semanticScore
) {
    public ScoredResult toScoredResult() {
        return new ScoredResult(productId, rrfScore);
    }
}
