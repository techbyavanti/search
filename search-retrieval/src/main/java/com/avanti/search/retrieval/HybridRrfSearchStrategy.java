package com.avanti.search.retrieval;

import java.util.List;

/**
 * Fuses the lexical and semantic strategies' ranked lists with Reciprocal
 * Rank Fusion. Each sub-strategy is queried for candidatePoolSize results
 * regardless of the topK requested here, so fusion always has a full pool to
 * work with (and so the pool matches what a downstream reranker would see).
 */
public final class HybridRrfSearchStrategy implements SearchStrategy {

    private final SearchStrategy lexical;
    private final SearchStrategy semantic;
    private final int candidatePoolSize;
    private final int rrfK;

    public HybridRrfSearchStrategy(SearchStrategy lexical, SearchStrategy semantic, int candidatePoolSize, int rrfK) {
        this.lexical = lexical;
        this.semantic = semantic;
        this.candidatePoolSize = candidatePoolSize;
        this.rrfK = rrfK;
    }

    @Override
    public List<ScoredResult> search(String query, int topK) {
        List<ScoredResult> lexicalResults = lexical.search(query, candidatePoolSize);
        List<ScoredResult> semanticResults = semantic.search(query, candidatePoolSize);
        List<ScoredResult> fused = RrfFusionService.fuse(List.of(lexicalResults, semanticResults), rrfK);
        return fused.size() > topK ? fused.subList(0, topK) : fused;
    }

    @Override
    public String name() {
        return "Hybrid (RRF, k=" + rrfK + ")";
    }
}
