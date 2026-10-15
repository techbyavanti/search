package com.avanti.search.retrieval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
        return searchWithSignals(query, topK).stream().map(HybridCandidate::toScoredResult).toList();
    }

    /** The same fused ranking as {@link #search}, with each candidate's lexical and semantic rank and score kept. */
    public List<HybridCandidate> searchWithSignals(String query, int topK) {
        List<ScoredResult> lexicalResults = lexical.search(query, candidatePoolSize);
        List<ScoredResult> semanticResults = semantic.search(query, candidatePoolSize);
        List<ScoredResult> fused = RrfFusionService.fuse(List.of(lexicalResults, semanticResults), rrfK);

        Map<String, Integer> lexicalRanks = ranksById(lexicalResults);
        Map<String, Integer> semanticRanks = ranksById(semanticResults);

        int size = Math.min(topK, fused.size());
        List<HybridCandidate> candidates = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            ScoredResult result = fused.get(i);
            int lexicalRank = lexicalRanks.getOrDefault(result.productId(), 0);
            int semanticRank = semanticRanks.getOrDefault(result.productId(), 0);
            candidates.add(new HybridCandidate(
                    result.productId(), i + 1, result.score(),
                    lexicalRank, lexicalRank > 0 ? lexicalResults.get(lexicalRank - 1).score() : 0.0,
                    semanticRank, semanticRank > 0 ? semanticResults.get(semanticRank - 1).score() : 0.0));
        }
        return candidates;
    }

    private static Map<String, Integer> ranksById(List<ScoredResult> results) {
        Map<String, Integer> ranks = new HashMap<>();
        for (int i = 0; i < results.size(); i++) {
            ranks.putIfAbsent(results.get(i).productId(), i + 1);
        }
        return ranks;
    }

    @Override
    public String name() {
        return "Hybrid (RRF, k=" + rrfK + ")";
    }
}
