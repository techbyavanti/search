package com.avanti.search.retrieval;

import java.util.List;

/**
 * A pluggable ranking strategy over the products index. Implementations are
 * shared, unmodified, between search-api (serving) and search-eval (offline
 * evaluation) so eval always measures exactly what's deployed.
 */
public interface SearchStrategy {

    /** Ranked results, highest score first, at most topK entries. */
    List<ScoredResult> search(String query, int topK);

    /** Short identifier used as the RESULTS.md row label. */
    String name();
}
