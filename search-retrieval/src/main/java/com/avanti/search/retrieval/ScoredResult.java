package com.avanti.search.retrieval;

/** A ranked product id from a SearchStrategy, in descending score order. */
public record ScoredResult(String productId, double score) {
}
