package com.avanti.search.common;

/** A row from WANDS label.csv — one query/product relevance judgment. */
public record WandsLabel(String queryId, String productId, RelevanceGrade grade) {
}
