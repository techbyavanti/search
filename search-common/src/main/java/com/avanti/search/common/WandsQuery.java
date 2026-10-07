package com.avanti.search.common;

/** A row from WANDS query.csv. */
public record WandsQuery(String queryId, String queryText, String queryClass) {
}
