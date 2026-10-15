package com.avanti.search.inference;

/**
 * Turns a search query into an L2-normalized embedding. Implemented by
 * {@link EmbeddingService}; strategies depend on this interface so a
 * caching decorator can sit in front of the model.
 */
@FunctionalInterface
public interface QueryEncoder {

    float[] embedQuery(String query);
}
