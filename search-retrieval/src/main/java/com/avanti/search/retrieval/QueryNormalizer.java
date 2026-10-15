package com.avanti.search.retrieval;

import java.util.Locale;

/**
 * Cache-key normalization for query text: trim, collapse whitespace,
 * lowercase. It doesn't change model output, because both models' tokenizers
 * are uncased (BertNormalizer, lowercase) and split on whitespace.
 */
final class QueryNormalizer {

    private QueryNormalizer() {
    }

    static String normalize(String query) {
        return query.strip().replaceAll("\\s+", " ").toLowerCase(Locale.ROOT);
    }
}
