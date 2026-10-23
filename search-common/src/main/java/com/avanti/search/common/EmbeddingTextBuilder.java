package com.avanti.search.common;

import java.util.StringJoiner;

/**
 * Builds the text passed to the embedding model for a product. Deliberately
 * excludes {@code productFeatures} (pipe-delimited attribute:value pairs) —
 * that attribute noise dilutes sentence-embedding quality more than it helps;
 * it is still indexed as a separate lexical field for BM25.
 */
public final class EmbeddingTextBuilder {

    private static final int MAX_DESCRIPTION_CHARS = 1200;

    private EmbeddingTextBuilder() {
    }

    public static String build(WandsProduct product) {
        return build(product.productName(), product.productClass(), product.categoryHierarchy(), product.productDescription());
    }

    /**
     * Source category hierarchies are inconsistently spaced around "/" ("Furniture / Beds" vs
     * "Furniture/Beds"), so whitespace next to each "/" is dropped and the levels are joined with
     * " > ". One linear pass, unlike a whitespace regex, which is quadratic on long blank runs.
     */
    static String formatCategoryHierarchy(String categoryHierarchy) {
        String[] levels = categoryHierarchy.split("/", -1);
        int last = levels.length - 1;
        StringJoiner joined = new StringJoiner(" > ");
        for (int i = 0; i <= last; i++) {
            String level = levels[i];
            if (i > 0) {
                level = level.stripLeading();
            }
            if (i < last) {
                level = level.stripTrailing();
            }
            joined.add(level);
        }
        return joined.toString();
    }

    /** Same field composition as {@link #build(WandsProduct)}, for callers reading fields back from Elasticsearch rather than a WandsProduct (e.g. reranking). */
    public static String build(String productName, String productClass, String categoryHierarchy, String productDescription) {
        StringBuilder text = new StringBuilder(productName);
        if (productClass != null) {
            text.append(". ").append(productClass);
        }
        if (categoryHierarchy != null) {
            text.append(". ").append(formatCategoryHierarchy(categoryHierarchy));
        }
        if (productDescription != null) {
            text.append(". ").append(truncate(productDescription, MAX_DESCRIPTION_CHARS));
        }
        return text.toString();
    }

    private static String truncate(String text, int maxChars) {
        return text.length() <= maxChars ? text : text.substring(0, maxChars);
    }
}
