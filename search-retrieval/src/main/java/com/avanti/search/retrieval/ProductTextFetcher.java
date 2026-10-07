package com.avanti.search.retrieval;

import com.avanti.search.common.EmbeddingTextBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Builds embedding-style text for a candidate set of ids, for reranking. */
public final class ProductTextFetcher {

    private ProductTextFetcher() {
    }

    /** Sources product fields from the shared ProductFeatureCache rather than a fresh Elasticsearch fetch every call. */
    public static Map<String, String> fetchTexts(ProductFeatureCache featureCache, List<String> productIds) {
        Map<String, CachedProductFeatures> features = featureCache.getBatch(productIds);

        Map<String, String> texts = new HashMap<>();
        for (Map.Entry<String, CachedProductFeatures> entry : features.entrySet()) {
            CachedProductFeatures product = entry.getValue();
            texts.put(entry.getKey(), EmbeddingTextBuilder.build(
                    product.productName(), product.productClass(), product.categoryHierarchy(), product.productDescription()));
        }
        return texts;
    }
}
