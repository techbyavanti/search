package com.avanti.search.retrieval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Bounded in-process LRU cache of {@link CachedProductFeatures}, keyed by
 * product id. Shared by {@link NeuralRerankStrategy} (feature vectors) and
 * {@link HybridRerankStrategy} (cross-encoder text) so a product fetched by
 * either strategy doesn't need a second Elasticsearch round trip for the
 * other. The fetch callback is injected rather than taking an
 * ElasticsearchClient directly, so tests can supply a fake without needing
 * a real cluster — production callers pass {@code ids ->
 * ProductLookup.fetchByIds(client, ids)}.
 */
public final class ProductFeatureCache {

    private final Function<List<String>, Map<String, ProductDocument>> fetcher;
    private final Map<String, CachedProductFeatures> cache;

    public ProductFeatureCache(Function<List<String>, Map<String, ProductDocument>> fetcher, int maxSize) {
        this.fetcher = fetcher;
        this.cache = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, CachedProductFeatures> eldest) {
                return size() > maxSize;
            }
        };
    }

    /** Returns cached/fetched features for the given ids; ids not found in Elasticsearch are simply absent from the result. */
    public synchronized Map<String, CachedProductFeatures> getBatch(List<String> productIds) {
        Map<String, CachedProductFeatures> result = new HashMap<>();
        List<String> missing = new ArrayList<>();
        for (String productId : productIds) {
            CachedProductFeatures cached = cache.get(productId);
            if (cached != null) {
                result.put(productId, cached);
            } else {
                missing.add(productId);
            }
        }

        if (!missing.isEmpty()) {
            Map<String, ProductDocument> fetched = fetcher.apply(missing);
            for (Map.Entry<String, ProductDocument> entry : fetched.entrySet()) {
                CachedProductFeatures features = CachedProductFeatures.from(entry.getValue());
                cache.put(entry.getKey(), features);
                result.put(entry.getKey(), features);
            }
        }

        return result;
    }

    public synchronized int size() {
        return cache.size();
    }
}
