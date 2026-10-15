package com.avanti.search.retrieval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Bounded in-process LRU cache of cross-encoder scores, keyed by
 * (normalized query, product id). A cross-encoder reads query and product
 * together, so nothing can be precomputed per product — but a repeated
 * query (the head of real search traffic) can skip the forward pass
 * entirely. Only the ids not already cached are sent to the scorer.
 *
 * <p>Query normalization ({@link QueryNormalizer}) doesn't change scores.
 * A {@code maxSize} of 0 disables caching — search-eval uses
 * that so its latency numbers measure the model, not cache hits. The
 * scorer runs outside the lock, so concurrent requests don't serialize on
 * model inference.
 */
public final class RerankScoreCache {

    private record Key(String query, String productId) {
    }

    private final int maxSize;
    private final Map<Key, Float> cache;

    public RerankScoreCache(int maxSize) {
        this.maxSize = maxSize;
        this.cache = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Key, Float> eldest) {
                return size() > maxSize;
            }
        };
    }

    public static RerankScoreCache disabled() {
        return new RerankScoreCache(0);
    }

    /**
     * Returns a score for every given id, calling {@code scorer} only for the
     * ids not already cached. The scorer must return a score for every id it
     * is given.
     */
    public Map<String, Float> getOrScore(String query, List<String> productIds,
                                         Function<List<String>, Map<String, Float>> scorer) {
        if (maxSize <= 0) {
            return scorer.apply(productIds);
        }

        String normalizedQuery = QueryNormalizer.normalize(query);
        Map<String, Float> result = new HashMap<>();
        List<String> missing = new ArrayList<>();
        synchronized (this) {
            for (String productId : productIds) {
                Float cached = cache.get(new Key(normalizedQuery, productId));
                if (cached != null) {
                    result.put(productId, cached);
                } else {
                    missing.add(productId);
                }
            }
        }

        if (!missing.isEmpty()) {
            Map<String, Float> scored = scorer.apply(missing);
            synchronized (this) {
                for (Map.Entry<String, Float> entry : scored.entrySet()) {
                    cache.put(new Key(normalizedQuery, entry.getKey()), entry.getValue());
                }
            }
            result.putAll(scored);
        }

        return result;
    }

    public synchronized int size() {
        return cache.size();
    }
}
