package com.avanti.search.retrieval;

import com.avanti.search.inference.QueryEncoder;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Bounded LRU cache in front of a {@link QueryEncoder}, keyed by normalized
 * query text ({@link QueryNormalizer}). One request can encode the same
 * query more than once: Neural Rerank embeds it inside Hybrid's semantic
 * retrieval and again for its cosine feature, and {@code /compare} runs four
 * strategies that each embed it. With this cache, the model runs once.
 *
 * <p>The encoder runs outside the lock, so concurrent requests don't
 * serialize on inference (two threads may both encode one new query; the
 * results are identical). Returns copies, so callers can't change cached
 * vectors.
 */
public final class CachingQueryEncoder implements QueryEncoder {

    private final QueryEncoder delegate;
    private final Map<String, float[]> cache;

    public CachingQueryEncoder(QueryEncoder delegate, int maxSize) {
        if (maxSize < 1) {
            throw new IllegalArgumentException("maxSize must be at least 1, got " + maxSize);
        }
        this.delegate = delegate;
        this.cache = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
                return size() > maxSize;
            }
        };
    }

    @Override
    public float[] embedQuery(String query) {
        String key = QueryNormalizer.normalize(query);
        float[] cached;
        synchronized (this) {
            cached = cache.get(key);
        }
        if (cached == null) {
            cached = delegate.embedQuery(query);
            synchronized (this) {
                cache.put(key, cached);
            }
        }
        return cached.clone();
    }

    public synchronized int size() {
        return cache.size();
    }
}
