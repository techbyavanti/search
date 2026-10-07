package com.avanti.search.retrieval;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

class ProductFeatureCacheTest {

    private static ProductDocument fakeDocument(String id) {
        return new ProductDocument(id, "Product " + id, "Class", "Category", "Description",
                "features", 10, 4.0, 5, new float[]{1.0f, 0.0f}, null);
    }

    @Test
    void fetchesMissingIdsAndCachesTheResult() {
        AtomicInteger fetchCalls = new AtomicInteger();
        ProductFeatureCache cache = new ProductFeatureCache(ids -> {
            fetchCalls.incrementAndGet();
            Map<String, ProductDocument> result = new HashMap<>();
            for (String id : ids) {
                result.put(id, fakeDocument(id));
            }
            return result;
        }, 100);

        Map<String, CachedProductFeatures> first = cache.getBatch(List.of("a", "b"));
        assertThat(first).containsOnlyKeys("a", "b");
        assertThat(fetchCalls.get()).isEqualTo(1);

        // Second call for the same ids should be served entirely from cache.
        Map<String, CachedProductFeatures> second = cache.getBatch(List.of("a", "b"));
        assertThat(second).containsOnlyKeys("a", "b");
        assertThat(fetchCalls.get()).isEqualTo(1);
    }

    @Test
    void onlyFetchesTheIdsNotAlreadyCached() {
        AtomicInteger fetchCalls = new AtomicInteger();
        ProductFeatureCache cache = new ProductFeatureCache(ids -> {
            fetchCalls.incrementAndGet();
            Map<String, ProductDocument> result = new HashMap<>();
            for (String id : ids) {
                result.put(id, fakeDocument(id));
            }
            return result;
        }, 100);

        cache.getBatch(List.of("a"));
        Map<String, CachedProductFeatures> mixed = cache.getBatch(List.of("a", "b"));

        assertThat(mixed).containsOnlyKeys("a", "b");
        assertThat(fetchCalls.get()).isEqualTo(2); // one for "a", one for the batch containing just "b"
    }

    @Test
    void idsMissingFromTheFetcherResultAreSimplyAbsent() {
        ProductFeatureCache cache = new ProductFeatureCache(ids -> Map.of(), 100);

        Map<String, CachedProductFeatures> result = cache.getBatch(List.of("missing"));

        assertThat(result).isEmpty();
    }

    @Test
    void evictsLeastRecentlyUsedEntriesBeyondMaxSize() {
        ProductFeatureCache cache = new ProductFeatureCache(ids -> {
            Map<String, ProductDocument> result = new HashMap<>();
            for (String id : ids) {
                result.put(id, fakeDocument(id));
            }
            return result;
        }, 2);

        cache.getBatch(List.of("a"));
        cache.getBatch(List.of("b"));
        assertThat(cache.size()).isEqualTo(2);

        cache.getBatch(List.of("c")); // evicts "a", the least recently touched
        assertThat(cache.size()).isEqualTo(2);

        AtomicInteger fetchCalls = new AtomicInteger();
        ProductFeatureCache countingCache = new ProductFeatureCache(ids -> {
            fetchCalls.incrementAndGet();
            Map<String, ProductDocument> result = new HashMap<>();
            for (String id : ids) {
                result.put(id, fakeDocument(id));
            }
            return result;
        }, 2);
        countingCache.getBatch(List.of("a"));
        countingCache.getBatch(List.of("b"));
        countingCache.getBatch(List.of("c")); // evicts "a"
        countingCache.getBatch(List.of("a")); // must refetch, since "a" was evicted

        assertThat(fetchCalls.get()).isEqualTo(4);
    }
}
