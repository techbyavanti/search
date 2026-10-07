package com.avanti.search.api;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.retrieval.ProductDocument;
import com.avanti.search.retrieval.ProductLookup;
import com.avanti.search.retrieval.ScoredResult;
import com.avanti.search.retrieval.SearchStrategy;
import io.swagger.v3.oas.annotations.Operation;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Thin REST layer over search-retrieval's SearchStrategy implementations —
 * the exact same strategy code search-eval scores offline, so results here
 * are never a different pipeline than the one the eval numbers describe.
 */
@RestController
@RequestMapping("/api/search")
public class SearchController {

    private final Map<StrategyType, SearchStrategy> strategiesByType;
    private final ElasticsearchClient client;

    public SearchController(Map<StrategyType, SearchStrategy> strategiesByType, ElasticsearchClient client) {
        this.strategiesByType = strategiesByType;
        this.client = client;
    }

    @Operation(summary = "Search with a single ranking strategy")
    @GetMapping
    public SearchResult search(
            @RequestParam String query,
            @RequestParam(defaultValue = "RERANK") StrategyType strategy,
            @RequestParam(defaultValue = "10") int topK) {
        SearchStrategy searchStrategy = strategiesByType.get(strategy);
        List<ScoredResult> results = searchStrategy.search(query, topK);
        return new SearchResult(query, searchStrategy.name(), toProductResults(results));
    }

    @Operation(summary = "Run all four ranking strategies for the same query, side by side")
    @GetMapping("/compare")
    public CompareResult compare(
            @RequestParam String query,
            @RequestParam(defaultValue = "10") int topK) {
        Map<String, StrategyResult> resultsByStrategy = new LinkedHashMap<>();
        for (StrategyType type : StrategyType.values()) {
            SearchStrategy searchStrategy = strategiesByType.get(type);

            long start = System.nanoTime();
            List<ScoredResult> results = searchStrategy.search(query, topK);
            long latencyMs = (System.nanoTime() - start) / 1_000_000;

            resultsByStrategy.put(searchStrategy.name(), new StrategyResult(latencyMs, toProductResults(results)));
        }
        return new CompareResult(query, resultsByStrategy);
    }

    private List<ProductResult> toProductResults(List<ScoredResult> results) {
        List<String> ids = results.stream().map(ScoredResult::productId).toList();
        Map<String, ProductDocument> documents = ProductLookup.fetchByIds(client, ids);

        return results.stream()
                .map(result -> {
                    ProductDocument doc = documents.get(result.productId());
                    if (doc == null) {
                        return new ProductResult(result.productId(), result.score(), null, null, null, null, null);
                    }
                    return new ProductResult(result.productId(), result.score(), doc.productName(), doc.productClass(),
                            doc.categoryHierarchy(), doc.averageRating(), doc.ratingCount());
                })
                .toList();
    }
}
