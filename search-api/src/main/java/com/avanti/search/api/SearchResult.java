package com.avanti.search.api;

import java.util.List;

public record SearchResult(String query, String strategy, List<ProductResult> results) {
}
