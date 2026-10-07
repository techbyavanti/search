package com.avanti.search.api;

import java.util.List;

public record StrategyResult(long latencyMs, List<ProductResult> results) {
}
