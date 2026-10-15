package com.avanti.search.api;

import java.util.Map;

/** Keyed by strategy display name (e.g. "Hybrid (RRF, k=40)"), in the same order as StrategyType's declared values. */
public record CompareResult(String query, Map<String, StrategyResult> resultsByStrategy) {
}
