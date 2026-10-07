package com.avanti.search.eval;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * A regression floor for CI: the minimum acceptable nDCG@10 per strategy,
 * matched by pipeline position (Lexical, Semantic, Hybrid, Rerank, in that
 * fixed order) rather than by exact strategy name — the Hybrid/Rerank
 * display names embed the swept RRF constant (e.g. "Hybrid (RRF, k=60)"),
 * which can legitimately change between runs without being a regression.
 */
public record EvalBaseline(List<StrategyBaseline> strategies) {

    public record StrategyBaseline(String name, double minNdcgAt10) {
    }

    public static EvalBaseline load(Path path) throws IOException {
        return new ObjectMapper().readValue(path.toFile(), EvalBaseline.class);
    }
}
