package com.avanti.search.eval;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class BaselineCheckerTest {

    @Test
    void noFailuresWhenAllStrategiesMeetTheirFloor() {
        List<StrategySummary> summaries = List.of(
                summary("Lexical (BM25)", 0.6707),
                summary("Semantic (bge-small-en-v1.5)", 0.6990));
        EvalBaseline baseline = new EvalBaseline(List.of(
                new EvalBaseline.StrategyBaseline("Lexical (BM25)", 0.65),
                new EvalBaseline.StrategyBaseline("Semantic (bge-small-en-v1.5)", 0.68)));

        assertThat(BaselineChecker.checkRegressions(summaries, baseline)).isEmpty();
    }

    @Test
    void flagsAStrategyThatFallsBelowItsFloor() {
        List<StrategySummary> summaries = List.of(summary("Lexical (BM25)", 0.60));
        EvalBaseline baseline = new EvalBaseline(List.of(
                new EvalBaseline.StrategyBaseline("Lexical (BM25)", 0.65)));

        List<String> failures = BaselineChecker.checkRegressions(summaries, baseline);

        assertThat(failures).hasSize(1);
        assertThat(failures.get(0)).contains("Lexical (BM25)").contains("0.6000").contains("0.6500");
    }

    @Test
    void matchesByPipelinePositionNotByExactName() {
        // Hybrid's display name embeds the swept RRF constant, which can legitimately
        // change between runs (e.g. k=60 -> k=80) without being a regression.
        List<StrategySummary> summaries = List.of(summary("Hybrid (RRF, k=80)", 0.75));
        EvalBaseline baseline = new EvalBaseline(List.of(
                new EvalBaseline.StrategyBaseline("Hybrid (RRF, k=60)", 0.70)));

        assertThat(BaselineChecker.checkRegressions(summaries, baseline)).isEmpty();
    }

    @Test
    void flagsAStrategyCountMismatch() {
        List<StrategySummary> summaries = List.of(summary("Lexical (BM25)", 0.67));
        EvalBaseline baseline = new EvalBaseline(List.of(
                new EvalBaseline.StrategyBaseline("Lexical (BM25)", 0.65),
                new EvalBaseline.StrategyBaseline("Semantic (bge-small-en-v1.5)", 0.68)));

        List<String> failures = BaselineChecker.checkRegressions(summaries, baseline);

        assertThat(failures).anyMatch(f -> f.contains("mismatch"));
    }

    private static StrategySummary summary(String name, double ndcgAt10) {
        return new StrategySummary(name, ndcgAt10, 0.0, 0.0, 0.0, 0.0, 0L);
    }
}
