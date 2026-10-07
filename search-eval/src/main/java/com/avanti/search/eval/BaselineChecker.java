package com.avanti.search.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Compares a completed eval run's summaries against an EvalBaseline's regression floors. */
public final class BaselineChecker {

    private BaselineChecker() {
    }

    public static List<String> checkRegressions(List<StrategySummary> summaries, EvalBaseline baseline) {
        List<String> failures = new ArrayList<>();
        int count = Math.min(summaries.size(), baseline.strategies().size());

        for (int i = 0; i < count; i++) {
            StrategySummary summary = summaries.get(i);
            EvalBaseline.StrategyBaseline expected = baseline.strategies().get(i);
            if (summary.ndcgAt10() < expected.minNdcgAt10()) {
                failures.add(String.format(Locale.ROOT,
                        "%s: nDCG@10 %.4f is below the baseline floor %.4f (baseline entry: \"%s\")",
                        summary.strategyName(), summary.ndcgAt10(), expected.minNdcgAt10(), expected.name()));
            }
        }

        if (summaries.size() != baseline.strategies().size()) {
            failures.add(String.format(Locale.ROOT,
                    "Strategy count mismatch: eval ran %d strategies, baseline has %d entries — update %s",
                    summaries.size(), baseline.strategies().size(), "the baseline file"));
        }

        return failures;
    }
}
