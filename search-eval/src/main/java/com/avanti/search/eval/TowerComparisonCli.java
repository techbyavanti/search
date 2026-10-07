package com.avanti.search.eval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.common.WandsCsvLoader;
import com.avanti.search.common.WandsDataset;
import com.avanti.search.common.WandsQuery;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.LearnedTowerSearchStrategy;
import com.avanti.search.retrieval.ScoredResult;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Standalone, throwaway driver for Track B's staged tower comparison — NOT
 * part of the permanent 5-strategy EvalCli flow (deliberately kept
 * separate so EvalCli doesn't churn before a winner is chosen; once one
 * is, LearnedTowerSearchStrategy gets wired into EvalCli properly as a 6th
 * strategy the same way NeuralRerankStrategy was).
 *
 * Usage: java -cp search-eval/target/search-eval.jar
 *        com.avanti.search.eval.TowerComparisonCli <model-dir> <label>
 *
 * Scores LearnedTowerSearchStrategy (against whatever is currently in the
 * learned_embedding field — see IngestionCli --learned-model-dir) on the
 * same held-out 20% query split Track A's neural reranker uses, so the
 * two tracks' held-out sets are directly comparable in spirit even though
 * they measure different strategies.
 */
public final class TowerComparisonCli {

    private static final int TOP_K = 50;
    private static final int NDCG_K = 10;

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: TowerComparisonCli <query-tower-model-dir> <label>");
            System.exit(1);
        }
        Path modelDir = Path.of(args[0]);
        String label = args[1];

        WandsDataset dataset = WandsCsvLoader.load(Path.of("dataset"));
        Set<String> heldOutIds = new HashSet<>(Files.readAllLines(
                Path.of("search-eval/src/main/resources/neural-reranker-eval-queries.txt"), StandardCharsets.UTF_8));
        List<WandsQuery> heldOutQueries = dataset.queries().stream()
                .filter(q -> heldOutIds.contains(q.queryId()))
                .toList();
        System.out.println("Scoring " + label + " on " + heldOutQueries.size() + " held-out queries...");

        try (ElasticsearchClient client = ElasticsearchClients.create("http://localhost:9200");
             EmbeddingService queryTowerEmbeddingService = new EmbeddingService(modelDir)) {
            LearnedTowerSearchStrategy strategy = new LearnedTowerSearchStrategy(client, queryTowerEmbeddingService);

            List<QueryMetrics> perQuery = new ArrayList<>();
            for (WandsQuery query : heldOutQueries) {
                List<ScoredResult> results = strategy.search(query.queryText(), TOP_K);
                List<String> rankedIds = results.stream().map(ScoredResult::productId).toList();
                var judgments = dataset.judgmentsFor(query.queryId());
                perQuery.add(new QueryMetrics(
                        query.queryId(),
                        MetricsCalculator.ndcgAtK(rankedIds, judgments, NDCG_K),
                        MetricsCalculator.reciprocalRank(rankedIds, judgments),
                        MetricsCalculator.recallAtK(rankedIds, judgments, 10),
                        MetricsCalculator.recallAtK(rankedIds, judgments, TOP_K),
                        MetricsCalculator.precisionAtK(rankedIds, judgments, 10)
                ));
            }

            StrategySummary summary = StrategySummary.aggregate(label, perQuery, List.of());
            System.out.printf(Locale.ROOT,
                    "%s: nDCG@10=%.4f MRR=%.4f Recall@10=%.4f Recall@50=%.4f Precision@10=%.4f%n",
                    summary.strategyName(), summary.ndcgAt10(), summary.mrr(),
                    summary.recallAt10(), summary.recallAt50(), summary.precisionAt10());
        }
    }
}
