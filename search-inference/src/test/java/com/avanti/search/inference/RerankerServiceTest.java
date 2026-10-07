package com.avanti.search.inference;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises the real ONNX cross-encoder downloaded by
 * scripts/download-models.sh. Skipped (not failed) when the model isn't
 * present, same as EmbeddingServiceTest.
 */
class RerankerServiceTest {

    private static final Path MODEL_DIR = Path.of("../models/ms-marco-MiniLM-L-6-v2");

    @BeforeAll
    static void modelIsPresent() {
        assumeTrue(Files.exists(MODEL_DIR.resolve("model.onnx")),
                "Skipping: run scripts/download-models.sh to fetch the reranker model first");
    }

    @Test
    void scoresARelevantDocumentHigherThanAnIrrelevantOne() throws IOException {
        try (RerankerService reranker = new RerankerService(MODEL_DIR)) {
            String query = "queen size platform bed frame";

            float relevantScore = reranker.score(query, "Solid wood queen platform bed frame with wooden slats.");
            float irrelevantScore = reranker.score(query, "Brass desk lamp with an adjustable arm.");

            assertThat(relevantScore).isGreaterThan(irrelevantScore);
        }
    }

    @Test
    void scoreBatchPreservesRelativeOrderingFromIndividualScoreCalls() throws IOException {
        // Dynamic INT8 quantization computes scale factors per call from the actual batch
        // contents, so exact score values legitimately shift a little depending on what's
        // batched together — asserting exact equality here would be testing an invariant
        // the quantized model doesn't actually have. Relative ranking is what reranking
        // depends on, and that's stable.
        try (RerankerService reranker = new RerankerService(MODEL_DIR)) {
            String query = "platform bed frame";
            String docA = "Solid wood queen platform bed frame.";
            String docB = "Brass desk lamp with an adjustable arm.";

            List<Float> batchScores = reranker.scoreBatch(query, List.of(docA, docB));

            assertThat(batchScores.get(0)).isGreaterThan(batchScores.get(1));
            assertThat(reranker.score(query, docA)).isGreaterThan(reranker.score(query, docB));
        }
    }
}
