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
 * Exercises the real ONNX model produced by
 * scripts/train-neural-reranker.sh. Skipped (not failed) when the model
 * isn't present, same as EmbeddingServiceTest/RerankerServiceTest.
 */
class NeuralRerankerServiceTest {

    private static final Path MODEL_DIR = Path.of("../models/neural-reranker");

    @BeforeAll
    static void modelIsPresent() {
        assumeTrue(Files.exists(MODEL_DIR.resolve("model.onnx")),
                "Skipping: run scripts/train-neural-reranker.sh to train the neural reranker model first");
    }

    // Order: search-retrieval's RerankFeatureBuilder.FEATURE_NAMES (17 features).
    private static final float[] STRONG = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.9f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 4.5f, 5.0f, 1.0f};
    private static final float[] MIDDLE = {0.6f, 0.1f, 0.1f, 0.5f, 0.1f, 0.8f, -0.1f, 0.5f, 0.5f, 0.0f, 0.5f, 0.5f, 0.0f, 2.0f, 4.0f, 2.0f, 1.0f};
    private static final float[] WEAK = {0.3f, 0.02f, 0.0f, 0.0f, 0.02f, 0.6f, -0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 1.0f};

    @Test
    void reportsTheFeatureCountItWasTrainedOn() throws IOException {
        try (NeuralRerankerService service = new NeuralRerankerService(MODEL_DIR)) {
            assertThat(service.featureCount()).isEqualTo(STRONG.length);
        }
    }

    @Test
    void scoresAStrongMatchAboveAWeakOne() throws IOException {
        try (NeuralRerankerService service = new NeuralRerankerService(MODEL_DIR)) {
            List<Float> scores = service.scoreBatch(List.of(STRONG, WEAK));

            assertThat(scores.get(0)).isGreaterThan(scores.get(1));
        }
    }

    @Test
    void scoreBatchReturnsOneScorePerInputVector() throws IOException {
        try (NeuralRerankerService service = new NeuralRerankerService(MODEL_DIR)) {
            List<Float> scores = service.scoreBatch(List.of(STRONG, MIDDLE, WEAK));

            assertThat(scores).hasSize(3);
        }
    }
}
