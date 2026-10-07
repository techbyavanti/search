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

    @Test
    void scoresAHigherRrfScoreHigherAllElseEqual() throws IOException {
        try (NeuralRerankerService service = new NeuralRerankerService(MODEL_DIR)) {
            float[] strongCandidate = {0.05f, 0.9f, 1.0f, 1.0f, 4.5f, 3.0f};
            float[] weakCandidate = {0.001f, 0.1f, 0.0f, 0.0f, 3.0f, 0.0f};

            List<Float> scores = service.scoreBatch(List.of(strongCandidate, weakCandidate));

            assertThat(scores.get(0)).isGreaterThan(scores.get(1));
        }
    }

    @Test
    void scoreBatchReturnsOneScorePerInputVector() throws IOException {
        try (NeuralRerankerService service = new NeuralRerankerService(MODEL_DIR)) {
            List<Float> scores = service.scoreBatch(List.of(
                    new float[]{0.03f, 0.5f, 0.5f, 1.0f, 4.0f, 2.0f},
                    new float[]{0.02f, 0.4f, 0.0f, 0.0f, 4.0f, 1.0f},
                    new float[]{0.01f, 0.3f, 0.0f, 0.0f, 4.0f, 0.0f}));

            assertThat(scores).hasSize(3);
        }
    }
}
