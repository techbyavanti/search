package com.avanti.search.inference;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises the real ONNX model + tokenizer downloaded by
 * scripts/download-models.sh. Skipped (not failed) when the model isn't
 * present, since models/ is gitignored and fetched on demand, same as
 * dataset/ for WandsCsvLoader-based tests.
 */
class EmbeddingServiceTest {

    private static final Path MODEL_DIR = Path.of("../models/bge-small-en-v1.5");

    @BeforeAll
    static void modelIsPresent() {
        assumeTrue(Files.exists(MODEL_DIR.resolve("model.onnx")),
                "Skipping: run scripts/download-models.sh to fetch the embedding model first");
    }

    @Test
    void producesNormalized384DimensionalVectors() throws IOException {
        try (EmbeddingService service = new EmbeddingService(MODEL_DIR)) {
            float[] embedding = service.embedDocument("A modern platform bed frame with a wooden headboard.");

            assertThat(embedding).hasSize(EmbeddingService.DIMENSIONS);
            assertThat(norm(embedding)).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-4));
        }
    }

    @Test
    void similarTextsAreCloserThanDissimilarOnes() throws IOException {
        try (EmbeddingService service = new EmbeddingService(MODEL_DIR)) {
            float[] bedFrame = service.embedDocument("A queen size platform bed frame with wooden slats.");
            float[] bedFrameSimilar = service.embedDocument("Wooden queen bed frame with slatted platform base.");
            float[] deskLamp = service.embedDocument("A brass desk lamp with an adjustable arm.");

            double similarSimilarity = cosineSimilarity(bedFrame, bedFrameSimilar);
            double dissimilarSimilarity = cosineSimilarity(bedFrame, deskLamp);

            assertThat(similarSimilarity).isGreaterThan(dissimilarSimilarity);
        }
    }

    @Test
    void queryEmbeddingDiffersFromDocumentEmbeddingForSameText() throws IOException {
        try (EmbeddingService service = new EmbeddingService(MODEL_DIR)) {
            String text = "platform bed frame";

            float[] asQuery = service.embedQuery(text);
            float[] asDocument = service.embedDocument(text);

            assertThat(asQuery).isNotEqualTo(asDocument);
        }
    }

    private static double norm(float[] vector) {
        double sumSquares = 0.0;
        for (float v : vector) {
            sumSquares += (double) v * v;
        }
        return Math.sqrt(sumSquares);
    }

    private static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }
}
