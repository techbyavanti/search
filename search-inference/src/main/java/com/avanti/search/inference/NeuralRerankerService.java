package com.avanti.search.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Scores rerank feature vectors (see search-retrieval's RerankFeatureBuilder,
 * whose fixed feature order is this model's input contract) with a small
 * MLP via ONNX Runtime. Unlike RerankerService, there's no tokenizer — the
 * input is already a fixed-length numeric vector, not text, which is what
 * makes this strategy fast: no transformer forward pass, just a handful of
 * cheap linear layers over mostly-cached features.
 */
public final class NeuralRerankerService implements AutoCloseable {

    private static final String INPUT_NAME = "features";
    private static final String OUTPUT_NAME = "score";

    private final OrtEnvironment environment;
    private final OrtSession session;

    public NeuralRerankerService(Path modelDir) throws IOException {
        this.environment = OrtEnvironment.getEnvironment();
        try {
            this.session = environment.createSession(modelDir.resolve("model.onnx").toString(), new OrtSession.SessionOptions());
        } catch (ai.onnxruntime.OrtException e) {
            throw new IOException("Failed to load ONNX model from " + modelDir, e);
        }
    }

    /** Scores a batch of feature vectors (each the length RerankFeatureBuilder produces) in one forward pass. */
    public List<Float> scoreBatch(List<float[]> featureVectors) {
        int batchSize = featureVectors.size();
        int featureCount = featureVectors.get(0).length;

        FloatBuffer input = FloatBuffer.allocate(batchSize * featureCount);
        for (float[] vector : featureVectors) {
            input.put(vector);
        }
        input.rewind();

        long[] shape = {batchSize, featureCount};
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(environment, input, shape);
             OrtSession.Result result = session.run(Map.of(INPUT_NAME, inputTensor))) {

            float[][] scores = (float[][]) result.get(OUTPUT_NAME)
                    .orElseThrow(() -> new IllegalStateException("Model produced no '" + OUTPUT_NAME + "' output"))
                    .getValue();

            List<Float> batchScores = new ArrayList<>(batchSize);
            for (float[] score : scores) {
                batchScores.add(score[0]);
            }
            return batchScores;
        } catch (ai.onnxruntime.OrtException e) {
            throw new UncheckedIOException(new IOException("ONNX inference failed", e));
        }
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (ai.onnxruntime.OrtException e) {
            throw new UncheckedIOException(new IOException("Failed to close ONNX session", e));
        }
    }
}
