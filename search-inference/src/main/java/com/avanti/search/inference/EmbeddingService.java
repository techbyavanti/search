package com.avanti.search.inference;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Embeds text with {@code bge-small-en-v1.5} via ONNX Runtime + a standalone
 * HuggingFace tokenizer — no Python inference server. BGE models are trained
 * with CLS-token pooling (the first token of last_hidden_state), not mean
 * pooling, and require an asymmetric instruction prefix on queries only; both
 * are documented on the model card and easy to get wrong since most sentence-
 * embedding models (e.g. plain sentence-transformers) mean-pool instead.
 */
public final class EmbeddingService implements AutoCloseable {

    public static final int DIMENSIONS = 384;

    private static final String QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
    private static final int MAX_SEQUENCE_LENGTH = 256;
    private static final String OUTPUT_NAME = "last_hidden_state";

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;

    public EmbeddingService(Path modelDir) throws IOException {
        this.environment = OrtEnvironment.getEnvironment();
        try {
            this.session = environment.createSession(modelDir.resolve("model.onnx").toString(), new OrtSession.SessionOptions());
        } catch (ai.onnxruntime.OrtException e) {
            throw new IOException("Failed to load ONNX model from " + modelDir, e);
        }
        this.tokenizer = HuggingFaceTokenizer.builder()
                .optTokenizerPath(modelDir.resolve("tokenizer.json"))
                .optPadding(true)
                .optTruncation(true)
                .optMaxLength(MAX_SEQUENCE_LENGTH)
                .build();
    }

    /** Embeds a search query, applying the asymmetric query instruction prefix BGE was trained with. */
    public float[] embedQuery(String query) {
        return embedBatch(List.of(QUERY_PREFIX + query)).get(0);
    }

    /** Embeds a single product/document text (no instruction prefix). */
    public float[] embedDocument(String text) {
        return embedBatch(List.of(text)).get(0);
    }

    /** Embeds a batch of product/document texts (no instruction prefix) in one forward pass. */
    public List<float[]> embedDocuments(List<String> texts) {
        return embedBatch(texts);
    }

    private List<float[]> embedBatch(List<String> texts) {
        Encoding[] encodings = tokenizer.batchEncode(texts);
        int batchSize = encodings.length;
        int seqLen = encodings[0].getIds().length;

        LongBuffer inputIds = LongBuffer.allocate(batchSize * seqLen);
        LongBuffer attentionMask = LongBuffer.allocate(batchSize * seqLen);
        LongBuffer tokenTypeIds = LongBuffer.allocate(batchSize * seqLen);
        for (Encoding encoding : encodings) {
            inputIds.put(encoding.getIds());
            attentionMask.put(encoding.getAttentionMask());
            tokenTypeIds.put(encoding.getTypeIds());
        }
        inputIds.rewind();
        attentionMask.rewind();
        tokenTypeIds.rewind();

        long[] shape = {batchSize, seqLen};
        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(environment, inputIds, shape);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(environment, attentionMask, shape);
             OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(environment, tokenTypeIds, shape);
             OrtSession.Result result = session.run(Map.of(
                     "input_ids", inputIdsTensor,
                     "attention_mask", attentionMaskTensor,
                     "token_type_ids", tokenTypeIdsTensor))) {

            float[][][] lastHiddenState = (float[][][]) result.get(OUTPUT_NAME)
                    .orElseThrow(() -> new IllegalStateException("Model produced no '" + OUTPUT_NAME + "' output"))
                    .getValue();

            List<float[]> embeddings = new ArrayList<>(batchSize);
            for (float[][] tokenVectors : lastHiddenState) {
                embeddings.add(normalize(tokenVectors[0]));
            }
            return embeddings;
        } catch (ai.onnxruntime.OrtException e) {
            throw new UncheckedIOException(new IOException("ONNX inference failed", e));
        }
    }

    private static float[] normalize(float[] vector) {
        double sumSquares = 0.0;
        for (float v : vector) {
            sumSquares += (double) v * v;
        }
        double norm = Math.sqrt(sumSquares);
        if (norm == 0.0) {
            return vector;
        }
        float[] normalized = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = (float) (vector[i] / norm);
        }
        return normalized;
    }

    @Override
    public void close() {
        tokenizer.close();
        try {
            session.close();
        } catch (ai.onnxruntime.OrtException e) {
            throw new UncheckedIOException(new IOException("Failed to close ONNX session", e));
        }
    }
}
