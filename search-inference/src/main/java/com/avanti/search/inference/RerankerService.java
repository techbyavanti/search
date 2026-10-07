package com.avanti.search.inference;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.util.PairList;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;

/**
 * Scores (query, document) pairs with the {@code ms-marco-MiniLM-L-6-v2}
 * cross-encoder via ONNX Runtime. Unlike EmbeddingService, this model has a
 * sequence-classification head that outputs a single relevance logit per
 * pair directly (no pooling needed) — higher is more relevant. The pair is
 * tokenized together (query and document in one sequence, BERT-style),
 * which is what lets a cross-encoder attend across both texts jointly and
 * score more accurately than comparing two independently-encoded vectors,
 * at the cost of only being usable to rescore a small candidate set rather
 * than search a whole index.
 *
 * <p>Each call scores a whole candidate pool (e.g. 50 documents) in one
 * forward pass, which is far more compute than a single embedding call. A
 * caller running many queries concurrently (e.g. the eval harness's
 * virtual-thread-per-query loop) would otherwise fire that many full
 * forward passes at once — and since ONNX Runtime sessions default to
 * intra-op parallelism across all cores, concurrent callers multiply
 * rather than share that parallelism, thrashing the CPU. Session threading
 * is pinned to 1 and callers are gated by a semaphore sized to the core
 * count instead, so at most one forward pass runs per core.
 */
public final class RerankerService implements AutoCloseable {

    private static final int MAX_SEQUENCE_LENGTH = 256;
    private static final String OUTPUT_NAME = "logits";

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final Semaphore inferenceGate = new Semaphore(Math.max(1, Runtime.getRuntime().availableProcessors()));

    public RerankerService(Path modelDir) throws IOException {
        this.environment = OrtEnvironment.getEnvironment();
        try {
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setIntraOpNumThreads(1);
            this.session = environment.createSession(modelDir.resolve("model.onnx").toString(), options);
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

    /** Scores one (query, document) pair; higher means more relevant. */
    public float score(String query, String document) {
        return scoreBatch(query, List.of(document)).get(0);
    }

    /** Scores a query against a batch of documents in one forward pass. */
    public List<Float> scoreBatch(String query, List<String> documents) {
        PairList<String, String> pairs = new PairList<>(Collections.nCopies(documents.size(), query), documents);
        Encoding[] encodings = tokenizer.batchEncode(pairs);

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
        inferenceGate.acquireUninterruptibly();
        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(environment, inputIds, shape);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(environment, attentionMask, shape);
             OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(environment, tokenTypeIds, shape);
             OrtSession.Result result = session.run(Map.of(
                     "input_ids", inputIdsTensor,
                     "attention_mask", attentionMaskTensor,
                     "token_type_ids", tokenTypeIdsTensor))) {

            float[][] logits = (float[][]) result.get(OUTPUT_NAME)
                    .orElseThrow(() -> new IllegalStateException("Model produced no '" + OUTPUT_NAME + "' output"))
                    .getValue();

            List<Float> scores = new ArrayList<>(batchSize);
            for (float[] logit : logits) {
                scores.add(logit[0]);
            }
            return scores;
        } catch (ai.onnxruntime.OrtException e) {
            throw new UncheckedIOException(new IOException("ONNX inference failed", e));
        } finally {
            inferenceGate.release();
        }
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
