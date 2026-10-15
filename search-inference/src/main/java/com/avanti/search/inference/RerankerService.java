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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
 *
 * <p>With one thread per forward pass, a single request would use only one
 * core. So {@link #scoreBatch} splits a large batch into chunks of
 * {@code chunkSize} pairs and runs the chunks in parallel, each as its own
 * single-threaded forward pass through the same semaphore. A lone request
 * then spreads across cores, while a burst of concurrent requests still
 * shares them instead of oversubscribing. Smaller chunks also pad to their
 * own longest pair rather than the whole batch's. Because the model uses
 * dynamic INT8 quantization (scales computed per forward pass), a pair's
 * exact score can shift slightly with chunk boundaries; relative ordering
 * is what reranking depends on, and the eval harness measures it.
 */
public final class RerankerService implements AutoCloseable {

    private static final int MAX_SEQUENCE_LENGTH = 256;
    private static final String OUTPUT_NAME = "logits";
    public static final int DEFAULT_CHUNK_SIZE = 5;

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final Semaphore inferenceGate = new Semaphore(Math.max(1, Runtime.getRuntime().availableProcessors()));
    private final ExecutorService chunkExecutor = Executors.newVirtualThreadPerTaskExecutor();
    private final int chunkSize;

    public RerankerService(Path modelDir) throws IOException {
        this(modelDir, DEFAULT_CHUNK_SIZE);
    }

    /** @param chunkSize max (query, document) pairs per forward pass; larger batches are split and run in parallel */
    public RerankerService(Path modelDir, int chunkSize) throws IOException {
        if (chunkSize < 1) {
            throw new IllegalArgumentException("chunkSize must be >= 1, was " + chunkSize);
        }
        this.chunkSize = chunkSize;
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

    /**
     * Scores a query against a batch of documents, in input order. Batches
     * larger than {@code chunkSize} are split into chunks scored in parallel.
     */
    public List<Float> scoreBatch(String query, List<String> documents) {
        if (documents.size() <= chunkSize) {
            return scoreChunk(query, documents);
        }

        List<Future<List<Float>>> chunks = new ArrayList<>();
        for (int start = 0; start < documents.size(); start += chunkSize) {
            List<String> chunk = documents.subList(start, Math.min(start + chunkSize, documents.size()));
            chunks.add(chunkExecutor.submit(() -> scoreChunk(query, chunk)));
        }

        List<Float> scores = new ArrayList<>(documents.size());
        try {
            for (Future<List<Float>> chunk : chunks) {
                scores.addAll(chunk.get());
            }
        } catch (InterruptedException e) {
            chunks.forEach(chunk -> chunk.cancel(true));
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while reranking", e);
        } catch (ExecutionException e) {
            chunks.forEach(chunk -> chunk.cancel(true));
            if (e.getCause() instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
            throw new IllegalStateException("Reranking failed", e.getCause());
        }
        return scores;
    }

    /** Scores one chunk of documents in a single forward pass. */
    private List<Float> scoreChunk(String query, List<String> documents) {
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
        chunkExecutor.close();
        tokenizer.close();
        try {
            session.close();
        } catch (ai.onnxruntime.OrtException e) {
            throw new UncheckedIOException(new IOException("Failed to close ONNX session", e));
        }
    }
}
