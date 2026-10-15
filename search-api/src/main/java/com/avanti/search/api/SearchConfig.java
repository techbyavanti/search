package com.avanti.search.api;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.inference.NeuralRerankerService;
import com.avanti.search.inference.RerankerService;
import com.avanti.search.retrieval.CachingQueryEncoder;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.HybridRerankStrategy;
import com.avanti.search.retrieval.HybridRrfSearchStrategy;
import com.avanti.search.retrieval.LearnedTowerSearchStrategy;
import com.avanti.search.retrieval.LexicalSearchStrategy;
import com.avanti.search.retrieval.NeuralRerankStrategy;
import com.avanti.search.retrieval.ProductFeatureCache;
import com.avanti.search.retrieval.ProductLookup;
import com.avanti.search.retrieval.RerankScoreCache;
import com.avanti.search.retrieval.SearchStrategy;
import com.avanti.search.retrieval.SemanticSearchStrategy;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.util.Map;

/** Wires the six SearchStrategy implementations exactly as search-eval does, over one shared ElasticsearchClient. */
@Configuration
public class SearchConfig {

    @Bean
    public ElasticsearchClient elasticsearchClient(SearchProperties properties) {
        return ElasticsearchClients.create(properties.elasticsearch().host());
    }

    @Bean
    public EmbeddingService embeddingService(SearchProperties properties) throws IOException {
        return new EmbeddingService(properties.models().embeddingDir());
    }

    /**
     * The winning (shared-tower) fine-tuned encoder from Track B's tower
     * comparison — see LearnedTowerSearchStrategy's Javadoc and TRAINING.md
     * for the head-to-head numbers against a two-tower alternative. A
     * distinct bean from {@code embeddingService} despite the same Java
     * type; disambiguated from it by parameter name below, Spring's usual
     * fallback when multiple beans share a type.
     */
    @Bean
    public EmbeddingService learnedTowerEmbeddingService(SearchProperties properties) throws IOException {
        return new EmbeddingService(properties.models().learnedTowerDir());
    }

    @Bean
    public RerankerService rerankerService(SearchProperties properties) throws IOException {
        return new RerankerService(properties.models().rerankerDir(), properties.reranker().chunkSize());
    }

    @Bean
    public NeuralRerankerService neuralRerankerService(SearchProperties properties) throws IOException {
        return new NeuralRerankerService(properties.models().neuralRerankerDir());
    }

    @Bean
    public ProductFeatureCache productFeatureCache(ElasticsearchClient client, SearchProperties properties) {
        return new ProductFeatureCache(ids -> ProductLookup.fetchFeaturesByIds(client, ids), properties.featureCache().maxSize());
    }

    /** Shared by every strategy that embeds the query with bge-small-en-v1.5, so a request encodes it once. */
    @Bean
    public CachingQueryEncoder queryEncoder(EmbeddingService embeddingService, SearchProperties properties) {
        return new CachingQueryEncoder(embeddingService, properties.queryEmbeddingCache().maxSize());
    }

    @Bean
    public RerankScoreCache rerankScoreCache(SearchProperties properties) {
        return new RerankScoreCache(properties.reranker().scoreCacheMaxSize());
    }

    @Bean
    public LexicalSearchStrategy lexicalSearchStrategy(ElasticsearchClient client) {
        return new LexicalSearchStrategy(client);
    }

    @Bean
    public SemanticSearchStrategy semanticSearchStrategy(ElasticsearchClient client, CachingQueryEncoder queryEncoder) {
        return new SemanticSearchStrategy(client, queryEncoder);
    }

    /** Same embeddings in a float HNSW field, with graph and query parameters tuned by HnswTuningCli. */
    @Bean
    public SemanticSearchStrategy semanticHnswSearchStrategy(
            ElasticsearchClient client, CachingQueryEncoder queryEncoder, SearchProperties properties) {
        return SemanticSearchStrategy.tunedHnsw(client, queryEncoder, properties.semanticHnsw().numCandidates());
    }

    @Bean
    public HybridRrfSearchStrategy hybridRrfSearchStrategy(
            LexicalSearchStrategy lexical, @Qualifier("semanticSearchStrategy") SemanticSearchStrategy semantic,
            SearchProperties properties) {
        return new HybridRrfSearchStrategy(lexical, semantic, properties.hybrid().candidatePoolSize(), properties.hybrid().rrfK());
    }

    @Bean
    public HybridRerankStrategy hybridRerankStrategy(
            HybridRrfSearchStrategy hybrid, ProductFeatureCache featureCache, RerankerService rerankerService,
            RerankScoreCache rerankScoreCache, SearchProperties properties) {
        return new HybridRerankStrategy(hybrid, featureCache, rerankerService, rerankScoreCache, properties.hybrid().candidatePoolSize());
    }

    @Bean
    public NeuralRerankStrategy neuralRerankStrategy(
            HybridRrfSearchStrategy hybrid, CachingQueryEncoder queryEncoder,
            NeuralRerankerService neuralRerankerService, ProductFeatureCache featureCache, SearchProperties properties) {
        return new NeuralRerankStrategy(hybrid, queryEncoder, neuralRerankerService, featureCache, properties.hybrid().candidatePoolSize());
    }

    @Bean
    public LearnedTowerSearchStrategy learnedTowerSearchStrategy(
            ElasticsearchClient client, EmbeddingService learnedTowerEmbeddingService) {
        return new LearnedTowerSearchStrategy(client, learnedTowerEmbeddingService);
    }

    @Bean
    public Map<StrategyType, SearchStrategy> strategiesByType(
            LexicalSearchStrategy lexical,
            @Qualifier("semanticSearchStrategy") SemanticSearchStrategy semantic,
            @Qualifier("semanticHnswSearchStrategy") SemanticSearchStrategy semanticHnsw,
            HybridRrfSearchStrategy hybrid, HybridRerankStrategy rerank, NeuralRerankStrategy neuralRerank,
            LearnedTowerSearchStrategy learnedTower) {
        return Map.of(
                StrategyType.LEXICAL, lexical,
                StrategyType.SEMANTIC, semantic,
                StrategyType.SEMANTIC_HNSW, semanticHnsw,
                StrategyType.HYBRID, hybrid,
                StrategyType.RERANK, rerank,
                StrategyType.NEURAL_RERANK, neuralRerank,
                StrategyType.LEARNED_TOWER, learnedTower
        );
    }
}
