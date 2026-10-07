package com.avanti.search.api;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.inference.NeuralRerankerService;
import com.avanti.search.inference.RerankerService;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.HybridRerankStrategy;
import com.avanti.search.retrieval.HybridRrfSearchStrategy;
import com.avanti.search.retrieval.LearnedTowerSearchStrategy;
import com.avanti.search.retrieval.LexicalSearchStrategy;
import com.avanti.search.retrieval.NeuralRerankStrategy;
import com.avanti.search.retrieval.ProductFeatureCache;
import com.avanti.search.retrieval.ProductLookup;
import com.avanti.search.retrieval.SearchStrategy;
import com.avanti.search.retrieval.SemanticSearchStrategy;
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
        return new RerankerService(properties.models().rerankerDir());
    }

    @Bean
    public NeuralRerankerService neuralRerankerService(SearchProperties properties) throws IOException {
        return new NeuralRerankerService(properties.models().neuralRerankerDir());
    }

    @Bean
    public ProductFeatureCache productFeatureCache(ElasticsearchClient client, SearchProperties properties) {
        return new ProductFeatureCache(ids -> ProductLookup.fetchByIds(client, ids), properties.featureCache().maxSize());
    }

    @Bean
    public LexicalSearchStrategy lexicalSearchStrategy(ElasticsearchClient client) {
        return new LexicalSearchStrategy(client);
    }

    @Bean
    public SemanticSearchStrategy semanticSearchStrategy(ElasticsearchClient client, EmbeddingService embeddingService) {
        return new SemanticSearchStrategy(client, embeddingService);
    }

    @Bean
    public HybridRrfSearchStrategy hybridRrfSearchStrategy(
            LexicalSearchStrategy lexical, SemanticSearchStrategy semantic, SearchProperties properties) {
        return new HybridRrfSearchStrategy(lexical, semantic, properties.hybrid().candidatePoolSize(), properties.hybrid().rrfK());
    }

    @Bean
    public HybridRerankStrategy hybridRerankStrategy(
            HybridRrfSearchStrategy hybrid, ProductFeatureCache featureCache, RerankerService rerankerService, SearchProperties properties) {
        return new HybridRerankStrategy(hybrid, featureCache, rerankerService, properties.hybrid().candidatePoolSize());
    }

    @Bean
    public NeuralRerankStrategy neuralRerankStrategy(
            HybridRrfSearchStrategy hybrid, ElasticsearchClient client, EmbeddingService embeddingService,
            NeuralRerankerService neuralRerankerService, ProductFeatureCache featureCache, SearchProperties properties) {
        return new NeuralRerankStrategy(hybrid, client, embeddingService, neuralRerankerService, featureCache, properties.hybrid().candidatePoolSize());
    }

    @Bean
    public LearnedTowerSearchStrategy learnedTowerSearchStrategy(
            ElasticsearchClient client, EmbeddingService learnedTowerEmbeddingService) {
        return new LearnedTowerSearchStrategy(client, learnedTowerEmbeddingService);
    }

    @Bean
    public Map<StrategyType, SearchStrategy> strategiesByType(
            LexicalSearchStrategy lexical, SemanticSearchStrategy semantic,
            HybridRrfSearchStrategy hybrid, HybridRerankStrategy rerank, NeuralRerankStrategy neuralRerank,
            LearnedTowerSearchStrategy learnedTower) {
        return Map.of(
                StrategyType.LEXICAL, lexical,
                StrategyType.SEMANTIC, semantic,
                StrategyType.HYBRID, hybrid,
                StrategyType.RERANK, rerank,
                StrategyType.NEURAL_RERANK, neuralRerank,
                StrategyType.LEARNED_TOWER, learnedTower
        );
    }
}
