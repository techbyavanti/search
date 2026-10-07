package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;

public final class ElasticsearchClients {

    private ElasticsearchClients() {
    }

    public static ElasticsearchClient create() {
        return create(SearchConstants.DEFAULT_ELASTICSEARCH_HOST);
    }

    public static ElasticsearchClient create(String host) {
        return ElasticsearchClient.of(b -> b.host(host));
    }
}
