package com.avanti.search.eval;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class EvalCliTest {

    @Test
    void fileSlugMatchesTheCommittedResultFileNames() {
        assertThat(EvalCli.fileSlug("Hybrid (RRF, k=40)")).isEqualTo("hybrid-rrf-k-40");
        assertThat(EvalCli.fileSlug("Semantic (bge-small-en-v1.5)")).isEqualTo("semantic-bge-small-en-v1-5");
        assertThat(EvalCli.fileSlug("Hybrid + Cross-Encoder Rerank")).isEqualTo("hybrid-cross-encoder-rerank");
    }

    @Test
    void fileSlugDropsLeadingAndTrailingSeparators() {
        assertThat(EvalCli.fileSlug("(Neural Rerank)")).isEqualTo("neural-rerank");
    }
}
