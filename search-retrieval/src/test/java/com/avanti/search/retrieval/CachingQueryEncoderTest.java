package com.avanti.search.retrieval;

import com.avanti.search.inference.QueryEncoder;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CachingQueryEncoderTest {

    private final List<String> calls = new ArrayList<>();
    private final QueryEncoder model = query -> {
        calls.add(query);
        return new float[]{query.length(), 1.0f};
    };

    @Test
    void encodesARepeatedQueryOnce() {
        CachingQueryEncoder encoder = new CachingQueryEncoder(model, 10);

        float[] first = encoder.embedQuery("oak table");
        float[] second = encoder.embedQuery("oak table");

        assertThat(calls).containsExactly("oak table");
        assertThat(second).containsExactly(first);
    }

    @Test
    void queriesDifferingOnlyInCaseAndWhitespaceShareAnEntry() {
        CachingQueryEncoder encoder = new CachingQueryEncoder(model, 10);

        encoder.embedQuery("Oak  Table");
        encoder.embedQuery("  oak table ");

        assertThat(calls).hasSize(1);
        assertThat(encoder.size()).isEqualTo(1);
    }

    @Test
    void callersCannotChangeTheCachedVector() {
        CachingQueryEncoder encoder = new CachingQueryEncoder(model, 10);

        encoder.embedQuery("lamp")[0] = 99f;

        assertThat(encoder.embedQuery("lamp")[0]).isEqualTo(4f);
    }

    @Test
    void evictsTheLeastRecentlyUsedQuery() {
        CachingQueryEncoder encoder = new CachingQueryEncoder(model, 2);

        encoder.embedQuery("a");
        encoder.embedQuery("b");
        encoder.embedQuery("a");
        encoder.embedQuery("c");   // evicts "b"
        encoder.embedQuery("a");
        encoder.embedQuery("b");

        assertThat(calls).containsExactly("a", "b", "c", "b");
    }

    @Test
    void rejectsAZeroSize() {
        assertThatThrownBy(() -> new CachingQueryEncoder(model, 0)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void normalizeTrimsCollapsesWhitespaceAndLowercases() {
        assertThat(QueryNormalizer.normalize("  Modern\tOAK   table ")).isEqualTo("modern oak table");
    }
}
