package com.avanti.search.retrieval;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class RrfFusionServiceTest {

    @Test
    void documentRankedFirstInBothListsScoresHighest() {
        List<ScoredResult> lexical = List.of(
                new ScoredResult("a", 10.0),
                new ScoredResult("b", 8.0));
        List<ScoredResult> semantic = List.of(
                new ScoredResult("a", 0.9),
                new ScoredResult("b", 0.7));

        List<ScoredResult> fused = RrfFusionService.fuse(List.of(lexical, semantic), 60);

        assertThat(fused).extracting(ScoredResult::productId).containsExactly("a", "b");
    }

    @Test
    void computesExactReciprocalRankScores() {
        List<ScoredResult> lexical = List.of(
                new ScoredResult("a", 10.0),
                new ScoredResult("b", 8.0));
        List<ScoredResult> semantic = List.of(
                new ScoredResult("b", 0.9),
                new ScoredResult("a", 0.7));

        List<ScoredResult> fused = RrfFusionService.fuse(List.of(lexical, semantic), 60);

        double expectedScoreA = 1.0 / (60 + 1) + 1.0 / (60 + 2);
        double expectedScoreB = 1.0 / (60 + 2) + 1.0 / (60 + 1);

        assertThat(fused).hasSize(2);
        assertThat(expectedScoreA).isEqualTo(expectedScoreB);
        for (ScoredResult result : fused) {
            double expected = result.productId().equals("a") ? expectedScoreA : expectedScoreB;
            assertThat(result.score()).isEqualTo(expected, org.assertj.core.data.Offset.offset(1e-9));
        }
    }

    @Test
    void documentPresentInOnlyOneListStillContributesItsPartialScore() {
        List<ScoredResult> lexical = List.of(new ScoredResult("lexicalOnly", 10.0));
        List<ScoredResult> semantic = List.of(new ScoredResult("semanticOnly", 0.9));

        List<ScoredResult> fused = RrfFusionService.fuse(List.of(lexical, semantic), 60);

        assertThat(fused).extracting(ScoredResult::productId).containsExactlyInAnyOrder("lexicalOnly", "semanticOnly");
        assertThat(fused).allSatisfy(r -> assertThat(r.score()).isEqualTo(1.0 / 61, org.assertj.core.data.Offset.offset(1e-9)));
    }

    @Test
    void documentAppearingInBothListsOutranksOneAppearingInOnlyOne() {
        List<ScoredResult> lexical = List.of(
                new ScoredResult("both", 10.0),
                new ScoredResult("lexicalOnly", 8.0));
        List<ScoredResult> semantic = List.of(new ScoredResult("both", 0.5));

        List<ScoredResult> fused = RrfFusionService.fuse(List.of(lexical, semantic), 60);

        assertThat(fused.get(0).productId()).isEqualTo("both");
    }

    @Test
    void smallerKAmplifiesTopRankDifferences() {
        List<ScoredResult> singleList = List.of(
                new ScoredResult("first", 10.0),
                new ScoredResult("second", 9.0));

        List<ScoredResult> fusedSmallK = RrfFusionService.fuse(List.of(singleList), 1);
        List<ScoredResult> fusedLargeK = RrfFusionService.fuse(List.of(singleList), 1000);

        double smallKGap = fusedSmallK.get(0).score() - fusedSmallK.get(1).score();
        double largeKGap = fusedLargeK.get(0).score() - fusedLargeK.get(1).score();

        assertThat(smallKGap).isGreaterThan(largeKGap);
    }
}
