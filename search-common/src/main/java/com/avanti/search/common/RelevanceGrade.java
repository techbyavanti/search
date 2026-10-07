package com.avanti.search.common;

import java.util.Locale;

/** WANDS relevance judgment grades, ordered so {@code grade()} feeds nDCG directly. */
public enum RelevanceGrade {
    IRRELEVANT(0),
    PARTIAL(1),
    EXACT(2);

    private final int grade;

    RelevanceGrade(int grade) {
        this.grade = grade;
    }

    public int grade() {
        return grade;
    }

    public boolean isRelevant() {
        return grade >= 1;
    }

    public static RelevanceGrade fromLabel(String label) {
        return switch (label.trim().toLowerCase(Locale.ROOT)) {
            case "exact" -> EXACT;
            case "partial" -> PARTIAL;
            case "irrelevant" -> IRRELEVANT;
            default -> throw new IllegalArgumentException("Unknown WANDS relevance label: " + label);
        };
    }
}
