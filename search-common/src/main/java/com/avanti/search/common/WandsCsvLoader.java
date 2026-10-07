package com.avanti.search.common;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads WANDS product.csv / query.csv / label.csv. Despite the .csv extension,
 * the files are tab-delimited with unquoted fields (verified against the
 * upstream dataset — descriptions contain literal commas).
 */
public final class WandsCsvLoader {

    private static final CSVFormat WANDS_FORMAT = CSVFormat.Builder.create(CSVFormat.TDF)
            .setHeader()
            .setSkipHeaderRecord(true)
            .setQuote(null)
            .build();

    private WandsCsvLoader() {
    }

    public static WandsDataset load(Path datasetDir) {
        List<WandsProduct> products = loadProducts(datasetDir.resolve("product.csv"));
        List<WandsQuery> queries = loadQueries(datasetDir.resolve("query.csv"));
        List<WandsLabel> labels = loadLabels(datasetDir.resolve("label.csv"));

        Map<String, Map<String, RelevanceGrade>> judgmentsByQuery = new HashMap<>();
        for (WandsLabel label : labels) {
            judgmentsByQuery
                    .computeIfAbsent(label.queryId(), id -> new HashMap<>())
                    .put(label.productId(), label.grade());
        }

        return new WandsDataset(products, queries, labels, judgmentsByQuery);
    }

    private static List<WandsProduct> loadProducts(Path path) {
        List<WandsProduct> products = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
             CSVParser parser = WANDS_FORMAT.parse(reader)) {
            for (CSVRecord record : parser) {
                products.add(new WandsProduct(
                        record.get("product_id"),
                        record.get("product_name"),
                        blankToNull(record.get("product_class")),
                        blankToNull(record.get("category hierarchy")),
                        blankToNull(record.get("product_description")),
                        record.get("product_features"),
                        parseInt(record.get("rating_count")),
                        parseDouble(record.get("average_rating")),
                        parseInt(record.get("review_count"))
                ));
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load " + path, e);
        }
        return products;
    }

    private static List<WandsQuery> loadQueries(Path path) {
        List<WandsQuery> queries = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
             CSVParser parser = WANDS_FORMAT.parse(reader)) {
            for (CSVRecord record : parser) {
                queries.add(new WandsQuery(
                        record.get("query_id"),
                        record.get("query"),
                        blankToNull(record.get("query_class"))
                ));
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load " + path, e);
        }
        return queries;
    }

    private static List<WandsLabel> loadLabels(Path path) {
        List<WandsLabel> labels = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
             CSVParser parser = WANDS_FORMAT.parse(reader)) {
            for (CSVRecord record : parser) {
                labels.add(new WandsLabel(
                        record.get("query_id"),
                        record.get("product_id"),
                        RelevanceGrade.fromLabel(record.get("label"))
                ));
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load " + path, e);
        }
        return labels;
    }

    private static String blankToNull(String value) {
        return (value == null || value.isBlank()) ? null : value;
    }

    /** Rating fields are stored as floats (e.g. "15.0") even though they're conceptually counts. */
    private static Integer parseInt(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return (int) Double.parseDouble(value);
    }

    private static Double parseDouble(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Double.parseDouble(value);
    }
}
