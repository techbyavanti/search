#!/usr/bin/env python3
"""Extracts full WANDS product.csv records for every product that appears
in the GitHub Pages demo's captured query results (docs/data/q*.json),
writing docs/data/product-details.json.

This is what powers the "why was this shown" per-result detail modal in
docs/index.html (the PRODUCT_DETAILS block) — after running this, re-embed
that block the same way docs/index.html's QUERY_DATA is re-embedded from
docs/data/q*.json following scripts/capture-demo-snapshots.sh.

Run after capture-demo-snapshots.sh, whenever the demo queries or their
top results change, so the two stay in sync.
"""

import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_CSV = ROOT_DIR / "dataset" / "product.csv"
DATA_DIR = ROOT_DIR / "docs" / "data"
OUT_FILE = DATA_DIR / "product-details.json"


def parse_features(raw):
    if not raw:
        return []
    out = []
    for part in raw.split("|"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, value = part.partition(":")
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        out.append({"key": key, "value": value})
    return out


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_ids():
    ids = set()
    for path in sorted(DATA_DIR.glob("q*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for strategy_result in data["resultsByStrategy"].values():
            for result in strategy_result["results"]:
                ids.add(result["productId"])
    return ids


def main():
    ids = collect_ids()
    products = {}
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["product_id"]
            if pid not in ids:
                continue
            products[pid] = {
                "productId": pid,
                "productName": row["product_name"] or None,
                "productClass": row["product_class"] or None,
                "categoryHierarchy": row["category hierarchy"] or None,
                "description": row["product_description"] or None,
                "features": parse_features(row["product_features"]),
                "averageRating": to_float(row["average_rating"]),
                "ratingCount": to_float(row["rating_count"]),
                "reviewCount": to_float(row["review_count"]),
            }

    missing = ids - products.keys()
    if missing:
        raise SystemExit(f"product.csv missing {len(missing)} referenced ids: {sorted(missing)}")

    OUT_FILE.write_text(
        json.dumps(products, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {len(products)} product records -> {OUT_FILE}")


if __name__ == "__main__":
    main()
