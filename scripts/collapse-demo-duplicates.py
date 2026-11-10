#!/usr/bin/env python3
"""Collapses same-named listings in captured GitHub Pages demo results.

The WANDS catalog holds separate listings that share one product name, and
each is a distinct product id, so the ranking strategies can return several
of them. A shopper reads those as one result repeated. Each strategy keeps
only the best-ranked listing per name, in rank order, cut to --top-k.

This applies only to the demo snapshots. The strategies and the offline
evaluation score every listing.

Run by scripts/capture-demo-snapshots.sh on results fetched deeper than
--top-k, so that collapsing still leaves --top-k results per strategy.
"""

import argparse
import json
from pathlib import Path


def name_key(result):
    name = result["productName"]
    if name is None:
        return ("id", result["productId"])
    return ("name", " ".join(name.casefold().split()))


def collapse(results, top_k):
    seen = set()
    distinct = []
    for result in results:
        key = name_key(result)
        if key not in seen:
            seen.add(key)
            distinct.append(result)
    return distinct[:top_k]


def collapse_file(path, top_k):
    data = json.loads(path.read_text(encoding="utf-8"))
    for strategy, strategy_result in data["resultsByStrategy"].items():
        collapsed = collapse(strategy_result["results"], top_k)
        if len(collapsed) < top_k:
            raise SystemExit(
                f"{path.name}: {strategy} has {len(collapsed)} distinct names, "
                f"needs {top_k}; raise FETCH_K in capture-demo-snapshots.sh"
            )
        strategy_result["results"] = collapsed
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.files:
        collapse_file(path, args.top_k)


if __name__ == "__main__":
    main()
