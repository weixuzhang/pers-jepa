#!/usr/bin/env python
"""Variant of prepare_amazon_reviews.py.

datasets>=3 dropped support for loading-script-based HF dataset repos, and
McAuley-Lab/Amazon-Reviews-2023 ships one, so `datasets.load_dataset(...,
trust_remote_code=True)` now fails. This script downloads the same
underlying raw jsonl files directly via huggingface_hub and reuses the
original build_user_groups/build_examples logic verbatim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import write_jsonl
from scripts.data.prepare_amazon_reviews import build_examples, build_user_groups, clean_text


def eligible_user_set(review_iter, *, min_reviews: int, min_text_chars: int, max_users: int | None, seed: int) -> set[str]:
    """Streaming first pass: users with >= min_reviews usable reviews, optionally a
    seeded random subset of them. Lets the 8 GB Movies_and_TV file be processed
    without materialising it (the second pass keeps only these users' rows)."""
    import random
    from collections import Counter

    counts: Counter[str] = Counter()
    for row in review_iter:
        user_id = row.get("user_id")
        if user_id and len(clean_text(row.get("text"))) >= min_text_chars:
            counts[str(user_id)] += 1
    users = sorted(u for u, c in counts.items() if c >= min_reviews)
    print(f"Prefilter: {len(users)} users with >= {min_reviews} usable reviews (of {len(counts)}).", flush=True)
    if max_users is not None and len(users) > max_users:
        random.Random(seed).shuffle(users)
        users = users[:max_users]
        print(f"Prefilter: keeping a seeded random subset of {len(users)} users.", flush=True)
    return set(users)


def open_text(path: str):
    """Plain or gzip-compressed jsonl (the McAuley mirror ships .jsonl.gz)."""
    if str(path).endswith(".gz"):
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_jsonl_from_hub(repo_id: str, filename: str, *, max_records: int | None, local_path: str | None = None):
    if local_path is None:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open_text(local_path) as handle:
        for idx, line in enumerate(handle):
            if max_records is not None and idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="McAuley-Lab/Amazon-Reviews-2023")
    parser.add_argument("--review-file", default="raw/review_categories/All_Beauty.jsonl")
    parser.add_argument("--meta-file", default="raw/meta_categories/meta_All_Beauty.jsonl")
    parser.add_argument("--min-reviews", type=int, default=20)
    parser.add_argument("--history-min", type=int, default=3)
    parser.add_argument("--history-max", type=int, default=6)
    parser.add_argument("--examples-per-user", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=1500)
    parser.add_argument("--max-raw-records", type=int, default=None)
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--history-review-chars", type=int, default=500)
    parser.add_argument("--target-review-chars", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-meta", action="store_true")
    parser.add_argument("--local-review-file", default=None, help="local .jsonl/.jsonl.gz instead of the HF download")
    parser.add_argument("--local-meta-file", default=None, help="local .jsonl/.jsonl.gz instead of the HF download")
    parser.add_argument("--prefilter-users", action="store_true",
                        help="two-pass streaming: keep only rows of users with >= --min-reviews usable reviews")
    parser.add_argument("--max-users", type=int, default=None, help="with --prefilter-users: seeded random cap on users")
    parser.add_argument("--require-item-title", action="store_true",
                        help="drop reviews whose item has no title in meta (Movies_and_TV: ~36%% of items, mostly "
                             "Prime Video) so prompts never name a bare ASIN; --min-reviews then applies to titled reviews")
    args = parser.parse_args()

    if args.history_min < 1 or args.history_max < args.history_min:
        raise ValueError("--history-max must be >= --history-min >= 1")
    if args.min_reviews <= args.history_min:
        raise ValueError("--min-reviews must be greater than --history-min")

    print("Downloading + scanning reviews...", flush=True)
    if args.prefilter_users:
        keep = eligible_user_set(
            iter_jsonl_from_hub(args.dataset, args.review_file, max_records=args.max_raw_records, local_path=args.local_review_file),
            min_reviews=args.min_reviews, min_text_chars=args.min_text_chars, max_users=args.max_users, seed=args.seed,
        )
        reviews = [
            row for row in iter_jsonl_from_hub(args.dataset, args.review_file, max_records=args.max_raw_records, local_path=args.local_review_file)
            if str(row.get("user_id")) in keep
        ]
    else:
        reviews = list(iter_jsonl_from_hub(args.dataset, args.review_file, max_records=args.max_raw_records, local_path=args.local_review_file))
    print(f"Loaded {len(reviews)} raw review records.", flush=True)

    item_titles: dict[str, str] = {}
    if not args.skip_meta:
        print("Downloading + scanning meta...", flush=True)
        for row in iter_jsonl_from_hub(args.dataset, args.meta_file, max_records=None, local_path=args.local_meta_file):
            parent_asin = row.get("parent_asin")
            title = clean_text(row.get("title"))
            if parent_asin and title:
                item_titles[str(parent_asin)] = title
        print(f"Loaded {len(item_titles)} item titles.", flush=True)

    if args.require_item_title:
        before = len(reviews)
        reviews = [r for r in reviews if str(r.get("parent_asin") or r.get("asin") or "") in item_titles]
        print(f"Kept {len(reviews)}/{before} reviews whose item has a title.", flush=True)

    groups = build_user_groups(
        reviews,
        min_text_chars=args.min_text_chars,
        max_raw_records=None,
    )
    examples = build_examples(
        groups,
        item_titles=item_titles,
        min_reviews=args.min_reviews,
        history_min=args.history_min,
        history_max=args.history_max,
        examples_per_user=args.examples_per_user,
        max_examples=args.max_examples,
        seed=args.seed,
        history_review_chars=args.history_review_chars,
        target_review_chars=args.target_review_chars,
    )
    write_jsonl(args.output, examples)

    eligible_users = sum(1 for rows in groups.values() if len(rows) >= args.min_reviews)
    print(
        f"Wrote {len(examples)} examples to {args.output} "
        f"from {eligible_users} users with >= {args.min_reviews} reviews "
        f"({len(groups)} users after text filtering)."
    )


if __name__ == "__main__":
    main()
