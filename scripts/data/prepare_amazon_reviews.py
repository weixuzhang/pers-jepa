#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import write_jsonl


def clean_text(value: Any, *, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "."
    return text


def format_review(review: dict, *, max_chars: int) -> str:
    title = clean_text(review.get("title"))
    body = clean_text(review.get("text"), max_chars=max_chars)
    if title and body:
        return f"{title} - {body}"
    return title or body


def load_dataset_split(dataset_name: str, config_name: str, split: str):
    from datasets import load_dataset

    return load_dataset(
        dataset_name,
        config_name,
        split=split,
        trust_remote_code=True,
    )


def load_item_titles(
    *,
    dataset_name: str,
    meta_config: str,
    split: str,
    skip_meta: bool,
) -> dict[str, str]:
    if skip_meta:
        return {}
    metadata = load_dataset_split(dataset_name, meta_config, split)
    titles: dict[str, str] = {}
    for row in metadata:
        parent_asin = row.get("parent_asin")
        title = clean_text(row.get("title"))
        if parent_asin and title:
            titles[str(parent_asin)] = title
    return titles


def build_user_groups(reviews, *, min_text_chars: int, max_raw_records: int | None):
    groups: dict[str, list[dict]] = defaultdict(list)
    for idx, row in enumerate(reviews):
        if max_raw_records is not None and idx >= max_raw_records:
            break
        user_id = row.get("user_id")
        body = clean_text(row.get("text"))
        if not user_id or len(body) < min_text_chars:
            continue
        groups[str(user_id)].append(dict(row))
    return groups


def build_examples(
    groups: dict[str, list[dict]],
    *,
    item_titles: dict[str, str],
    min_reviews: int,
    history_min: int,
    history_max: int,
    examples_per_user: int,
    max_examples: int | None,
    seed: int,
    history_review_chars: int,
    target_review_chars: int,
) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    eligible_users = [
        user_id for user_id, user_reviews in groups.items()
        if len(user_reviews) >= min_reviews
    ]
    rng.shuffle(eligible_users)

    for user_id in eligible_users:
        user_reviews = sorted(
            groups[user_id],
            key=lambda row: int(row.get("timestamp") or 0),
        )
        candidate_indices = list(range(max(history_min, 1), len(user_reviews)))
        rng.shuffle(candidate_indices)
        made_for_user = 0

        for target_idx in candidate_indices:
            previous = user_reviews[:target_idx]
            if len(previous) < history_min:
                continue
            history_count = rng.randint(history_min, min(history_max, len(previous)))
            history = rng.sample(previous, k=history_count)
            history_lines = [
                f"{idx + 1}. {format_review(review, max_chars=history_review_chars)}"
                for idx, review in enumerate(history)
            ]
            target_review = user_reviews[target_idx]
            parent_asin = str(target_review.get("parent_asin") or target_review.get("asin") or "")
            item_name = item_titles.get(parent_asin) or parent_asin or "this product"
            rating = target_review.get("rating", "")
            rating_text = f"{float(rating):g}" if isinstance(rating, (int, float)) else str(rating or "")
            prompt = (
                f"Write a {rating_text}-star review for a product named "
                f"\"{clean_text(item_name, max_chars=160)}\"."
            )
            profile = "Previous reviews by this user:\n" + "\n".join(history_lines)
            target = format_review(target_review, max_chars=target_review_chars)
            example_id = f"{user_id}-{target_review.get('timestamp', target_idx)}-{parent_asin}"
            examples.append({
                "id": example_id,
                "profile": profile,
                "prompt": prompt,
                "target": target,
                "metadata": {
                    "source": "McAuley-Lab/Amazon-Reviews-2023",
                    "user_id": user_id,
                    "parent_asin": parent_asin,
                    "rating": rating,
                    "target_review_title": clean_text(target_review.get("title")),
                    "target_timestamp": target_review.get("timestamp"),
                    "history_count": history_count,
                    "user_review_count": len(user_reviews),
                },
            })
            made_for_user += 1
            if max_examples is not None and len(examples) >= max_examples:
                return examples
            if made_for_user >= examples_per_user:
                break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Amazon Reviews 2023 user-history pairs for Pers-JEPA."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="McAuley-Lab/Amazon-Reviews-2023")
    parser.add_argument("--review-config", default="raw_review_All_Beauty")
    parser.add_argument("--meta-config", default="raw_meta_All_Beauty")
    parser.add_argument("--split", default="full")
    parser.add_argument("--min-reviews", type=int, default=50)
    parser.add_argument("--history-min", type=int, default=5)
    parser.add_argument("--history-max", type=int, default=10)
    parser.add_argument("--examples-per-user", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-raw-records", type=int, default=None)
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--history-review-chars", type=int, default=500)
    parser.add_argument("--target-review-chars", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-meta", action="store_true")
    args = parser.parse_args()

    if args.history_min < 1 or args.history_max < args.history_min:
        raise ValueError("--history-max must be >= --history-min >= 1")
    if args.min_reviews <= args.history_min:
        raise ValueError("--min-reviews must be greater than --history-min")

    reviews = load_dataset_split(args.dataset, args.review_config, args.split)
    item_titles = load_item_titles(
        dataset_name=args.dataset,
        meta_config=args.meta_config,
        split=args.split,
        skip_meta=args.skip_meta,
    )
    groups = build_user_groups(
        reviews,
        min_text_chars=args.min_text_chars,
        max_raw_records=args.max_raw_records,
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
