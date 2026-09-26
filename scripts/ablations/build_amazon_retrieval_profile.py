#!/usr/bin/env python
"""Build a retrieval-based profile-prompting baseline for
Amazon All Beauty. Instead of the original random sample of history_min..max
previous reviews, select the top-k previous reviews most similar (TF-IDF
cosine) to the current review-request query, from the user's FULL history
pool (all reviews strictly before the target timestamp).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.data.prepare_amazon_reviews import clean_text, format_review


def iter_jsonl_from_hub(repo_id: str, filename: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="McAuley-Lab/Amazon-Reviews-2023")
    parser.add_argument("--review-file", default="raw/review_categories/All_Beauty.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--history-review-chars", type=int, default=500)
    args = parser.parse_args()

    with open(args.heldout_jsonl, "r", encoding="utf-8") as handle:
        heldout = [json.loads(line) for line in handle]

    target_user_ids = {rec["metadata"]["user_id"] for rec in heldout}
    print(f"Need history for {len(target_user_ids)} users", flush=True)

    by_user: dict[str, list[dict]] = {uid: [] for uid in target_user_ids}
    for row in iter_jsonl_from_hub(args.dataset, args.review_file):
        uid = row.get("user_id")
        if uid in by_user:
            by_user[uid].append(row)
    for uid in by_user:
        by_user[uid].sort(key=lambda r: int(r.get("timestamp") or 0))

    output_records = []
    for rec in heldout:
        uid = rec["metadata"]["user_id"]
        target_ts = int(rec["metadata"]["target_timestamp"])
        candidates = [r for r in by_user[uid] if int(r.get("timestamp") or 0) < target_ts]
        candidates = [r for r in candidates if len(clean_text(r.get("text"))) >= 40]
        if len(candidates) < 2:
            output_records.append(rec)
            continue

        query = rec["prompt"]
        docs = [format_review(r, max_chars=args.history_review_chars) for r in candidates]
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(docs + [query])
        except ValueError:
            output_records.append(rec)
            continue
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        top_k = min(args.top_k, len(docs))
        top_idx = sims.argsort()[::-1][:top_k]

        history_lines = [f"{i + 1}. {docs[j]}" for i, j in enumerate(top_idx)]
        retrieval_profile = "Previous reviews by this user (retrieved by relevance):\n" + "\n".join(history_lines)

        new_rec = dict(rec)
        new_rec["profile"] = retrieval_profile
        new_rec["metadata"] = dict(rec["metadata"])
        new_rec["metadata"]["profile_type"] = "retrieval_top_k"
        new_rec["metadata"]["retrieval_k"] = top_k
        new_rec["metadata"]["retrieval_candidate_pool_size"] = len(candidates)
        output_records.append(new_rec)

    with open(args.output, "w", encoding="utf-8") as handle:
        for rec in output_records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pool_sizes = [r["metadata"].get("retrieval_candidate_pool_size", 0) for r in output_records]
    print(f"Wrote {len(output_records)} records. Median candidate pool size: {sorted(pool_sizes)[len(pool_sizes)//2]}")


if __name__ == "__main__":
    main()
