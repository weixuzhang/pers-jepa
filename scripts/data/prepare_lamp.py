#!/usr/bin/env python
"""Prepare any LaMP task (2, 3, 4, 5, 7) as Pers-JEPA paired-view jsonl.

Input: the official LaMP `<split>_questions.json` and `<split>_outputs.json`.
Each LaMP entry is one query with that user's history as `profile`; LaMP has
no user id field, so `metadata.user_id` is a stable hash of the profile item
ids (identical histories -> same user; within the user-based splits each user
appears once, so per-user calibration must come from --expand-profile-pairs,
which turns profile items into extra (prompt, target) pairs for the
generation tasks where an item carries both fields: LaMP-4 text->title,
LaMP-5 abstract->title).

Task formats (profile item fields -> profile line; task prompt = LaMP input):
  2  movie tagging   {description, tag}      -> classification, exact match
  3  product rating  {text, score}           -> rating 1-5, MAE
  4  news headline   {text, title}           -> generation, ROUGE
  5  scholarly title {abstract, title}       -> generation, ROUGE
  7  tweet paraphrase {text}                 -> generation, ROUGE
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import write_jsonl


def clean(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "."
    return text


FORMATS = {
    "2": lambda it, c: f"Movie: {clean(it.get('description'), c)} -> tag: {clean(it.get('tag'))}",
    "3": lambda it, c: f"Review: {clean(it.get('text'), c)} -> score: {clean(it.get('score'))}",
    "4": lambda it, c: f"{clean(it.get('title'))} - {clean(it.get('text'), c)}",
    "5": lambda it, c: f"{clean(it.get('title'))} - {clean(it.get('abstract'), c)}",
    "7": lambda it, c: clean(it.get("text"), c),
}
HEADERS = {
    "2": "This user's previously tagged movies:",
    "3": "This user's previous product reviews and scores:",
    "4": "Previous news articles written by this author:",
    "5": "Previous papers by this author:",
    "7": "Previous tweets by this user:",
}
# profile items that can become extra calibration pairs (prompt -> target)
EXPANSIONS = {
    "2": ("description", "tag", "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: "),
    "3": ("text", "score", "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: "),
    "4": ("text", "title", "Generate a headline for the following article: "),
    "5": ("abstract", "title", "Generate a title for the following abstract of a paper: "),
}
# LaMP-7 profile items are bare tweets: calibration pairs need a neutral
# paraphrase as the *input* (see scripts/data/neutralize_lamp7_profiles.py).


def user_id_of(profile: list[dict]) -> str:
    ids = sorted(str(it.get("id", "")) for it in profile)
    return hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(FORMATS))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-profile-items", type=int, default=10)
    ap.add_argument("--profile-item-chars", type=int, default=300)
    ap.add_argument("--expand-profile-pairs", type=int, default=0,
                    help="LaMP-2/3/4/5: also emit up to N (prompt,target) pairs per user built from profile items (per-user calibration; LaMP-7 needs neutralize_lamp7_profiles.py)")
    ap.add_argument("--keep-raw-profile", action="store_true", help="store the raw profile items in metadata.raw_profile (needed for LaMP-7 neutralization)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    golds = {str(row["id"]): str(row["output"]) for row in json.loads(Path(args.outputs).read_text(encoding="utf-8"))["golds"]}
    rng = random.Random(args.seed)
    if args.max_examples and args.max_examples < len(questions):
        questions = rng.sample(questions, k=args.max_examples)

    fmt, header = FORMATS[args.task], HEADERS[args.task]
    records, n_expanded = [], 0
    for row in questions:
        rid = str(row["id"])
        profile_items = list(row.get("profile") or [])
        uid = user_id_of(profile_items)
        shown = profile_items[: args.max_profile_items]
        profile = header + "\n" + "\n".join(f"{i + 1}. {fmt(it, args.profile_item_chars)}" for i, it in enumerate(shown))
        records.append({
            "id": rid, "profile": profile, "prompt": clean(row.get("input")), "target": golds.get(rid, ""),
            "metadata": {"source": f"LaMP_{args.task}", "split": args.split, "user_id": uid,
                         "profile_count": len(profile_items), "used_profile_count": len(shown), "kind": "query",
                         **({"raw_profile": profile_items} if args.keep_raw_profile else {})},
        })
        if args.expand_profile_pairs and args.task in EXPANSIONS:
            src, tgt, instr = EXPANSIONS[args.task]
            rest = profile_items[args.max_profile_items:] or profile_items
            for j, it in enumerate(rest[: args.expand_profile_pairs]):
                if not it.get(src) or not it.get(tgt):
                    continue
                others = [x for x in profile_items if x is not it][: args.max_profile_items]
                prof = header + "\n" + "\n".join(f"{i + 1}. {fmt(x, args.profile_item_chars)}" for i, x in enumerate(others))
                records.append({
                    "id": f"{rid}__p{j}", "profile": prof, "prompt": instr + clean(it.get(src)), "target": clean(it.get(tgt)),
                    "metadata": {"source": f"LaMP_{args.task}", "split": args.split, "user_id": uid, "kind": "profile_pair"},
                })
                n_expanded += 1

    records.sort(key=lambda r: r["id"])
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} LaMP-{args.task} {args.split} records ({n_expanded} expanded profile pairs, "
          f"{len(set(r['metadata']['user_id'] for r in records))} users) to {args.output}")


if __name__ == "__main__":
    main()
