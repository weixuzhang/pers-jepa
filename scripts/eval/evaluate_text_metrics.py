#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def f1_score(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for idx_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[idx_b - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[idx_b]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_candidate(records: list[dict], key: str) -> dict[str, float]:
    totals = {
        "exact_match": 0.0,
        "token_f1": 0.0,
        "rouge1_f1": 0.0,
        "rouge_l_f1": 0.0,
        "pred_tokens": 0.0,
        "target_tokens": 0.0,
    }
    for record in records:
        pred = str(record.get(key, ""))
        target = str(record.get("target", ""))
        pred_tokens = tokenize(pred)
        target_tokens = tokenize(target)
        totals["exact_match"] += float(normalize(pred) == normalize(target))
        totals["token_f1"] += f1_score(pred_tokens, target_tokens)
        totals["rouge1_f1"] += f1_score(pred_tokens, target_tokens)
        totals["rouge_l_f1"] += rouge_l_f1(pred_tokens, target_tokens)
        totals["pred_tokens"] += len(pred_tokens)
        totals["target_tokens"] += len(target_tokens)
    n = max(len(records), 1)
    return {key: value / n for key, value in totals.items()} | {"n_examples": len(records)}


def write_lamp_predictions(records: list[dict], key: str, path: Path) -> None:
    payload = {
        "task": "LaMP_5",
        "golds": [
            {"id": str(record["id"]), "output": str(record.get(key, ""))}
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate title/text generation records against target text.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate-keys",
        nargs="+",
        default=["raw_generic_output", "generic_pred_output", "jepa_steered_output"],
    )
    parser.add_argument(
        "--lamp-prediction-dir",
        default=None,
        help="Optional directory for LaMP-format prediction JSON files.",
    )
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = {
        key: evaluate_candidate(records, key)
        for key in args.candidate_keys
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.lamp_prediction_dir:
        pred_dir = Path(args.lamp_prediction_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        for key in args.candidate_keys:
            write_lamp_predictions(records, key, pred_dir / f"{key}.json")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
