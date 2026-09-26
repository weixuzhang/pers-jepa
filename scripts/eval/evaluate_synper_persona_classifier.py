#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def dotted_get(record: dict[str, Any], field: str, default: Any = None) -> Any:
    value: Any = record
    for part in field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, dict) and isinstance(value.get("metadata"), dict) and part in value["metadata"]:
            value = value["metadata"][part]
        else:
            return default
    return value


def persona_of(record: dict[str, Any]) -> str:
    value = dotted_get(record, "metadata.persona")
    if value:
        return str(value)
    value = dotted_get(record, "metadata.metadata.persona")
    if value:
        return str(value)
    group_id = record.get("group_id")
    if group_id:
        return str(group_id)
    persona_id = record.get("persona_id")
    if persona_id:
        return str(persona_id)
    profile = str(record.get("profile", ""))
    for line in profile.splitlines():
        if line.lower().startswith("persona:"):
            return line.split(":", 1)[1].strip()
    return ""


class NaiveBayesPersonaClassifier:
    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.labels: list[str] = []
        self.label_token_counts: dict[str, Counter[str]] = {}
        self.label_totals: dict[str, int] = {}
        self.label_priors: dict[str, float] = {}
        self.vocab: set[str] = set()

    def fit(self, rows: list[tuple[str, str]]) -> None:
        by_label: dict[str, Counter[str]] = defaultdict(Counter)
        label_docs = Counter()
        for label, text in rows:
            if not label:
                continue
            tokens = tokenize(text)
            by_label[label].update(tokens)
            self.vocab.update(tokens)
            label_docs[label] += 1
        self.labels = sorted(by_label)
        total_docs = sum(label_docs.values())
        self.label_token_counts = dict(by_label)
        self.label_totals = {label: sum(counter.values()) for label, counter in by_label.items()}
        self.label_priors = {
            label: math.log(label_docs[label] / total_docs)
            for label in self.labels
        }

    def score(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        vocab_size = max(len(self.vocab), 1)
        scores = {}
        for label in self.labels:
            counter = self.label_token_counts[label]
            denom = self.label_totals[label] + self.alpha * vocab_size
            value = self.label_priors[label]
            for token in tokens:
                value += math.log((counter[token] + self.alpha) / denom)
            # Normalize by length so very short/long generations are not judged
            # only by sequence length.
            scores[label] = value / max(len(tokens), 1)
        return scores

    def predict(self, text: str) -> tuple[str, float]:
        scores = self.score(text)
        if not scores:
            return "", 0.0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
        return ranked[0][0], margin


def macro_f1(labels: list[str], preds: list[str]) -> float:
    values = sorted(set(labels) | set(preds))
    scores = []
    for label in values:
        tp = sum(1 for y, p in zip(labels, preds) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, preds) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, preds) if y == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SynPer generations with a simple persona text classifier.")
    parser.add_argument("--train", default="data/synper/train_300.jsonl")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", default="tables/synper_persona_classifier_accuracy.csv")
    parser.add_argument("--confusion-output", default="tables/synper_persona_classifier_confusion.csv")
    parser.add_argument(
        "--candidate-keys",
        nargs="+",
        default=[
            "raw_generic_output",
            "generic_pred_output",
            "personalized_prompt_output",
            "global_answer_token_mean_delta_output",
            "group_answer_token_mean_delta_output",
        ],
    )
    args = parser.parse_args()

    train_records = read_jsonl(Path(args.train))
    train_rows = [(persona_of(record), str(record.get("target", ""))) for record in train_records]
    clf = NaiveBayesPersonaClassifier(alpha=1.0)
    clf.fit(train_rows)

    generation_records = read_jsonl(Path(args.generations))
    summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for key in args.candidate_keys:
        golds: list[str] = []
        preds: list[str] = []
        margins: list[float] = []
        confusion = Counter()
        for record in generation_records:
            gold = persona_of(record)
            pred, margin = clf.predict(str(record.get(key, "")))
            golds.append(gold)
            preds.append(pred)
            margins.append(margin)
            confusion[(gold, pred)] += 1
        accuracy = sum(1 for gold, pred in zip(golds, preds) if gold == pred) / max(len(golds), 1)
        summary_rows.append(
            {
                "dataset": "SynPer",
                "candidate_key": key,
                "persona_accuracy": accuracy,
                "macro_f1": macro_f1(golds, preds),
                "avg_margin": sum(margins) / max(len(margins), 1),
                "n_examples": len(golds),
                "classifier": "word_naive_bayes_train300_targets",
            }
        )
        for (gold, pred), count in sorted(confusion.items()):
            confusion_rows.append(
                {
                    "candidate_key": key,
                    "gold_persona": gold,
                    "predicted_persona": pred,
                    "count": count,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    confusion_path = Path(args.confusion_output)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(confusion_rows)

    print(json.dumps({"output": str(output_path), "confusion": str(confusion_path), "rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
