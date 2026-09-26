#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def persona_of(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    nested_metadata = metadata.get("metadata") if isinstance(metadata, dict) else None
    if not isinstance(nested_metadata, dict):
        nested_metadata = {}
    persona = (
        metadata.get("persona")
        or metadata.get("persona_id")
        or nested_metadata.get("persona")
        or nested_metadata.get("persona_id")
    )
    if persona:
        return str(persona)
    persona = record.get("persona_id")
    if persona:
        return str(persona)
    profile = str(record.get("profile", ""))
    for line in profile.splitlines():
        if line.lower().startswith("persona:"):
            return line.split(":", 1)[1].strip()
    if ":" in profile and profile.startswith("The "):
        return profile.split(":", 1)[0].strip()
    return ""


def build_logistic_regression(**kwargs: Any) -> LogisticRegression:
    try:
        return LogisticRegression(**kwargs)
    except TypeError:
        kwargs.pop("multi_class", None)
        return LogisticRegression(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SynPer generations with a stronger persona classifier.")
    parser.add_argument("--train", default="data/synper/clean/train_10000.jsonl")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confusion-output", required=True)
    parser.add_argument(
        "--candidate-keys",
        nargs="+",
        required=True,
    )
    args = parser.parse_args()

    train_records = read_jsonl(Path(args.train))
    train_texts = [str(row.get("target", "")) for row in train_records]
    train_labels = [persona_of(row) for row in train_records]

    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        build_logistic_regression(max_iter=2000, multi_class="auto", n_jobs=1),
    )
    clf.fit(train_texts, train_labels)

    generation_records = read_jsonl(Path(args.generations))
    labels = [persona_of(row) for row in generation_records]
    summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    label_order = sorted(set(train_labels))
    for key in args.candidate_keys:
        texts = [str(row.get(key, "")) for row in generation_records]
        preds = clf.predict(texts)
        probs = clf.predict_proba(texts)
        pred_scores = probs.max(axis=1)
        accuracy = sum(int(y == p) for y, p in zip(labels, preds)) / max(len(labels), 1)
        summary_rows.append(
            {
                "dataset": "SynPer",
                "candidate_key": key,
                "persona_accuracy": accuracy,
                "macro_f1": f1_score(labels, preds, average="macro"),
                "avg_confidence": float(pred_scores.mean()) if len(pred_scores) else 0.0,
                "n_examples": len(labels),
                "classifier": "char_tfidf_logreg_train10k_targets",
            }
        )
        for gold in label_order:
            for pred in label_order:
                count = sum(1 for y, p in zip(labels, preds) if y == gold and p == pred)
                if count:
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
        fieldnames = list(confusion_rows[0].keys()) if confusion_rows else [
            "candidate_key",
            "gold_persona",
            "predicted_persona",
            "count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(confusion_rows)

    print(json.dumps({"output": str(output_path), "confusion": str(confusion_path), "rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
