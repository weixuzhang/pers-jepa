#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def label_of(record: dict[str, Any], field: str) -> str:
    value = dotted_get(record, field)
    return str(value).strip() if value not in (None, "") else "__missing__"


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


def build_logistic_regression(LogisticRegression: Any, **kwargs: Any) -> Any:
    try:
        return LogisticRegression(**kwargs)
    except TypeError:
        kwargs.pop("multi_class", None)
        return LogisticRegression(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated text with a strong TF-IDF + logistic-regression group-style classifier.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--generations", required=True, nargs="+",
                        help="one generation jsonl (with --output/--confusion-output) or several: the classifier is "
                             "trained ONCE and each file gets <file>_userstyle.csv / _userstyle_confusion.csv next to it")
    parser.add_argument("--label-field", required=True, help="Dotted field path used as the class label, e.g. metadata.user_id")
    parser.add_argument("--output", default=None, help="required for a single --generations file")
    parser.add_argument("--confusion-output", default=None)
    parser.add_argument("--output-suffix", default="_userstyle.csv", help="multi-file mode: suffix replacing .jsonl")
    parser.add_argument("--max-features", type=int, default=40000,
                        help="TF-IDF vocabulary cap for EACH of the word and char vectorizers (default 40000 = the "
                             "Amazon/SynPer protocol; lower it for thousands of classes: scipy's L-BFGS-B segfaults "
                             "when n_classes x n_features exceeds ~1e8)")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--train-text-key", default="target")
    parser.add_argument(
        "--candidate-keys",
        nargs="+",
        default=[
            "raw_generic_output",
            "jepa_steered_output",
            "generic_pred_output",
            "personalized_prompt_output",
            "global_answer_token_mean_delta_output",
            "group_answer_token_mean_delta_output",
            "shuffled_group_answer_token_mean_delta_output",
            "dense_profile_vector_profile_only_output",
            "dense_profile_vector_task_conditioned_output",
        ],
    )
    args = parser.parse_args()
    if len(args.generations) == 1 and not args.output:
        parser.error("--output and --confusion-output are required for a single --generations file")
    if len(args.generations) > 1 and args.output:
        parser.error("--output cannot be used with several --generations files (outputs are derived per file)")

    try:
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn and scipy to use the strong style classifier.") from exc

    train_records = read_jsonl(Path(args.train))
    train_labels = [label_of(record, args.label_field) for record in train_records]
    train_texts = [str(record.get(args.train_text_key, "")) for record in train_records]
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=args.max_features, lowercase=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=args.max_features, lowercase=True)
    x_train = hstack([word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts)]).tocsr()
    clf = build_logistic_regression(
        LogisticRegression,
        max_iter=3000,
        multi_class="auto",
        C=4.0,
        n_jobs=1,
    )
    clf.fit(x_train, train_labels)

    for gen_path in args.generations:
      if args.output:
        out_path, conf_path = Path(args.output), Path(args.confusion_output)
      else:
        stem = str(gen_path)[:-len(".jsonl")] if str(gen_path).endswith(".jsonl") else str(gen_path)
        out_path = Path(stem + args.output_suffix)
        conf_path = Path(stem + args.output_suffix.replace(".csv", "_confusion.csv"))
      score_file(args, Path(gen_path), out_path, conf_path, word_vec, char_vec, clf, hstack)


def score_file(args, generations: Path, output_path: Path, confusion_path: Path, word_vec, char_vec, clf, hstack) -> None:
    generation_records = read_jsonl(generations)
    output_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for key in args.candidate_keys:
        labels: list[str] = []
        preds: list[str] = []
        margins: list[float] = []
        confusion: dict[tuple[str, str], int] = {}
        texts = [str(record.get(key, "")) for record in generation_records]
        x_eval = hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()
        prob = clf.predict_proba(x_eval)
        pred_labels = clf.classes_[prob.argmax(axis=1)]
        top2 = prob.argsort(axis=1)[:, -2:]
        for record, pred_label, row_prob, row_top2 in zip(generation_records, pred_labels, prob, top2):
            gold = label_of(record, args.label_field)
            pred = str(pred_label)
            labels.append(gold)
            preds.append(pred)
            sorted_top2 = sorted(row_top2.tolist(), key=lambda idx: row_prob[idx], reverse=True)
            margin = float(row_prob[sorted_top2[0]] - row_prob[sorted_top2[1]]) if len(sorted_top2) > 1 else 0.0
            margins.append(margin)
            confusion[(gold, pred)] = confusion.get((gold, pred), 0) + 1
        accuracy = sum(1 for y, p in zip(labels, preds) if y == p) / max(len(labels), 1)
        output_rows.append(
            {
                "dataset": args.dataset,
                "label_field": args.label_field,
                "candidate_key": key,
                "style_accuracy": accuracy,
                "macro_f1": macro_f1(labels, preds),
                "avg_margin": sum(margins) / max(len(margins), 1),
                "n_examples": len(labels),
                "classifier": f"tfidf_logreg_word12_char35_train_targets_maxfeat{args.max_features}",
            }
        )
        for (gold, pred), count in sorted(confusion.items()):
            confusion_rows.append(
                {
                    "candidate_key": key,
                    "gold_label": gold,
                    "predicted_label": pred,
                    "count": count,
                }
            )

    if not output_rows:
        print(f"[userstyle] {generations}: none of the candidate keys present, skipped"); return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(confusion_rows)

    print(json.dumps({"output": str(output_path), "confusion_output": str(confusion_path)}, indent=2))


if __name__ == "__main__":
    main()
