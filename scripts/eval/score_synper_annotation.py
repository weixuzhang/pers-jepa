#!/usr/bin/env python
"""Score a SynPer human-validation annotation round.

Unblinds annotations against the key, computes per-arm task-correctness and
persona-identification accuracy, and measures item-level agreement between
the annotator's persona picks and the paper's char-TF-IDF persona
classifier (replicated with the exact config of
evaluate_synper_persona_classifier_strong.py) on the identical items.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cohens_kappa(a, b):
    assert len(a) == len(b) and a
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(
        (sum(1 for x in a if x == l) / n) * (sum(1 for y in b if y == l) / n)
        for l in labels
    )
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train", default="data/synper/train_10000.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    items = {r["item_id"]: r for r in read_jsonl(ann_dir / "annotation_items.jsonl")}
    key = {r["item_id"]: r for r in read_jsonl(ann_dir / "annotation_key.jsonl")}
    annotations = {r["item_id"]: r for r in read_jsonl(args.annotations)}
    assert set(annotations) == set(key), "annotation ids must match key ids"

    # Replicate the paper's persona classifier exactly.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train = read_jsonl(args.train)
    train_texts = [str(r.get("target", "")) for r in train]
    train_labels = [r["metadata"]["persona"] for r in train]
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, n_jobs=1),
    )
    clf.fit(train_texts, train_labels)

    item_ids = sorted(annotations)
    clf_preds = dict(zip(item_ids, clf.predict([items[i]["output"] for i in item_ids])))

    per_arm = defaultdict(lambda: {
        "n": 0, "task_correct": 0,
        "human_persona_correct": 0, "clf_persona_correct": 0,
        "human_none_picks": 0,
    })
    human_picks, clf_picks = [], []
    rows = []
    for item_id in item_ids:
        ann, k = annotations[item_id], key[item_id]
        arm = k["arm"]
        stats = per_arm[arm]
        stats["n"] += 1
        stats["task_correct"] += ann["task_correct"]
        stats["human_persona_correct"] += int(ann["persona_pick"] == k["true_persona"])
        stats["clf_persona_correct"] += int(clf_preds[item_id] == k["true_persona"])
        stats["human_none_picks"] += int(ann["persona_pick"] == "none")
        human_picks.append(ann["persona_pick"])
        clf_picks.append(clf_preds[item_id])
        rows.append({
            "item_id": item_id, "arm": arm, "true_persona": k["true_persona"],
            "task_correct": ann["task_correct"],
            "human_pick": ann["persona_pick"], "clf_pick": clf_preds[item_id],
            "confidence": ann.get("persona_confidence"),
        })

    summary = {
        "per_arm": {
            arm: {
                "n": s["n"],
                "task_correct_rate": s["task_correct"] / s["n"],
                "human_persona_accuracy": s["human_persona_correct"] / s["n"],
                "classifier_persona_accuracy": s["clf_persona_correct"] / s["n"],
                "human_none_rate": s["human_none_picks"] / s["n"],
            }
            for arm, s in sorted(per_arm.items())
        },
        "human_vs_classifier": {
            "n_items": len(item_ids),
            "raw_agreement": sum(1 for h, c in zip(human_picks, clf_picks) if h == c) / len(item_ids),
            "cohens_kappa": cohens_kappa(human_picks, clf_picks),
            "note": "classifier has no 'none' option; every human 'none' pick is a guaranteed disagreement",
        },
        "human_vs_classifier_excluding_none": {},
        "per_item": rows,
    }
    keep = [(h, c) for h, c in zip(human_picks, clf_picks) if h != "none"]
    if keep:
        hk, ck = zip(*keep)
        summary["human_vs_classifier_excluding_none"] = {
            "n_items": len(keep),
            "raw_agreement": sum(1 for h, c in keep if h == c) / len(keep),
            "cohens_kappa": cohens_kappa(list(hk), list(ck)),
        }

    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for arm, s in summary["per_arm"].items():
        print(f"{arm}: task_ok={s['task_correct_rate']:.2f} "
              f"human_persona_acc={s['human_persona_accuracy']:.2f} "
              f"clf_persona_acc={s['classifier_persona_accuracy']:.2f} "
              f"none_rate={s['human_none_rate']:.2f}")
    print("human-vs-classifier:", json.dumps(summary["human_vs_classifier"], default=str))
    print("excluding none:", json.dumps(summary["human_vs_classifier_excluding_none"]))


if __name__ == "__main__":
    main()
