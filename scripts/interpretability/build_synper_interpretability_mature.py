#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.hidden import require_torch
from persjepa.intervention import load_jepa_sae, load_standard_sae


STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "our",
    "that",
    "the",
    "their",
    "this",
    "with",
    "you",
    "your",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def short(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def fnum(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def persona_from_profile(profile: str) -> str:
    match = re.search(r"Persona:\s*(.+?)(?:\n|$)", str(profile))
    return match.group(1).strip() if match else "__missing__"


def style_tags_from_profile(profile: str) -> list[str]:
    match = re.search(r"Style constraints:\s*(.+)", str(profile), flags=re.S)
    if not match:
        return []
    raw = match.group(1).strip().rstrip(".")
    tags = []
    for tag in re.split(r",|\sand\s", raw):
        tag = " ".join(tag.strip().lower().split())
        if tag:
            tags.append(tag)
    return tags


def metadata_value(record: dict[str, Any], key: str) -> str:
    metadata = record.get("metadata") or {}
    if key in metadata:
        return str(metadata[key])
    nested = metadata.get("metadata") if isinstance(metadata, dict) else None
    if isinstance(nested, dict) and key in nested:
        return str(nested[key])
    return ""


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", str(text))
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def eta_squared(values: list[float], labels: list[str]) -> float:
    valid = [(v, y) for v, y in zip(values, labels) if y]
    if len(valid) < 3:
        return float("nan")
    groups: dict[str, list[float]] = defaultdict(list)
    for value, label in valid:
        groups[label].append(float(value))
    if len(groups) < 2:
        return float("nan")
    all_values = [value for value, _label in valid]
    mean = sum(all_values) / len(all_values)
    total = sum((value - mean) ** 2 for value in all_values)
    if total <= 1e-12:
        return float("nan")
    between = 0.0
    for group_values in groups.values():
        group_mean = sum(group_values) / len(group_values)
        between += len(group_values) * (group_mean - mean) ** 2
    return between / total


def binary_eta(values: list[float], labels: list[int]) -> tuple[float, float, float]:
    y = ["1" if label else "0" for label in labels]
    score = eta_squared(values, y)
    pos = [value for value, label in zip(values, labels) if label]
    neg = [value for value, label in zip(values, labels) if not label]
    mean_pos = sum(pos) / len(pos) if pos else float("nan")
    mean_neg = sum(neg) / len(neg) if neg else float("nan")
    return score, mean_pos, mean_neg


def load_latent_matrix(hidden_pairs: Path, checkpoint: Path, sae_type: str, device: str):
    torch = require_torch()
    payload = torch.load(hidden_pairs, map_location="cpu")
    examples = payload["examples"]
    if sae_type == "jepa_sae":
        model, _meta = load_jepa_sae(str(checkpoint), device=torch.device(device))
        source = payload["h_gen"].to(device)
    elif sae_type == "standard_sae":
        model, _meta = load_standard_sae(str(checkpoint), device=torch.device(device))
        source = payload["h_pers"].to(device)
    else:
        raise ValueError(sae_type)
    dtype_source = next(model.parameters(), None)
    dtype = dtype_source.dtype if dtype_source is not None else source.dtype
    with torch.no_grad():
        latent, _pre = model.encode(source.to(dtype=dtype))
    return latent.detach().float().cpu(), examples


def build_style_correlations(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = require_torch()
    payload = torch.load(args.hidden_pairs, map_location="cpu")
    examples = payload["examples"]
    style_counts = Counter()
    example_tags = []
    for example in examples:
        tags = style_tags_from_profile(str(example.get("profile", "")))
        example_tags.append(set(tags))
        style_counts.update(tags)
    style_tags = sorted(tag for tag, count in style_counts.items() if count >= args.min_style_count)

    rows: list[dict[str, Any]] = []
    for sae_type, checkpoint, source_key in [
        ("jepa_sae", args.jepa_checkpoint, "h_gen"),
        ("standard_sae", args.standard_checkpoint, "h_pers"),
    ]:
        if not checkpoint:
            continue
        if sae_type == "jepa_sae":
            model, _meta = load_jepa_sae(str(checkpoint), device=args.device)
        else:
            model, _meta = load_standard_sae(str(checkpoint), device=args.device)
        source = payload[source_key].to(args.device)
        dtype_source = next(model.parameters(), None)
        dtype = dtype_source.dtype if dtype_source is not None else source.dtype
        with torch.no_grad():
            latent, _pre = model.encode(source.to(dtype=dtype))
        latent = latent.detach().float().cpu()
        active = latent > 0
        activation_frequency = active.float().mean(dim=0).tolist()
        mean_activation = latent.mean(dim=0).tolist()
        for latent_id in range(latent.shape[1]):
            values = latent[:, latent_id].tolist()
            for tag in style_tags:
                labels = [1 if tag in tags else 0 for tags in example_tags]
                score, mean_pos, mean_neg = binary_eta(values, labels)
                rows.append(
                    {
                        "dataset": "SynPer",
                        "sae_type": sae_type,
                        "latent_id": latent_id,
                        "style_tag": tag,
                        "metric": "eta_squared_binary",
                        "score": score,
                        "mean_activation_positive": mean_pos,
                        "mean_activation_negative": mean_neg,
                        "positive_count": sum(labels),
                        "activation_frequency": activation_frequency[latent_id],
                        "mean_activation": mean_activation[latent_id],
                    }
                )
    return rows


def build_classifier(train_rows: list[dict[str, Any]], label_fn):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts = [str(row.get("target", "")) for row in train_rows]
    labels = [label_fn(row) for row in train_rows]
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, n_jobs=1),
    )
    clf.fit(texts, labels)
    return clf


def classifier_true_prob(clf, texts: list[str], labels: list[str]) -> tuple[list[str], list[float]]:
    preds = [str(value) for value in clf.predict(texts)]
    probs = clf.predict_proba(texts)
    classes = [str(value) for value in clf.classes_]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    true_probs = [
        float(probs[row_idx, class_to_idx[label]]) if label in class_to_idx else 0.0
        for row_idx, label in enumerate(labels)
    ]
    return preds, true_probs


def average(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else float("nan")


def build_behavior(args: argparse.Namespace, train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = read_jsonl(args.intervention_jsonl)
    id_to_train = {str(row.get("id")): row for row in train_rows}
    persona_clf = build_classifier(train_rows, lambda row: persona_from_profile(str(row.get("profile", ""))))
    task_clf = build_classifier(train_rows, lambda row: metadata_value(row, "task_type") or "__missing__")

    enriched = []
    for record in records:
        source = id_to_train.get(str(record.get("example_id")), {})
        true_persona = persona_from_profile(str(source.get("profile") or record.get("profile_snippet", "")))
        true_task = metadata_value(source, "task_type") or "__missing__"
        style_tags = style_tags_from_profile(str(source.get("profile") or record.get("profile_snippet", "")))
        output = str(record.get("output", ""))
        style_words = [word for tag in style_tags for word in tokenize(tag)]
        output_words = set(tokenize(output))
        style_hit = (
            sum(1 for word in style_words if word in output_words) / max(len(style_words), 1)
            if style_words
            else 0.0
        )
        enriched.append({**record, "true_persona": true_persona, "true_task": true_task, "style_keyword_hit_rate": style_hit})

    persona_preds, persona_probs = classifier_true_prob(
        persona_clf,
        [str(row.get("output", "")) for row in enriched],
        [str(row["true_persona"]) for row in enriched],
    )
    task_preds, task_probs = classifier_true_prob(
        task_clf,
        [str(row.get("output", "")) for row in enriched],
        [str(row["true_task"]) for row in enriched],
    )
    for row, persona_pred, persona_prob, task_pred, task_prob in zip(
        enriched,
        persona_preds,
        persona_probs,
        task_preds,
        task_probs,
    ):
        row["persona_pred"] = persona_pred
        row["true_persona_prob"] = persona_prob
        row["persona_correct"] = float(persona_pred == row["true_persona"])
        row["task_pred"] = task_pred
        row["true_task_prob"] = task_prob
        row["task_correct"] = float(task_pred == row["true_task"])

    by_latent_intervention: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_latent_intervention[(int(row["latent_id"]), str(row["intervention"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    latent_ids = sorted({latent_id for latent_id, _intervention in by_latent_intervention})
    for latent_id in latent_ids:
        base_rows = by_latent_intervention[(latent_id, "base")]
        base_persona_prob = average(base_rows, "true_persona_prob")
        base_task_prob = average(base_rows, "true_task_prob")
        base_style_hit = average(base_rows, "style_keyword_hit_rate")
        base_f1 = average(base_rows, "token_f1")
        for intervention in ["base", "zero", "amplify", "swap_low"]:
            rows = by_latent_intervention[(latent_id, intervention)]
            if not rows:
                continue
            summary_rows.append(
                {
                    "dataset": "SynPer",
                    "latent_id": latent_id,
                    "suggested_label": rows[0].get("suggested_label", ""),
                    "intervention": intervention,
                    "n_examples": len(rows),
                    "token_f1": average(rows, "token_f1"),
                    "rouge_l_f1": average(rows, "rouge_l_f1"),
                    "persona_accuracy": average(rows, "persona_correct"),
                    "avg_true_persona_prob": average(rows, "true_persona_prob"),
                    "task_accuracy": average(rows, "task_correct"),
                    "avg_true_task_prob": average(rows, "true_task_prob"),
                    "style_keyword_hit_rate": average(rows, "style_keyword_hit_rate"),
                    "delta_token_f1_vs_base": average(rows, "token_f1") - base_f1,
                    "delta_true_persona_prob_vs_base": average(rows, "true_persona_prob") - base_persona_prob,
                    "delta_true_task_prob_vs_base": average(rows, "true_task_prob") - base_task_prob,
                    "delta_style_keyword_hit_vs_base": average(rows, "style_keyword_hit_rate") - base_style_hit,
                }
            )
    return summary_rows


def best_score(rows: list[dict[str, str]], *, sae_type: str, latent_id: str, field: str) -> float:
    vals = []
    for row in rows:
        if row.get("sae_type") != sae_type or str(row.get("latent_id")) != str(latent_id):
            continue
        if row.get("label_field") != field:
            continue
        try:
            vals.append(abs(float(row.get("score", "nan"))))
        except ValueError:
            pass
    vals = [value for value in vals if not math.isnan(value)]
    return max(vals) if vals else float("nan")


def best_style(style_rows: list[dict[str, Any]], *, sae_type: str, latent_id: str) -> tuple[str, float]:
    candidates = []
    for row in style_rows:
        if row.get("sae_type") != sae_type or str(row.get("latent_id")) != str(latent_id):
            continue
        try:
            score = abs(float(row.get("score", "nan")))
        except ValueError:
            continue
        if not math.isnan(score):
            candidates.append((score, str(row.get("style_tag", ""))))
    if not candidates:
        return "", float("nan")
    score, tag = max(candidates)
    return tag, score


def activation_frequency_for_latent(label_rows: list[dict[str, str]], style_rows: list[dict[str, Any]], latent_id: str) -> float:
    candidates: list[float] = []
    for row in label_rows:
        if row.get("sae_type") == "jepa_sae" and str(row.get("latent_id")) == str(latent_id):
            try:
                value = float(row.get("activation_frequency", "nan"))
            except ValueError:
                continue
            if not math.isnan(value):
                candidates.append(value)
    for row in style_rows:
        if row.get("sae_type") == "jepa_sae" and str(row.get("latent_id")) == str(latent_id):
            try:
                value = float(row.get("activation_frequency", "nan"))
            except ValueError:
                continue
            if not math.isnan(value):
                candidates.append(value)
    return candidates[0] if candidates else float("nan")


def summarize_probe(label_rows: list[dict[str, str]], style_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for sae_type in ["jepa_sae", "standard_sae"]:
        for field, source in [
            ("metadata.persona", "label"),
            ("metadata.task_type", "label"),
            ("style_tag", "style"),
        ]:
            if source == "style":
                vals = [
                    abs(float(row["score"]))
                    for row in style_rows
                    if row.get("sae_type") == sae_type and str(row.get("score", "nan")).lower() != "nan"
                ]
            else:
                vals = [
                    abs(float(row["score"]))
                    for row in label_rows
                    if row.get("sae_type") == sae_type
                    and row.get("label_field") == field
                    and str(row.get("score", "nan")).lower() != "nan"
                ]
            vals = sorted(vals, reverse=True)
            out.append(
                {
                    "dataset": "SynPer",
                    "sae_type": sae_type,
                    "label_group": field,
                    "max_abs_association": vals[0] if vals else float("nan"),
                    "top5_mean_abs_association": sum(vals[:5]) / max(len(vals[:5]), 1),
                    "median_abs_association": vals[len(vals) // 2] if vals else float("nan"),
                    "n_scores": len(vals),
                }
            )
    return out


def parse_metadata_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {}
    metadata = obj.get("metadata", obj)
    return metadata if isinstance(metadata, dict) else {}


def build_feature_cards(
    *,
    top_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    style_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    intervention_records: list[dict[str, Any]],
    latent_ids: list[str],
    output_md: Path,
) -> list[dict[str, Any]]:
    top_by_latent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in top_rows:
        if row.get("sae_type") == "jepa_sae":
            top_by_latent[str(row.get("latent_id"))].append(row)

    behavior_by_key = {
        (str(row["latent_id"]), str(row["intervention"])): row
        for row in behavior_rows
    }
    reps: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in intervention_records:
        latent_id = str(row.get("latent_id"))
        if not reps[latent_id]:
            reps[latent_id]["_example_id"] = row.get("example_id")
        if row.get("example_id") == reps[latent_id].get("_example_id"):
            reps[latent_id][str(row.get("intervention"))] = row

    card_rows: list[dict[str, Any]] = []
    md = [
        "# SynPer JEPA-SAE Mature Feature Cards",
        "",
        "Each card combines top activations, persona/style/task associations, and generation-side single-latent edits.",
        "Labels are still provisional, but the evidence is now broader than reconstruction MSE: top examples, label correlations, and generated behavior are tied to the same latent ids.",
        "",
    ]
    for latent_id in latent_ids:
        tops = top_by_latent.get(str(latent_id), [])[:5]
        personas = Counter()
        tasks = Counter()
        words = Counter()
        for row in tops[:20]:
            metadata = parse_metadata_json(row.get("metadata", ""))
            profile = row.get("profile_snippet", "")
            persona = metadata.get("persona") or persona_from_profile(profile)
            task = metadata.get("task_type") or ""
            if persona:
                personas[str(persona)] += 1
            if task:
                tasks[str(task)] += 1
            words.update(tokenize(row.get("target_snippet", "")))
        keywords = " / ".join(word for word, _count in words.most_common(4))
        top_persona = personas.most_common(1)[0][0] if personas else ""
        top_task = tasks.most_common(1)[0][0] if tasks else ""
        style_tag, style_score = best_style(style_rows, sae_type="jepa_sae", latent_id=str(latent_id))
        persona_score = best_score(label_rows, sae_type="jepa_sae", latent_id=str(latent_id), field="metadata.persona")
        task_score = best_score(label_rows, sae_type="jepa_sae", latent_id=str(latent_id), field="metadata.task_type")
        activation_frequency = activation_frequency_for_latent(label_rows, style_rows, str(latent_id))
        base = behavior_by_key.get((str(latent_id), "base"), {})
        zero = behavior_by_key.get((str(latent_id), "zero"), {})
        amp = behavior_by_key.get((str(latent_id), "amplify"), {})
        swap = behavior_by_key.get((str(latent_id), "swap_low"), {})
        label = f"{top_task or 'mixed task'}; {style_tag or top_persona}; {keywords}".strip("; ")
        card_rows.append(
            {
                "dataset": "SynPer",
                "latent_id": latent_id,
                "provisional_label": label,
                "top_persona": top_persona,
                "top_task": top_task,
                "best_style_tag": style_tag,
                "style_assoc": style_score,
                "persona_assoc": persona_score,
                "task_assoc": task_score,
                "activation_frequency": activation_frequency,
                "base_f1": base.get("token_f1", ""),
                "base_true_persona_prob": base.get("avg_true_persona_prob", ""),
                "zero_delta_f1": zero.get("delta_token_f1_vs_base", ""),
                "zero_delta_true_persona_prob": zero.get("delta_true_persona_prob_vs_base", ""),
                "amplify_delta_f1": amp.get("delta_token_f1_vs_base", ""),
                "amplify_delta_true_persona_prob": amp.get("delta_true_persona_prob_vs_base", ""),
                "swap_low_delta_f1": swap.get("delta_token_f1_vs_base", ""),
                "swap_low_delta_true_persona_prob": swap.get("delta_true_persona_prob_vs_base", ""),
                "top_examples": " || ".join(short(row.get("target_snippet", ""), 100) for row in tops[:3]),
            }
        )

        md.extend(
            [
                f"## Latent {latent_id}: {label}",
                "",
                f"- Associations: style `{style_tag}` = `{fnum(style_score)}`, persona = `{fnum(persona_score)}`, task = `{fnum(task_score)}`.",
                f"- Top activated persona/task: `{top_persona}` / `{top_task}`.",
                f"- Generation behavior deltas vs base: zero F1 `{fnum(zero.get('delta_token_f1_vs_base'))}`, zero true-persona prob `{fnum(zero.get('delta_true_persona_prob_vs_base'))}`; "
                f"amplify F1 `{fnum(amp.get('delta_token_f1_vs_base'))}`, amplify true-persona prob `{fnum(amp.get('delta_true_persona_prob_vs_base'))}`; "
                f"swap-low F1 `{fnum(swap.get('delta_token_f1_vs_base'))}`, swap-low true-persona prob `{fnum(swap.get('delta_true_persona_prob_vs_base'))}`.",
                "",
                "Top activated examples:",
            ]
        )
        for row in tops[:5]:
            md.append(f"- `{row.get('example_id', '')}` act `{fnum(row.get('activation'))}`: {short(row.get('target_snippet', ''), 220)}")
        rep = reps.get(str(latent_id), {})
        if rep:
            md.extend(
                [
                    "",
                    "Representative generated behavior:",
                    f"- Target: {short(rep.get('base', {}).get('target', ''), 220)}",
                    f"- Base: {short(rep.get('base', {}).get('output', ''), 220)}",
                    f"- Zero: {short(rep.get('zero', {}).get('output', ''), 220)}",
                    f"- Amplify: {short(rep.get('amplify', {}).get('output', ''), 220)}",
                    f"- Swap-low: {short(rep.get('swap_low', {}).get('output', ''), 220)}",
                ]
            )
        md.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md), encoding="utf-8")
    return card_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mature SynPer JEPA-SAE interpretability artifacts.")
    parser.add_argument("--hidden-pairs", type=Path, default=Path("runs/synper_qwen25_3b_hidden_anchor_train10k/hidden_pairs_anchor.pt"))
    parser.add_argument("--jepa-checkpoint", type=Path, default=Path("runs/synper_qwen25_3b_hidden_anchor_train10k/ld128_k08/jepa_sae_mse.pt"))
    parser.add_argument("--standard-checkpoint", type=Path, default=Path("runs/synper_qwen25_3b_hidden_anchor_train10k/ld128_k08/standard_sae.pt"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/synper/train_10000.jsonl"))
    parser.add_argument("--top-examples", type=Path, default=Path("tables/sparse_latent_top_examples_synper.csv"))
    parser.add_argument("--label-correlations", type=Path, default=Path("tables/sparse_latent_label_correlations_synper.csv"))
    parser.add_argument("--intervention-jsonl", type=Path, default=Path("runs/synper_qwen25_3b_hidden_anchor_train10k/interventions/jepa_sae_latent_generation_interventions_synper_top10.jsonl"))
    parser.add_argument("--style-output", type=Path, default=Path("tables/sparse_latent_style_tag_correlations_synper.csv"))
    parser.add_argument("--behavior-output", type=Path, default=Path("tables/jepa_sae_generation_behavior_interventions_synper_top10.csv"))
    parser.add_argument("--probe-summary-output", type=Path, default=Path("tables/synper_sae_latent_probe_summary_mature.csv"))
    parser.add_argument("--cards-output", type=Path, default=Path("tables/jepa_sae_feature_cards_synper_mature.csv"))
    parser.add_argument("--cards-md-output", type=Path, default=Path("analysis/jepa_sae_feature_cards_synper_mature.md"))
    parser.add_argument("--min-style-count", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_jsonl)
    label_rows = read_csv(args.label_correlations)
    top_rows = read_csv(args.top_examples)
    intervention_records = read_jsonl(args.intervention_jsonl)

    style_rows = build_style_correlations(args)
    write_csv(args.style_output, style_rows)
    behavior_rows = build_behavior(args, train_rows)
    write_csv(args.behavior_output, behavior_rows)
    probe_rows = summarize_probe(label_rows, style_rows)
    write_csv(args.probe_summary_output, probe_rows)

    latent_ids = []
    for row in intervention_records:
        latent_id = str(row.get("latent_id"))
        if latent_id not in latent_ids:
            latent_ids.append(latent_id)
    card_rows = build_feature_cards(
        top_rows=top_rows,
        label_rows=label_rows,
        style_rows=style_rows,
        behavior_rows=behavior_rows,
        intervention_records=intervention_records,
        latent_ids=latent_ids,
        output_md=args.cards_md_output,
    )
    write_csv(args.cards_output, card_rows)
    print(
        json.dumps(
            {
                "style_correlations": str(args.style_output),
                "behavior": str(args.behavior_output),
                "probe_summary": str(args.probe_summary_output),
                "feature_cards": str(args.cards_output),
                "feature_cards_md": str(args.cards_md_output),
                "n_cards": len(card_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
