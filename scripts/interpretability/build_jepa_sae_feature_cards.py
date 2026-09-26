#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def short(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def fnum(value: str | float | None, *, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def best_correlation(rows: list[dict[str, str]], latent_id: str) -> dict[str, str]:
    best: dict[str, str] = {}
    best_score = -1.0
    for row in rows:
        if row.get("sae_type") != "jepa_sae" or str(row.get("latent_id")) != str(latent_id):
            continue
        try:
            score = abs(float(row.get("score", "nan")))
        except ValueError:
            continue
        if math.isnan(score):
            continue
        if score > best_score:
            best = row
            best_score = score
    return best


def representative_outputs(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_latent: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if not path.exists():
        return by_latent
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            latent_id = str(record.get("latent_id"))
            intervention = str(record.get("intervention"))
            if intervention not in {"base", "zero", "amplify", "swap_low"}:
                continue
            current = by_latent[latent_id]
            if not current:
                current["_example_id"] = record.get("example_id")
            if record.get("example_id") != current.get("_example_id"):
                continue
            current[intervention] = record
    return by_latent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reviewer-facing JEPA-SAE sparse latent feature cards from existing analysis tables."
    )
    parser.add_argument("--dataset", default="lamp5")
    parser.add_argument("--top-examples", default="tables/sparse_latent_top_examples_lamp5.csv")
    parser.add_argument("--label-correlations", default="tables/sparse_latent_label_correlations_lamp5.csv")
    parser.add_argument("--intervention-summary", default="tables/jepa_sae_latent_generation_interventions_lamp5.csv")
    parser.add_argument("--intervention-jsonl", default="runs/paper_followups/jepa_sae_latent_generation_interventions_lamp5.jsonl")
    parser.add_argument("--output-csv", default="tables/jepa_sae_feature_cards_lamp5.csv")
    parser.add_argument("--output-md", default="analysis/jepa_sae_feature_cards_lamp5.md")
    parser.add_argument("--max-top-examples", type=int, default=3)
    args = parser.parse_args()

    top_rows = read_csv(Path(args.top_examples))
    corr_rows = read_csv(Path(args.label_correlations))
    intervention_rows = read_csv(Path(args.intervention_summary))
    representative = representative_outputs(Path(args.intervention_jsonl))

    top_by_latent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in top_rows:
        if row.get("sae_type") == "jepa_sae":
            top_by_latent[str(row.get("latent_id"))].append(row)

    intervention_by_latent: dict[str, list[dict[str, str]]] = defaultdict(list)
    latent_order: list[str] = []
    for row in intervention_rows:
        latent_id = str(row.get("latent_id"))
        if latent_id not in latent_order:
            latent_order.append(latent_id)
        intervention_by_latent[latent_id].append(row)

    card_rows: list[dict[str, Any]] = []
    md_lines = [
        f"# {args.dataset} JEPA-SAE Feature Cards",
        "",
        "These cards combine top activating examples, simple metadata correlations, and generation-side interventions for the same JEPA-SAE checkpoint.",
        "They support a cautious interpretability claim: the sparse codes expose causal and coverage signals, but the labels are provisional and not a mature semantic dictionary.",
        "",
    ]

    for latent_id in latent_order:
        summary = {row["intervention"]: row for row in intervention_by_latent[latent_id]}
        base = summary.get("base", {})
        zero = summary.get("zero", {})
        amplify = summary.get("amplify", {})
        swap = summary.get("swap_low", {})
        corr = best_correlation(corr_rows, latent_id)
        top_examples = top_by_latent.get(latent_id, [])[: args.max_top_examples]
        labels = str(base.get("suggested_label", "")).strip() or "manual review needed"

        representative_rows = representative.get(latent_id, {})
        base_out = representative_rows.get("base", {})
        zero_out = representative_rows.get("zero", {})
        amp_out = representative_rows.get("amplify", {})
        swap_out = representative_rows.get("swap_low", {})

        card_rows.append(
            {
                "dataset": args.dataset,
                "latent_id": latent_id,
                "suggested_label": labels,
                "keywords": base.get("keywords", ""),
                "activation_frequency": fnum(base.get("activation_frequency")),
                "mean_activation": fnum(base.get("mean_activation")),
                "best_label_field": corr.get("label_field", base.get("label_field", "")),
                "best_label_metric": corr.get("metric", ""),
                "best_label_score": fnum(corr.get("score", base.get("label_score", ""))),
                "base_f1_rl": f"{fnum(base.get('token_f1'))} / {fnum(base.get('rouge_l_f1'))}",
                "zero_delta_f1_rl": f"{fnum(zero.get('delta_token_f1_vs_base'))} / {fnum(zero.get('delta_rouge_l_vs_base'))}",
                "amplify_delta_f1_rl": f"{fnum(amplify.get('delta_token_f1_vs_base'))} / {fnum(amplify.get('delta_rouge_l_vs_base'))}",
                "swap_low_delta_f1_rl": f"{fnum(swap.get('delta_token_f1_vs_base'))} / {fnum(swap.get('delta_rouge_l_vs_base'))}",
                "top_targets": " || ".join(short(row.get("target_snippet", ""), 100) for row in top_examples),
                "representative_target": short(base_out.get("target", ""), 120),
                "representative_base_output": short(base_out.get("output", ""), 120),
                "representative_zero_output": short(zero_out.get("output", ""), 120),
                "representative_amplify_output": short(amp_out.get("output", ""), 120),
                "representative_swap_low_output": short(swap_out.get("output", ""), 120),
            }
        )

        md_lines.extend(
            [
                f"## Latent {latent_id}: {labels}",
                "",
                f"- Activation: frequency `{fnum(base.get('activation_frequency'))}`, mean `{fnum(base.get('mean_activation'))}`.",
                f"- Best simple label association: `{corr.get('label_field', base.get('label_field', ''))}` "
                f"({corr.get('metric', '')}) = `{fnum(corr.get('score', base.get('label_score', '')))}`.",
                f"- Generation intervention deltas vs base F1/RL: zero `{fnum(zero.get('delta_token_f1_vs_base'))} / {fnum(zero.get('delta_rouge_l_vs_base'))}`, "
                f"amplify `{fnum(amplify.get('delta_token_f1_vs_base'))} / {fnum(amplify.get('delta_rouge_l_vs_base'))}`, "
                f"swap-low `{fnum(swap.get('delta_token_f1_vs_base'))} / {fnum(swap.get('delta_rouge_l_vs_base'))}`.",
                "",
                "Top activated targets:",
            ]
        )
        for row in top_examples:
            md_lines.append(
                f"- `{row.get('example_id', '')}` act `{fnum(row.get('activation'))}`: {short(row.get('target_snippet', ''), 180)}"
            )
        if base_out:
            md_lines.extend(
                [
                    "",
                    "Representative generation edit:",
                    f"- Target: {short(base_out.get('target', ''), 180)}",
                    f"- Base: {short(base_out.get('output', ''), 180)}",
                    f"- Zero: {short(zero_out.get('output', ''), 180)}",
                    f"- Amplify: {short(amp_out.get('output', ''), 180)}",
                    f"- Swap-low: {short(swap_out.get('output', ''), 180)}",
                ]
            )
        md_lines.append("")

    write_csv(Path(args.output_csv), card_rows)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"cards": len(card_rows), "csv": args.output_csv, "md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
