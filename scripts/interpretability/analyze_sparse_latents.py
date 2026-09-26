#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.hidden import require_torch
from persjepa.intervention import load_jepa_sae, load_standard_sae


def dotted_get(record: dict[str, Any], field: str, default: Any = None) -> Any:
    value: Any = record
    for part in field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif (
            isinstance(value, dict)
            and isinstance(value.get("metadata"), dict)
            and part in value["metadata"]
        ):
            value = value["metadata"][part]
        else:
            return default
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def short(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def label_values(examples: list[dict[str, Any]], field: str) -> list[Any]:
    values = []
    for example in examples:
        if field == "target_word_count":
            values.append(len(str(example.get("target", "")).split()))
        elif field == "profile_word_count":
            values.append(len(str(example.get("profile", "")).split()))
        else:
            values.append(dotted_get(example, field, None))
    return values


def numeric_correlation(x, y) -> float:
    torch = require_torch()
    x = x.float()
    y = y.float()
    if x.numel() < 2 or float(x.std().item()) == 0.0 or float(y.std().item()) == 0.0:
        return float("nan")
    x = (x - x.mean()) / x.std().clamp_min(1e-8)
    y = (y - y.mean()) / y.std().clamp_min(1e-8)
    return float((x * y).mean().item())


def categorical_eta_squared(activation, labels: list[Any]) -> float:
    torch = require_torch()
    valid = [(float(a), str(label)) for a, label in zip(activation.tolist(), labels) if label not in (None, "")]
    if len(valid) < 3:
        return float("nan")
    groups: dict[str, list[float]] = {}
    for value, label in valid:
        groups.setdefault(label, []).append(value)
    if len(groups) < 2 or len(groups) > 80:
        return float("nan")
    all_values = torch.tensor([value for value, _label in valid], dtype=torch.float32)
    total = float(((all_values - all_values.mean()) ** 2).sum().item())
    if total <= 1e-12:
        return float("nan")
    between = 0.0
    for values in groups.values():
        tensor = torch.tensor(values, dtype=torch.float32)
        between += float(tensor.numel()) * float((tensor.mean() - all_values.mean()).item()) ** 2
    return between / total


def load_latents(checkpoint: str, hidden_payload: dict[str, Any], sae_type: str, device):
    torch = require_torch()
    if sae_type == "jepa":
        model, payload = load_jepa_sae(checkpoint, device=device)
        source = hidden_payload["h_gen"].float().to(device)
        with torch.no_grad():
            latent, pre = model.encode(source)
        return latent.cpu(), pre.cpu(), payload
    if sae_type == "standard":
        model, payload = load_standard_sae(checkpoint, device=device)
        source = hidden_payload["h_pers"].float().to(device)
        with torch.no_grad():
            latent, pre = model.encode(source)
        return latent.cpu(), pre.cpu(), payload
    raise ValueError(f"Unknown SAE type: {sae_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse latent top-example and label-correlation analysis.")
    parser.add_argument("--hidden-pairs", required=True)
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--standard-checkpoint", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--label-fields", nargs="+", default=[
        "metadata.persona",
        "metadata.task_type",
        "metadata.rating",
        "metadata.history_count",
        "metadata.used_profile_count",
        "target_word_count",
        "profile_word_count",
    ])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--table-dir", default="tables")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch = require_torch()
    hidden_payload = torch.load(args.hidden_pairs, map_location="cpu")
    examples = hidden_payload.get("examples", [])
    if not examples:
        raise ValueError("--hidden-pairs payload does not contain examples.")

    table_dir = Path(args.table_dir)
    analysis_dir = Path(args.analysis_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    slug = args.dataset_name.lower().replace(" ", "_").replace("-", "_")

    model_specs = [("jepa_sae", "jepa", args.jepa_checkpoint)]
    if args.standard_checkpoint:
        model_specs.append(("standard_sae", "standard", args.standard_checkpoint))

    top_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    catalog_entries: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []

    labels_by_field = {field: label_values(examples, field) for field in args.label_fields}
    for method_name, sae_type, checkpoint in model_specs:
        latents, pre_acts, payload = load_latents(checkpoint, hidden_payload, sae_type, torch.device(args.device))
        active = latents > 0
        activation_frequency = active.float().mean(dim=0)
        mean_activation = latents.float().mean(dim=0)
        for latent_id in range(latents.shape[1]):
            activation = latents[:, latent_id].float()
            values, indices = torch.topk(activation, k=min(args.top_n, activation.numel()))
            for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                example = examples[idx]
                top_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "sae_type": method_name,
                        "latent_id": latent_id,
                        "rank": rank,
                        "activation": value,
                        "activation_frequency": float(activation_frequency[latent_id].item()),
                        "mean_activation": float(mean_activation[latent_id].item()),
                        "example_id": example.get("id", ""),
                        "profile_snippet": short(example.get("profile", "")),
                        "prompt_snippet": short(example.get("generic_prompt", example.get("prompt", ""))),
                        "target_snippet": short(example.get("target", "")),
                        "metadata": json.dumps(example.get("metadata", {}), ensure_ascii=False),
                    }
                )
            best_score = 0.0
            best_label = ""
            best_kind = ""
            for field, labels in labels_by_field.items():
                numeric_values = []
                is_numeric = True
                for label in labels:
                    try:
                        numeric_values.append(float(label))
                    except (TypeError, ValueError):
                        is_numeric = False
                        break
                if is_numeric:
                    score = numeric_correlation(activation, torch.tensor(numeric_values, dtype=torch.float32))
                    kind = "pearson"
                    comparable_score = abs(score) if not math.isnan(score) else float("nan")
                else:
                    score = categorical_eta_squared(activation, labels)
                    kind = "eta_squared"
                    comparable_score = score
                corr_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "sae_type": method_name,
                        "latent_id": latent_id,
                        "label_field": field,
                        "metric": kind,
                        "score": score,
                        "activation_frequency": float(activation_frequency[latent_id].item()),
                        "mean_activation": float(mean_activation[latent_id].item()),
                    }
                )
                if not math.isnan(comparable_score) and comparable_score > best_score:
                    best_score = comparable_score
                    best_label = field
                    best_kind = kind
            top_examples = [
                top_rows[-min(args.top_n, activation.numel()) + i]
                for i in range(min(5, min(args.top_n, activation.numel())))
            ]
            catalog_entries.append(
                (
                    best_score,
                    {
                        "dataset": args.dataset_name,
                        "sae_type": method_name,
                        "latent_id": latent_id,
                        "best_label": best_label,
                        "best_metric": best_kind,
                        "best_score": best_score,
                        "activation_frequency": float(activation_frequency[latent_id].item()),
                        "mean_activation": float(mean_activation[latent_id].item()),
                        "checkpoint": checkpoint,
                        "latent_dim": payload.get("latent_dim"),
                        "top_k": payload.get("top_k"),
                    },
                    top_examples,
                )
            )

    top_table = table_dir / f"sparse_latent_top_examples_{slug}.csv"
    corr_table = table_dir / f"sparse_latent_label_correlations_{slug}.csv"
    compare_table = table_dir / f"standard_sae_vs_jepa_sae_latent_probe_{slug}.csv"
    write_csv(top_table, top_rows)
    write_csv(corr_table, corr_rows)
    compare_rows = []
    for row in corr_rows:
        compare_rows.append(
            {
                "Method": row["sae_type"],
                "Dataset": row["dataset"],
                "Probe task": row["label_field"],
                "Accuracy / Correlation": row["score"],
                "Avg active latents": "",
                "Downstream F1 / ROUGE-L": "",
                "Notes": row["metric"],
            }
        )
    write_csv(compare_table, compare_rows)

    catalog_entries.sort(key=lambda item: item[0], reverse=True)
    lines = [
        f"# {args.dataset_name} Sparse Latent Initial Catalog",
        "",
        f"- Hidden pairs: `{args.hidden_pairs}`",
        f"- JEPA checkpoint: `{args.jepa_checkpoint}`",
        f"- Standard checkpoint: `{args.standard_checkpoint or ''}`",
        f"- Top examples table: `{top_table}`",
        f"- Label correlation table: `{corr_table}`",
        "",
    ]
    for _score, meta, examples_for_latent in catalog_entries[:10]:
        lines.extend(
            [
                f"## {meta['sae_type']} latent {meta['latent_id']}",
                "",
                f"- Best label: `{meta['best_label']}` via `{meta['best_metric']}` = `{meta['best_score']:.4f}`",
                f"- Activation frequency: `{meta['activation_frequency']:.4f}`",
                f"- Mean activation: `{meta['mean_activation']:.4f}`",
                "- Suggested label: TODO human review",
                "",
                "Top examples:",
            ]
        )
        for row in examples_for_latent[:5]:
            lines.append(
                f"- `{row['example_id']}` act={float(row['activation']):.3f}: "
                f"profile=`{row['profile_snippet']}` target=`{row['target_snippet']}`"
            )
        lines.append("")
    catalog_path = analysis_dir / f"sparse_latent_catalog_{slug}.md"
    catalog_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"top_table": str(top_table), "corr_table": str(corr_table), "catalog": str(catalog_path)}, indent=2))


if __name__ == "__main__":
    main()
