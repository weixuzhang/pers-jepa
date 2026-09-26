#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.hidden import require_torch


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


def auto_group_field(examples: list[dict[str, Any]]) -> str | None:
    candidates = [
        "metadata.user_id",
        "metadata.metadata.user_id",
        "metadata.persona",
        "metadata.metadata.persona",
        "metadata.author_id",
        "metadata.source",
    ]
    for field in candidates:
        values = [dotted_get(example, field) for example in examples[:200]]
        values = [value for value in values if value not in (None, "")]
        if len(set(values)) >= 2:
            return field
    return None


def group_values(examples: list[dict[str, Any]], field: str | None) -> list[str]:
    values = []
    for idx, example in enumerate(examples):
        value = dotted_get(example, field, None) if field else None
        if value in (None, ""):
            value = f"ungrouped-{idx}"
        values.append(str(value))
    return values


def answer_token_example_means(payload: dict[str, Any]):
    torch = require_torch()
    source = payload["source_answer_tokens"].float()
    target = payload["target_answer_tokens"].float()
    mask = (payload["source_answer_token_mask"].bool() & payload["target_answer_token_mask"].bool()).float()
    denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    source_mean = (source * mask.unsqueeze(-1)).sum(dim=1) / denom
    target_mean = (target * mask.unsqueeze(-1)).sum(dim=1) / denom
    return source_mean, target_mean


def select_source_target(payload: dict[str, Any], kind: str):
    if kind == "answer_tokens":
        if "source_answer_tokens" not in payload:
            raise KeyError("answer_tokens requires --span-hidden with answer token tensors.")
        return answer_token_example_means(payload)
    if kind == "answer_mean":
        return payload["source_answer_span_mean"].float(), payload["target_answer_span_mean"].float()
    if kind == "anchor":
        return payload["source_anchor"].float(), payload["target_anchor"].float()
    if kind == "task_mean":
        return payload["source_span_mean"].float(), payload["target_task_span_mean"].float()
    raise ValueError(f"Unknown residual kind: {kind}")


def cosine_mean(a, b) -> float:
    torch = require_torch()
    if a.numel() == 0:
        return float("nan")
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean().item())


def method_metrics(source, target, pred, *, baseline_mse: float | None = None) -> dict[str, float]:
    torch = require_torch()
    mse = float(torch.nn.functional.mse_loss(pred.float(), target.float()).item())
    if baseline_mse is None:
        baseline_mse = float(torch.nn.functional.mse_loss(source.float(), target.float()).item())
    return {
        "hidden_mse": mse,
        "cosine_to_target": cosine_mean(pred, target),
        "gap_recovery": (baseline_mse - mse) / baseline_mse if baseline_mse else 0.0,
        "delta_norm": float((pred.float() - source.float()).norm(dim=-1).mean().item()),
        "target_delta_norm": float((target.float() - source.float()).norm(dim=-1).mean().item()),
        "n_eval": int(source.shape[0]),
    }


def split_by_group(
    groups: list[str],
    *,
    min_count: int,
    calibration_examples: int,
    seed: int,
) -> tuple[list[int], list[int], dict[str, list[int]], dict[str, list[int]]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)
    rng = random.Random(seed)
    cal_by_group: dict[str, list[int]] = {}
    eval_by_group: dict[str, list[int]] = {}
    cal_indices: list[int] = []
    eval_indices: list[int] = []
    for group, indices in by_group.items():
        if len(indices) < max(min_count, calibration_examples + 1):
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        cal = shuffled[:calibration_examples]
        eval_ = shuffled[calibration_examples:]
        cal_by_group[group] = cal
        eval_by_group[group] = eval_
        cal_indices.extend(cal)
        eval_indices.extend(eval_)
    return cal_indices, eval_indices, cal_by_group, eval_by_group


def make_group_deltas(residuals, cal_by_group: dict[str, list[int]]):
    torch = require_torch()
    deltas = {}
    for group, indices in cal_by_group.items():
        idx = torch.tensor(indices, dtype=torch.long)
        deltas[group] = residuals.index_select(0, idx).mean(dim=0)
    return deltas


def average_pair_cosine(vectors) -> float:
    torch = require_torch()
    if vectors.shape[0] < 2:
        return float("nan")
    x = torch.nn.functional.normalize(vectors.float(), dim=-1)
    sim = x @ x.T
    mask = ~torch.eye(x.shape[0], dtype=torch.bool)
    return float(sim[mask].mean().item())


def within_between_summary(residuals, groups: list[str], indices: list[int]) -> dict[str, float]:
    torch = require_torch()
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_group[groups[idx]].append(idx)
    within = []
    centroids = []
    for group_indices in by_group.values():
        if len(group_indices) >= 2:
            tensor_idx = torch.tensor(group_indices, dtype=torch.long)
            vecs = residuals.index_select(0, tensor_idx)
            within.append(average_pair_cosine(vecs))
            centroids.append(vecs.mean(dim=0))
    centroid_tensor = torch.stack(centroids, dim=0) if centroids else residuals.new_empty((0, residuals.shape[-1]))
    return {
        "within_group_residual_cosine": float(sum(within) / len(within)) if within else float("nan"),
        "between_group_centroid_cosine": average_pair_cosine(centroid_tensor),
        "n_groups": float(len(by_group)),
    }


def pca_directions(residuals, max_rank: int):
    torch = require_torch()
    centered = residuals.float() - residuals.float().mean(dim=0, keepdim=True)
    # Full SVD is fine for the current 300-3000 example artifacts and avoids sklearn.
    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    total_var = (s**2).sum().clamp_min(1e-12)
    explained = (s**2) / total_var
    return centered.mean(dim=0), vh[:max_rank], explained


def lowrank_predict(source, mean_delta, directions, coeffs):
    return source + mean_delta + coeffs @ directions


def ridge_coeff_predict(train_source, train_coeff, eval_source, *, ridge: float = 1e-2):
    torch = require_torch()
    x = torch.cat([train_source.float(), torch.ones(train_source.shape[0], 1)], dim=1)
    y = train_coeff.float()
    eye = torch.eye(x.shape[1])
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(x.T @ x + ridge * eye, x.T @ y)
    x_eval = torch.cat([eval_source.float(), torch.ones(eval_source.shape[0], 1)], dim=1)
    return x_eval @ weights


def kmeans(x, k: int, *, seed: int, iters: int = 40):
    torch = require_torch()
    if x.shape[0] < k:
        raise ValueError("k cannot exceed number of examples.")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)[:k]
    centroids = x.float().index_select(0, perm).clone()
    labels = None
    for _ in range(iters):
        distances = torch.cdist(x.float(), centroids)
        labels = distances.argmin(dim=1)
        new_centroids = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if bool(mask.any()):
                new_centroids.append(x.float()[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[cluster_id])
        new_centroids = torch.stack(new_centroids, dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-5, rtol=1e-5):
            break
        centroids = new_centroids
    assert labels is not None
    return centroids, labels


def cluster_predict(train_source, train_residuals, eval_source, k: int, *, seed: int):
    torch = require_torch()
    centroids, train_labels = kmeans(train_source, k, seed=seed)
    cluster_deltas = []
    for cluster_id in range(k):
        mask = train_labels == cluster_id
        if bool(mask.any()):
            cluster_deltas.append(train_residuals[mask].mean(dim=0))
        else:
            cluster_deltas.append(train_residuals.mean(dim=0))
    cluster_deltas = torch.stack(cluster_deltas, dim=0)
    eval_labels = torch.cdist(eval_source.float(), centroids).argmin(dim=1)
    return eval_source + cluster_deltas.index_select(0, eval_labels), train_labels


def cluster_purity(labels, groups: list[str], train_indices: list[int]) -> float:
    if len(train_indices) == 0:
        return float("nan")
    label_to_groups: dict[int, Counter] = defaultdict(Counter)
    for row, original_idx in enumerate(train_indices):
        label_to_groups[int(labels[row].item())][groups[original_idx]] += 1
    correct = sum(counter.most_common(1)[0][1] for counter in label_to_groups.values() if counter)
    return correct / len(train_indices)


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


def dataset_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hidden-only user/persona/group residual analysis."
    )
    parser.add_argument("--span-hidden", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--group-field", default="auto")
    parser.add_argument("--residual-kind", default="answer_tokens", choices=["answer_tokens", "answer_mean", "anchor", "task_mean"])
    parser.add_argument("--min-group-counts", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--calibration-examples", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--cluster-counts", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--lowrank-ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table-dir", default="tables")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--run-dir", default="runs/personalization_residuals")
    args = parser.parse_args()

    torch = require_torch()
    payload = torch.load(args.span_hidden, map_location="cpu")
    examples = payload.get("examples", [])
    if not examples:
        raise ValueError("--span-hidden payload does not contain examples.")

    group_field = auto_group_field(examples) if args.group_field == "auto" else args.group_field
    groups = group_values(examples, group_field)
    source, target = select_source_target(payload, args.residual_kind)
    residuals = target - source

    table_dir = Path(args.table_dir)
    analysis_dir = Path(args.analysis_dir)
    run_dir = Path(args.run_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    slug = dataset_slug(args.dataset_name)

    rows: list[dict[str, Any]] = []
    aux_rows: list[dict[str, Any]] = []
    checkpoint_groups: dict[str, Any] = {}

    for min_count in args.min_group_counts:
        for cal_n in args.calibration_examples:
            cal_idx, eval_idx, cal_by_group, eval_by_group = split_by_group(
                groups,
                min_count=min_count,
                calibration_examples=cal_n,
                seed=args.seed,
            )
            if not cal_idx or not eval_idx:
                continue
            cal_tensor = torch.tensor(cal_idx, dtype=torch.long)
            eval_tensor = torch.tensor(eval_idx, dtype=torch.long)
            train_source = source.index_select(0, cal_tensor)
            train_target = target.index_select(0, cal_tensor)
            train_residuals = residuals.index_select(0, cal_tensor)
            eval_source = source.index_select(0, eval_tensor)
            eval_target = target.index_select(0, eval_tensor)
            baseline_mse = float(torch.nn.functional.mse_loss(eval_source.float(), eval_target.float()).item())
            global_delta = train_residuals.mean(dim=0)
            group_delta_by_name = make_group_deltas(residuals, cal_by_group)
            pred_group = []
            for original_idx in eval_idx:
                pred_group.append(source[original_idx] + group_delta_by_name[groups[original_idx]])
            pred_group_tensor = torch.stack(pred_group, dim=0)

            common = {
                "dataset": args.dataset_name,
                "group_field": group_field or "",
                "residual_kind": args.residual_kind,
                "min_group_count": min_count,
                "calibration_examples_per_group": cal_n,
                "n_calibration": len(cal_idx),
                "n_groups": len(cal_by_group),
            }
            for method, setting, pred in [
                ("identity", "strict_profile_free", eval_source),
                ("global_answer_token_mean_delta", "strict_profile_free", eval_source + global_delta),
                ("group_answer_token_mean_delta", "stored_vector_profile_free", pred_group_tensor),
            ]:
                metric = method_metrics(eval_source, eval_target, pred, baseline_mse=baseline_mse)
                rows.append(common | {"method": method, "inference_setting": setting} | metric)

            max_rank = max(args.lowrank_ranks) if args.lowrank_ranks else 0
            if max_rank > 0 and train_residuals.shape[0] > 1:
                _center_mean, directions, explained = pca_directions(train_residuals, max_rank)
                mean_delta = train_residuals.mean(dim=0)
                train_centered = train_residuals - mean_delta
                train_coeff_all = train_centered @ directions.T
                eval_coeff_ridge_full = ridge_coeff_predict(train_source, train_coeff_all, eval_source)
                for rank in args.lowrank_ranks:
                    if rank > directions.shape[0]:
                        continue
                    dirs = directions[:rank]
                    coeff_ridge = eval_coeff_ridge_full[:, :rank]
                    pred_ridge = lowrank_predict(eval_source, mean_delta, dirs, coeff_ridge)
                    metric = method_metrics(eval_source, eval_target, pred_ridge, baseline_mse=baseline_mse)
                    aux_rows.append(
                        common
                        | {
                            "method": "lowrank_hgen_ridge",
                            "inference_setting": "strict_profile_free",
                            "hyperparameter": f"rank={rank}",
                            "cluster_purity": "",
                            "pca_cumulative_variance": float(explained[:rank].sum().item()),
                        }
                        | metric
                    )
                    group_coeffs = {}
                    for group, indices in cal_by_group.items():
                        local = torch.tensor([cal_idx.index(i) for i in indices], dtype=torch.long)
                        group_coeffs[group] = train_coeff_all.index_select(0, local)[:, :rank].mean(dim=0)
                    pred_user = []
                    for original_idx in eval_idx:
                        coeff = group_coeffs[groups[original_idx]]
                        pred_user.append(source[original_idx] + mean_delta + coeff @ dirs)
                    pred_user_tensor = torch.stack(pred_user, dim=0)
                    metric = method_metrics(eval_source, eval_target, pred_user_tensor, baseline_mse=baseline_mse)
                    aux_rows.append(
                        common
                        | {
                            "method": "lowrank_group_coefficients",
                            "inference_setting": "stored_vector_profile_free",
                            "hyperparameter": f"rank={rank}",
                            "cluster_purity": "",
                            "pca_cumulative_variance": float(explained[:rank].sum().item()),
                        }
                        | metric
                    )

            for k in args.cluster_counts:
                if k > len(cal_idx):
                    continue
                pred_cluster, train_labels = cluster_predict(train_source, train_residuals, eval_source, k, seed=args.seed)
                metric = method_metrics(eval_source, eval_target, pred_cluster, baseline_mse=baseline_mse)
                aux_rows.append(
                    common
                    | {
                        "method": "cluster_mean_delta_hgen",
                        "inference_setting": "strict_profile_free",
                        "hyperparameter": f"k={k}",
                        "cluster_purity": cluster_purity(train_labels, groups, cal_idx),
                        "pca_cumulative_variance": "",
                    }
                    | metric
                )

            key = f"min{min_count}_cal{cal_n}"
            checkpoint_groups[key] = {
                "global_delta": global_delta,
                "group_deltas": {group: delta for group, delta in group_delta_by_name.items()},
                "group_field": group_field,
                "min_group_count": min_count,
                "calibration_examples_per_group": cal_n,
                "n_groups": len(cal_by_group),
            }

    geometry_rows = []
    for min_count in args.min_group_counts:
        cal_idx, eval_idx, _cal_by_group, _eval_by_group = split_by_group(
            groups,
            min_count=min_count,
            calibration_examples=1,
            seed=args.seed,
        )
        indices = cal_idx + eval_idx
        if not indices:
            continue
        geometry = within_between_summary(residuals, groups, indices)
        subset = residuals.index_select(0, torch.tensor(indices, dtype=torch.long))
        max_rank = min(max(args.lowrank_ranks) if args.lowrank_ranks else 16, subset.shape[0], subset.shape[1])
        if max_rank <= 0:
            geometry_rows.append(
                {
                    "dataset": args.dataset_name,
                    "group_field": group_field or "",
                    "residual_kind": args.residual_kind,
                    "min_group_count": min_count,
                    "rank": 0,
                    "pca_cumulative_variance": "",
                    "within_group_residual_cosine": geometry["within_group_residual_cosine"],
                    "between_group_centroid_cosine": geometry["between_group_centroid_cosine"],
                    "n_groups": int(geometry["n_groups"]),
                    "n_examples": len(indices),
                }
            )
            continue
        _mean, _dirs, explained = pca_directions(subset, max_rank)
        for rank in sorted(set([1, 2, 4, 8, 16, max_rank])):
            if rank <= 0 or rank > len(explained):
                continue
            geometry_rows.append(
                {
                    "dataset": args.dataset_name,
                    "group_field": group_field or "",
                    "residual_kind": args.residual_kind,
                    "min_group_count": min_count,
                    "rank": rank,
                    "pca_cumulative_variance": float(explained[:rank].sum().item()),
                    "within_group_residual_cosine": geometry["within_group_residual_cosine"],
                    "between_group_centroid_cosine": geometry["between_group_centroid_cosine"],
                    "n_groups": int(geometry["n_groups"]),
                    "n_examples": len(indices),
                }
            )

    group_table = table_dir / f"{slug}_group_specific_residual_hidden.csv"
    aux_table = table_dir / f"{slug}_cluster_lowrank_residual_hidden.csv"
    geometry_table = table_dir / f"{slug}_residual_geometry_summary.csv"
    write_csv(group_table, rows)
    write_csv(aux_table, aux_rows)
    write_csv(geometry_table, geometry_rows)

    if "amazon" in slug:
        write_csv(table_dir / "amazon_user_specific_residual.csv", rows)
        write_csv(table_dir / "cluster_residual_lamp5_amazon_synper.csv", aux_rows)
    if "synper" in slug:
        write_csv(table_dir / "synper_persona_specific_residual.csv", rows)
    torch.save(
        {
            "model_type": "group_mean_delta_bank",
            "source_span_hidden": args.span_hidden,
            "dataset_name": args.dataset_name,
            "residual_kind": args.residual_kind,
            "settings": checkpoint_groups,
        },
        run_dir / f"{slug}_group_delta_bank.pt",
    )

    best_rows = sorted(
        [row for row in rows if row["method"] != "identity"],
        key=lambda row: row["hidden_mse"],
    )[:5]
    summary_lines = [
        f"# {args.dataset_name} Personalization Residual Summary",
        "",
        f"- Span hidden artifact: `{args.span_hidden}`",
        f"- Group field: `{group_field}`",
        f"- Residual kind: `{args.residual_kind}`",
        f"- Group table: `{group_table}`",
        f"- Cluster/low-rank table: `{aux_table}`",
        f"- Geometry table: `{geometry_table}`",
        "",
        "## Best Hidden Rows",
        "",
    ]
    for row in best_rows:
        summary_lines.append(
            f"- {row['method']} ({row['inference_setting']}), min={row['min_group_count']}, "
            f"cal={row['calibration_examples_per_group']}: MSE={row['hidden_mse']:.4f}, "
            f"gap={row['gap_recovery']:.3f}, cosine={row['cosine_to_target']:.3f}"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "These are hidden-space stored-vector/profile-free diagnostics, not downstream generation results. "
            "Use them to decide which group residual settings deserve GPU generation and judge evaluation.",
            "",
        ]
    )
    summary_path = analysis_dir / f"{slug}_user_group_residual_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    if "amazon" in slug:
        (analysis_dir / "user_group_residual_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    manifest_path = run_dir / f"{slug}_personalization_residual_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "span_hidden": args.span_hidden,
                "dataset_name": args.dataset_name,
                "group_field": group_field,
                "residual_kind": args.residual_kind,
                "tables": [str(group_table), str(aux_table), str(geometry_table)],
                "summary": str(summary_path),
                "checkpoint": str(run_dir / f"{slug}_group_delta_bank.pt"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"group_table": str(group_table), "aux_table": str(aux_table), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
