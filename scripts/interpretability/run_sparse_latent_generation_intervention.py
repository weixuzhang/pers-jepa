#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.data import PersonaExample, load_persona_jsonl
from persjepa.hidden import append_pred_tokens, require_torch, resolve_device, resolve_dtype
from persjepa.intervention import final_norm_module, load_causal_lm, load_jepa_sae


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "method",
    "model",
    "of",
    "on",
    "or",
    "paper",
    "system",
    "systems",
    "the",
    "their",
    "through",
    "to",
    "using",
    "via",
    "with",
}


def tokenize_words(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(str(text))
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def metric_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


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


def rouge_l_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    previous = [0] * (len(gold_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for idx, gold_token in enumerate(gold_tokens, start=1):
            if pred_token == gold_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(current[-1], previous[idx]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def text_metrics(output: str, target: str) -> dict[str, float]:
    pred = metric_tokens(output)
    gold = metric_tokens(target)
    return {
        "token_f1": f1_score(pred, gold),
        "rouge_l_f1": rouge_l_f1(pred, gold),
        "pred_tokens": float(len(pred)),
        "target_tokens": float(len(gold)),
    }


def short(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def parse_score(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value


def load_label_stats(
    *,
    label_corr_path: Path,
    top_examples_path: Path,
    latent_ids: list[int] | None,
    top_latents: int,
) -> dict[int, dict[str, Any]]:
    corr_rows = []
    if label_corr_path.exists():
        with label_corr_path.open(newline="", encoding="utf-8") as handle:
            corr_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("sae_type") == "jepa_sae"
            ]
    by_latent: dict[int, dict[str, Any]] = {}
    for row in corr_rows:
        score = parse_score(row.get("score", "nan"))
        if math.isnan(score):
            continue
        latent_id = int(row["latent_id"])
        current = by_latent.get(latent_id)
        if current is None or abs(score) > abs(current["score"]):
            by_latent[latent_id] = {
                "latent_id": latent_id,
                "label_field": row.get("label_field", ""),
                "metric": row.get("metric", ""),
                "score": score,
                "activation_frequency": parse_score(row.get("activation_frequency", "nan")),
                "mean_activation": parse_score(row.get("mean_activation", "nan")),
            }

    words_by_latent: dict[int, Counter[str]] = defaultdict(Counter)
    example_targets: dict[int, list[str]] = defaultdict(list)
    if top_examples_path.exists():
        with top_examples_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("sae_type") != "jepa_sae":
                    continue
                latent_id = int(row["latent_id"])
                if int(row.get("rank", 999)) > 20:
                    continue
                target = row.get("target_snippet", "")
                words_by_latent[latent_id].update(tokenize_words(target))
                if len(example_targets[latent_id]) < 3:
                    example_targets[latent_id].append(target)

    if latent_ids is None:
        candidates = sorted(
            by_latent.values(),
            key=lambda item: (
                -abs(item["score"]),
                -float(item.get("activation_frequency", 0.0)),
            ),
        )
        latent_ids = [int(item["latent_id"]) for item in candidates[:top_latents]]

    output: dict[int, dict[str, Any]] = {}
    for latent_id in latent_ids:
        stats = dict(by_latent.get(latent_id, {"latent_id": latent_id}))
        keywords = [word for word, _count in words_by_latent.get(latent_id, Counter()).most_common(4)]
        stats["suggested_label"] = " / ".join(keywords[:3]) if keywords else f"latent {latent_id}"
        stats["keywords"] = keywords
        stats["top_target_examples"] = example_targets.get(latent_id, [])
        output[latent_id] = stats
    return output


class LatentInterventionGenerator:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        jepa_sae,
        pred_token: str,
        predictor_tokens: int,
        layer: int,
        max_length: int,
        residual_scale: float,
        device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.jepa_sae = jepa_sae
        self.pred_token = pred_token
        self.predictor_tokens = predictor_tokens
        self.layer = layer
        self.max_length = max_length
        self.residual_scale = residual_scale
        self.device = device
        self.torch = require_torch()
        self.pred_token_id = tokenizer.convert_tokens_to_ids(pred_token)

    def tokenize(self, text: str):
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            return self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
        finally:
            self.tokenizer.truncation_side = original_side

    def anchor_positions(self, input_ids, attention_mask):
        if self.predictor_tokens == 0:
            return attention_mask.long().sum(dim=1) - 1
        is_pred = input_ids.eq(self.pred_token_id)
        if not self.torch.all(is_pred.any(dim=1)):
            raise ValueError("Prompt does not contain [PRED] tokens.")
        idx = self.torch.arange(input_ids.shape[1], device=input_ids.device)
        return idx.masked_fill(~is_pred, -1).max(dim=1).values

    def encode_latent(self, prompt: str):
        torch = self.torch
        prompt_with_pred = append_pred_tokens(prompt, self.pred_token, self.predictor_tokens)
        inputs = self.tokenize(prompt_with_pred)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[self.layer]
            positions = self.anchor_positions(inputs["input_ids"], inputs["attention_mask"])
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            h_gen = hidden[batch, positions]
            dtype_source = next(self.jepa_sae.parameters(), None)
            if dtype_source is None:
                dtype_source = next(self.jepa_sae.buffers())
            latent, _pre = self.jepa_sae.encode(h_gen.to(dtype=dtype_source.dtype))
        return latent[0].detach(), positions.detach(), inputs

    def generate_from_latent(
        self,
        *,
        inputs,
        positions,
        latent,
        max_new_tokens: int,
    ) -> str:
        torch = self.torch
        with torch.no_grad():
            delta = self.jepa_sae.delta_decoder(latent.unsqueeze(0)) * self.residual_scale
        applied = {"value": False}

        def add_delta(hidden):
            if applied["value"] or hidden.ndim != 3 or hidden.shape[1] <= int(positions.max()):
                return hidden
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            patched = hidden.clone()
            patched[batch, positions] = patched[batch, positions] + delta.to(hidden.dtype)
            applied["value"] = True
            return patched

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (add_delta(output[0]), *output[1:])
            return add_delta(output)

        if self.layer != -1:
            raise ValueError("This lightweight intervention script currently supports layer=-1.")
        handle = final_norm_module(self.model).register_forward_hook(hook)
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            handle.remove()
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def average(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(row[key] for row in rows) / len(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation-side JEPA-SAE single-latent interventions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--label-correlations", default="tables/sparse_latent_label_correlations_lamp5.csv")
    parser.add_argument("--top-examples", default="tables/sparse_latent_top_examples_lamp5.csv")
    parser.add_argument("--latent-ids", nargs="*", type=int, default=None)
    parser.add_argument("--top-latents", type=int, default=6)
    parser.add_argument("--examples-per-latent", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--predictor-tokens", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--amplify-factor", type=float, default=2.0)
    parser.add_argument("--swap-quantile", type=float, default=0.10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    predictor_tokens = args.predictor_tokens if args.predictor_tokens is not None else cfg.model.predictor_tokens
    layer = cfg.model.layer if args.layer is None else args.layer
    max_length = args.max_length or cfg.model.max_length
    device_name = args.device or cfg.model.device
    dtype = args.dtype or cfg.model.dtype

    examples = load_persona_jsonl(
        args.input,
        profile_field=cfg.data.profile_field,
        prompt_field=cfg.data.text_field,
        target_field=cfg.data.target_field,
        id_field=cfg.data.id_field,
        profile_template=cfg.data.profile_template,
    )[: args.max_examples]

    torch = require_torch()
    model, tokenizer, device = load_causal_lm(
        args.model_name or cfg.model.name,
        pred_token=cfg.model.pred_token,
        device=device_name,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    jepa_sae, payload = load_jepa_sae(args.checkpoint, device=device)
    generator = LatentInterventionGenerator(
        model=model,
        tokenizer=tokenizer,
        jepa_sae=jepa_sae,
        pred_token=cfg.model.pred_token,
        predictor_tokens=predictor_tokens,
        layer=layer,
        max_length=max_length,
        residual_scale=args.residual_scale,
        device=device,
    )

    label_stats = load_label_stats(
        label_corr_path=Path(args.label_correlations),
        top_examples_path=Path(args.top_examples),
        latent_ids=args.latent_ids,
        top_latents=args.top_latents,
    )
    latent_ids = list(label_stats)

    encoded: list[dict[str, Any]] = []
    for idx, example in enumerate(tqdm(examples, desc="encode_latents")):
        latent, positions, inputs = generator.encode_latent(example.generic_prompt)
        encoded.append(
            {
                "idx": idx,
                "example": example,
                "latent": latent.detach().cpu(),
                "positions": positions.detach().cpu(),
                "inputs": {key: value.detach().cpu() for key, value in inputs.items()},
            }
        )
    latent_matrix = torch.stack([item["latent"] for item in encoded], dim=0)
    low_values = {
        latent_id: float(torch.quantile(latent_matrix[:, latent_id].float(), args.swap_quantile).item())
        for latent_id in latent_ids
    }

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for latent_id in tqdm(latent_ids, desc="intervene_latents"):
            selected = torch.topk(
                latent_matrix[:, latent_id],
                k=min(args.examples_per_latent, latent_matrix.shape[0]),
            ).indices.tolist()
            for idx in selected:
                item = encoded[idx]
                example: PersonaExample = item["example"]
                latent = item["latent"].to(device)
                positions = item["positions"].to(device)
                inputs = {key: value.to(device) for key, value in item["inputs"].items()}
                variants = {
                    "base": latent,
                    "zero": latent.clone(),
                    "amplify": latent.clone(),
                    "swap_low": latent.clone(),
                }
                variants["zero"][latent_id] = 0.0
                variants["amplify"][latent_id] = variants["amplify"][latent_id] * args.amplify_factor
                variants["swap_low"][latent_id] = low_values[latent_id]
                for intervention, variant_latent in variants.items():
                    output = generator.generate_from_latent(
                        inputs=inputs,
                        positions=positions,
                        latent=variant_latent,
                        max_new_tokens=args.max_new_tokens,
                    )
                    metrics = text_metrics(output, example.target)
                    record = {
                        "latent_id": latent_id,
                        "suggested_label": label_stats[latent_id].get("suggested_label", f"latent {latent_id}"),
                        "intervention": intervention,
                        "example_id": str(example.id),
                        "activation": float(latent[latent_id].detach().float().cpu().item()),
                        "output": output,
                        "target": example.target,
                        "prompt_snippet": short(example.generic_prompt),
                        "profile_snippet": short(example.profile),
                        **metrics,
                    }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()

    by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        by_pair[(int(record["latent_id"]), str(record["example_id"]), record["intervention"])] = record

    summary_rows: list[dict[str, Any]] = []
    for latent_id in latent_ids:
        base_rows = [
            row
            for row in records
            if row["latent_id"] == latent_id and row["intervention"] == "base"
        ]
        base_f1 = average(base_rows, "token_f1")
        base_rl = average(base_rows, "rouge_l_f1")
        for intervention in ["base", "zero", "amplify", "swap_low"]:
            rows = [
                row
                for row in records
                if row["latent_id"] == latent_id and row["intervention"] == intervention
            ]
            stats = label_stats[latent_id]
            summary_rows.append(
                {
                    "latent_id": latent_id,
                    "suggested_label": stats.get("suggested_label", f"latent {latent_id}"),
                    "keywords": " / ".join(stats.get("keywords", [])),
                    "label_field": stats.get("label_field", ""),
                    "label_score": stats.get("score", float("nan")),
                    "activation_frequency": stats.get("activation_frequency", float("nan")),
                    "mean_activation": stats.get("mean_activation", float("nan")),
                    "intervention": intervention,
                    "n_examples": len(rows),
                    "token_f1": average(rows, "token_f1"),
                    "rouge_l_f1": average(rows, "rouge_l_f1"),
                    "pred_tokens": average(rows, "pred_tokens"),
                    "delta_token_f1_vs_base": average(rows, "token_f1") - base_f1,
                    "delta_rouge_l_vs_base": average(rows, "rouge_l_f1") - base_rl,
                    "checkpoint": args.checkpoint,
                    "latent_dim": payload.get("latent_dim"),
                    "top_k": payload.get("top_k"),
                    "residual_scale": args.residual_scale,
                    "examples_path": args.output_jsonl,
                }
            )
    write_csv(Path(args.summary_csv), summary_rows)
    labels_payload = {
        str(latent_id): {
            **label_stats[latent_id],
            "low_swap_value": low_values[latent_id],
        }
        for latent_id in latent_ids
    }
    Path(args.labels_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.labels_json).write_text(json.dumps(labels_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "summary_csv": args.summary_csv,
                "labels_json": args.labels_json,
                "latent_ids": latent_ids,
                "n_records": len(records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
