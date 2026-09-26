#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.hidden import append_pred_tokens, require_torch
from persjepa.intervention import generate_text, load_causal_lm


def dotted_get(record: dict[str, Any] | None, field: str, default: Any = None) -> Any:
    value: Any = record or {}
    for part in field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue
        while isinstance(value, dict) and isinstance(value.get("metadata"), dict):
            value = value["metadata"]
            if part in value:
                value = value[part]
                break
        else:
            return default
    return value


def example_group(example, group_field: str) -> str:
    payload = example.to_json()
    value = dotted_get(payload, group_field, None)
    if value in (None, "") and example.metadata:
        value = dotted_get(example.metadata, group_field.removeprefix("metadata."), None)
    return str(value) if value not in (None, "") else "__missing__"


def answer_token_example_means(payload: dict[str, Any]):
    torch = require_torch()
    source = payload["source_answer_tokens"].float()
    target = payload["target_answer_tokens"].float()
    mask = (payload["source_answer_token_mask"].bool() & payload["target_answer_token_mask"].bool()).float()
    denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return (source * mask.unsqueeze(-1)).sum(dim=1) / denom, (target * mask.unsqueeze(-1)).sum(dim=1) / denom


def source_target(payload: dict[str, Any], residual_kind: str):
    if residual_kind == "answer_tokens":
        return answer_token_example_means(payload)
    if residual_kind == "answer_mean":
        return payload["source_answer_span_mean"].float(), payload["target_answer_span_mean"].float()
    if residual_kind == "anchor":
        return payload["source_anchor"].float(), payload["target_anchor"].float()
    raise ValueError(f"Unsupported residual kind: {residual_kind}")


def train_group_deltas(span_hidden: str, group_field: str, residual_kind: str, min_group_count: int):
    torch = require_torch()
    payload = torch.load(span_hidden, map_location="cpu")
    examples = payload["examples"]
    source, target = source_target(payload, residual_kind)
    residuals = target - source
    groups = []
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        group = str(dotted_get(example, group_field, None) or dotted_get(example.get("metadata", {}), group_field.removeprefix("metadata."), "__missing__"))
        groups.append(group)
        by_group[group].append(idx)
    global_delta = residuals.mean(dim=0)
    group_deltas = {}
    for group, indices in by_group.items():
        if len(indices) >= min_group_count:
            group_deltas[group] = residuals.index_select(0, torch.tensor(indices, dtype=torch.long)).mean(dim=0)
    return global_delta, group_deltas


def build_shuffled_group_map(groups: list[str], *, seed: int) -> dict[str, str]:
    unique = sorted(set(groups))
    if len(unique) <= 1:
        return {group: group for group in unique}
    rng = random.Random(seed)
    shuffled = list(unique)
    while True:
        rng.shuffle(shuffled)
        if all(src != dst for src, dst in zip(unique, shuffled)):
            break
    return dict(zip(unique, shuffled))


def load_jsonl_by_id(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    records: dict[str, dict[str, Any]] = {}
    input_path = Path(path)
    if not input_path.exists():
        return records
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = str(record.get("id", "")).strip()
            if record_id:
                records[record_id] = record
    return records


class FixedResidualSteerer:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        pred_token: str,
        predictor_tokens: int,
        layer: int,
        max_length: int,
        device,
    ) -> None:
        from persjepa.intervention import decoder_layer_module, final_norm_module

        self.model = model
        self.tokenizer = tokenizer
        self.pred_token = pred_token
        self.predictor_tokens = predictor_tokens
        self.layer = layer
        self.max_length = max_length
        self.device = device
        self.torch = require_torch()
        self.pred_token_id = tokenizer.convert_tokens_to_ids(pred_token)
        self.decoder_layer_module = decoder_layer_module
        self.final_norm_module = final_norm_module

    def tokenize(self, text: str):
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        finally:
            self.tokenizer.truncation_side = original_side

    def _hook_module(self):
        if self.layer == -1:
            return self.final_norm_module(self.model)
        if self.layer <= 0:
            raise ValueError("Use layer=-1 or a positive hidden-state index.")
        return self.decoder_layer_module(self.model, self.layer - 1)

    def positions(self, input_ids, attention_mask):
        if self.predictor_tokens == 0:
            return attention_mask.long().sum(dim=1) - 1
        is_pred = input_ids.eq(self.pred_token_id)
        if not bool(is_pred.any().item()):
            raise ValueError("Prompt lost all [PRED] tokens after truncation.")
        idx = self.torch.arange(input_ids.shape[1], device=input_ids.device)
        return idx.masked_fill(~is_pred, -1).max(dim=1).values

    def generate(self, generic_prompt: str, delta, *, max_new_tokens: int) -> str:
        torch = self.torch
        prompt = append_pred_tokens(generic_prompt, self.pred_token, self.predictor_tokens)
        inputs = self.tokenize(prompt)
        positions = self.positions(inputs["input_ids"], inputs["attention_mask"])
        delta = delta.to(device=self.device)
        applied = {"value": False}

        def add_delta(hidden):
            if applied["value"] or hidden.ndim != 3 or hidden.shape[1] <= int(positions.max()):
                return hidden
            patched = hidden.clone()
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            patched[batch, positions] = patched[batch, positions] + delta.to(dtype=hidden.dtype)
            applied["value"] = True
            return patched

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (add_delta(output[0]), *output[1:])
            return add_delta(output)

        handle = self._hook_module().register_forward_hook(hook)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with global and stored-vector group residuals.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-span-hidden", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-field", default="metadata.user_id")
    parser.add_argument("--train-group-field", default=None)
    parser.add_argument("--eval-group-field", default=None)
    parser.add_argument("--residual-kind", default="answer_tokens", choices=["answer_tokens", "answer_mean", "anchor"])
    parser.add_argument("--min-group-count", type=int, default=3)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--group-scale", type=float, default=1.0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--predictor-tokens", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--reuse-baselines-from", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-groups", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffled-output-key", default="shuffled_group_answer_token_mean_delta_output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    examples = load_persona_jsonl(
        args.input,
        profile_field=cfg.data.profile_field,
        prompt_field=cfg.data.text_field,
        target_field=cfg.data.target_field,
        id_field=cfg.data.id_field,
        profile_template=cfg.data.profile_template,
    )
    random.Random(args.seed).shuffle(examples)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    train_group_field = args.train_group_field or args.group_field
    eval_group_field = args.eval_group_field or args.group_field
    global_delta, group_deltas = train_group_deltas(
        args.train_span_hidden,
        train_group_field,
        args.residual_kind,
        args.min_group_count,
    )
    shuffled_group_map = build_shuffled_group_map(list(group_deltas.keys()), seed=args.shuffle_seed) if args.shuffle_groups else {}
    model, tokenizer, device = load_causal_lm(
        args.model_name or cfg.model.name,
        pred_token=cfg.model.pred_token,
        device=args.device or cfg.model.device,
        dtype=args.dtype or cfg.model.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    predictor_tokens = args.predictor_tokens if args.predictor_tokens is not None else cfg.model.predictor_tokens
    steerer = FixedResidualSteerer(
        model,
        tokenizer,
        pred_token=cfg.model.pred_token,
        predictor_tokens=predictor_tokens,
        layer=cfg.model.layer if args.layer is None else args.layer,
        max_length=cfg.model.max_length if args.max_length is None else args.max_length,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reuse_by_id = load_jsonl_by_id(args.reuse_baselines_from)
    completed_ids = set(load_jsonl_by_id(str(output_path)).keys()) if args.resume else set()
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for example in tqdm(examples, desc="group_residual_eval"):
            if example.id in completed_ids:
                continue
            group = example_group(example, eval_group_field)
            group_delta = group_deltas.get(group, global_delta)
            shuffled_group = shuffled_group_map.get(group)
            shuffled_delta = group_deltas.get(shuffled_group, global_delta) if shuffled_group is not None else None
            generic_with_pred = append_pred_tokens(example.generic_prompt, cfg.model.pred_token, predictor_tokens)
            reused = reuse_by_id.get(example.id, {})
            raw = reused.get("raw_generic_output")
            if raw is None:
                raw = generate_text(model, tokenizer, example.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
            pred = reused.get("generic_pred_output")
            if pred is None:
                pred = generate_text(model, tokenizer, generic_with_pred, device=device, max_new_tokens=args.max_new_tokens)
            personalized = reused.get("personalized_prompt_output")
            if personalized is None:
                personalized = generate_text(model, tokenizer, example.personalized_prompt, device=device, max_new_tokens=args.max_new_tokens)
            global_out = steerer.generate(example.generic_prompt, args.global_scale * global_delta, max_new_tokens=args.max_new_tokens)
            group_out = steerer.generate(example.generic_prompt, args.group_scale * group_delta, max_new_tokens=args.max_new_tokens)
            record = {
                **example.to_json(),
                "group_field": eval_group_field,
                "train_group_field": train_group_field,
                "group_id": group,
                "group_residual_available": group in group_deltas,
                "raw_generic_output": raw,
                "generic_pred_output": pred,
                "personalized_prompt_output": personalized,
                "global_answer_token_mean_delta_output": global_out,
                "group_answer_token_mean_delta_output": group_out,
                "inference_settings": {
                    "global_answer_token_mean_delta_output": "strict_profile_free",
                    "group_answer_token_mean_delta_output": "stored_vector_profile_free",
                    "personalized_prompt_output": "profile_prompt_reference",
                },
            }
            if args.shuffle_groups:
                record["shuffled_group_id"] = shuffled_group
                record["shuffled_group_residual_available"] = shuffled_group in group_deltas if shuffled_group is not None else False
                record[args.shuffled_output_key] = steerer.generate(
                    example.generic_prompt,
                    args.group_scale * (shuffled_delta if shuffled_delta is not None else global_delta),
                    max_new_tokens=args.max_new_tokens,
                )
                record["inference_settings"][args.shuffled_output_key] = "stored_latent_negative_control"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Wrote group residual generations to {output_path}")


if __name__ == "__main__":
    main()
