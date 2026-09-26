#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.hidden import append_pred_tokens, require_torch
from persjepa.intervention import decoder_layer_module, final_norm_module, generate_text, load_causal_lm


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


def example_group(example: dict[str, Any], group_field: str) -> str:
    value = dotted_get(example, group_field, None)
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


def stratified_example_split(groups: list[str], *, eval_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for indices in by_group.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if eval_fraction <= 0 or len(shuffled) <= 1:
            train_indices.extend(shuffled)
            continue
        n_eval = max(1, int(round(len(shuffled) * eval_fraction)))
        n_eval = min(n_eval, len(shuffled) - 1)
        eval_indices.extend(shuffled[:n_eval])
        train_indices.extend(shuffled[n_eval:])
    return train_indices, eval_indices


def batch_iter(seq: list[str], batch_size: int):
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def mean_pool_last_hidden(model, tokenizer, texts: list[str], *, device, max_length: int, batch_size: int):
    torch = require_torch()
    vectors = []
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        for chunk in tqdm(list(batch_iter(texts, batch_size)), desc="profile_embed", leave=False):
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].float()
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            vectors.append(pooled.cpu())
    finally:
        tokenizer.truncation_side = original_side
    return torch.cat(vectors, dim=0) if vectors else torch.empty(0)


def build_group_profile_vectors(
    examples: list[dict[str, Any]],
    *,
    model,
    tokenizer,
    device,
    group_field: str,
    profile_max_length: int,
    profile_batch_size: int,
    profile_items_per_group: int,
):
    grouped_profiles: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        group = example_group(example, group_field)
        profile = str(example.get("profile", "")).strip()
        if profile and len(grouped_profiles[group]) < profile_items_per_group:
            grouped_profiles[group].append(profile)
    groups = sorted(grouped_profiles)
    flat_texts: list[str] = []
    owners: list[str] = []
    for group in groups:
        for text in grouped_profiles[group]:
            flat_texts.append(text)
            owners.append(group)
    embeddings = mean_pool_last_hidden(
        model,
        tokenizer,
        flat_texts,
        device=device,
        max_length=profile_max_length,
        batch_size=profile_batch_size,
    )
    group_vectors = {}
    group_counts = defaultdict(int)
    for group, vector in zip(owners, embeddings):
        if group not in group_vectors:
            group_vectors[group] = vector.clone()
        else:
            group_vectors[group] += vector
        group_counts[group] += 1
    for group, count in group_counts.items():
        group_vectors[group] /= max(count, 1)
    return group_vectors, group_counts


class ProfileOnlyDecoder(require_torch().nn.Module):
    def __init__(self, profile_dim: int, hidden_dim: int) -> None:
        super().__init__()
        torch = require_torch()
        self.decoder = torch.nn.Linear(profile_dim, hidden_dim)

    def forward(self, profile_vec):
        return self.decoder(profile_vec)


class TaskConditionedDecoder(require_torch().nn.Module):
    def __init__(self, profile_dim: int, hidden_dim: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        torch = require_torch()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(profile_dim + hidden_dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, anchor_hidden, profile_vec):
        return self.net(require_torch().cat([anchor_hidden, profile_vec], dim=-1))


def train_decoder(
    model,
    train_inputs: tuple,
    train_target,
    eval_inputs: tuple,
    eval_target,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device,
):
    torch = require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_eval = float("inf")
    train_size = train_target.shape[0]
    indices = list(range(train_size))
    for _epoch in range(epochs):
        random.shuffle(indices)
        model.train()
        for start in range(0, train_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            batch_target = train_target.index_select(0, idx)
            if len(train_inputs) == 1:
                prediction = model(train_inputs[0].index_select(0, idx))
            else:
                prediction = model(
                    train_inputs[0].index_select(0, idx),
                    train_inputs[1].index_select(0, idx),
                )
            loss = torch.nn.functional.mse_loss(prediction.float(), batch_target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if len(eval_inputs) == 1:
                eval_pred = model(eval_inputs[0])
            else:
                eval_pred = model(eval_inputs[0], eval_inputs[1])
            eval_loss = float(torch.nn.functional.mse_loss(eval_pred.float(), eval_target.float()).item())
        if eval_loss < best_eval:
            best_eval = eval_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return best_eval


def hidden_metrics(source, target, delta):
    torch = require_torch()
    pred = source + delta
    baseline = float(torch.nn.functional.mse_loss(source.float(), target.float()).item())
    mse = float(torch.nn.functional.mse_loss(pred.float(), target.float()).item())
    cosine = float(torch.nn.functional.cosine_similarity(pred.float(), target.float(), dim=-1).mean().item())
    gap = (baseline - mse) / baseline if baseline else 0.0
    return {
        "hidden_mse": mse,
        "cosine_to_target": cosine,
        "gap_recovery": gap,
        "delta_norm": float(delta.float().norm(dim=-1).mean().item()),
        "target_delta_norm": float((target.float() - source.float()).norm(dim=-1).mean().item()),
        "n_eval": int(source.shape[0]),
    }


def calibrate_scalar(delta, target_delta) -> float:
    torch = require_torch()
    numerator = (delta.float() * target_delta.float()).sum()
    denominator = (delta.float() * delta.float()).sum().clamp_min(1e-12)
    alpha = float((numerator / denominator).item())
    return max(alpha, 0.0)


class DenseProfileSteerer:
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
        profile_only_model,
        task_conditioned_model,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pred_token = pred_token
        self.predictor_tokens = predictor_tokens
        self.layer = layer
        self.max_length = max_length
        self.device = device
        self.profile_only_model = profile_only_model
        self.task_conditioned_model = task_conditioned_model
        self.torch = require_torch()
        self.pred_token_id = tokenizer.convert_tokens_to_ids(pred_token)

    def tokenize(self, text: str):
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        finally:
            self.tokenizer.truncation_side = original_side

    def positions(self, input_ids, attention_mask):
        if self.predictor_tokens == 0:
            return attention_mask.long().sum(dim=1) - 1
        is_pred = input_ids.eq(self.pred_token_id)
        if not bool(is_pred.any().item()):
            raise ValueError("Prompt lost all [PRED] tokens after truncation.")
        idx = self.torch.arange(input_ids.shape[1], device=input_ids.device)
        return idx.masked_fill(~is_pred, -1).max(dim=1).values

    def _hook_module(self):
        if self.layer == -1:
            return final_norm_module(self.model)
        if self.layer <= 0:
            raise ValueError("Use layer=-1 or a positive hidden-state index.")
        return decoder_layer_module(self.model, self.layer - 1)

    def _compute_inputs(self, generic_prompt: str):
        prompt = append_pred_tokens(generic_prompt, self.pred_token, self.predictor_tokens)
        inputs = self.tokenize(prompt)
        positions = self.positions(inputs["input_ids"], inputs["attention_mask"])
        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[self.layer]
            batch = self.torch.arange(hidden.shape[0], device=hidden.device)
            anchor_hidden = hidden[batch, positions].float()
        return inputs, positions, anchor_hidden

    def _generate_with_delta(self, inputs, positions, delta, *, max_new_tokens: int):
        torch = self.torch
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

    def generate(self, generic_prompt: str, profile_vec, *, max_new_tokens: int):
        inputs, positions, anchor_hidden = self._compute_inputs(generic_prompt)
        with self.torch.no_grad():
            profile_delta = self.profile_only_model(profile_vec)
            task_delta = self.task_conditioned_model(anchor_hidden, profile_vec)
        return {
            "inputs": inputs,
            "positions": positions,
            "profile_delta": profile_delta,
            "task_correction": task_delta,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate dense profile-vector baselines.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-span-hidden", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-field", required=True)
    parser.add_argument("--residual-kind", default="answer_tokens", choices=["answer_tokens", "answer_mean", "anchor"])
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--profile-max-length", type=int, default=384)
    parser.add_argument("--profile-batch-size", type=int, default=16)
    parser.add_argument("--profile-items-per-group", type=int, default=5)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--train-epochs", type=int, default=25)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--profile-only-scale", type=float, default=1.0)
    parser.add_argument("--task-conditioned-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default=None)
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
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_path = checkpoint_dir / f"{output_path.stem}_dense_profile_vector_summary.json"
    profile_only_ckpt = checkpoint_dir / "dense_profile_vector_profile_only.pt"
    task_conditioned_ckpt = checkpoint_dir / "dense_profile_vector_task_conditioned.pt"

    model, tokenizer, device = load_causal_lm(
        args.model_name or cfg.model.name,
        pred_token=cfg.model.pred_token,
        device=args.device or cfg.model.device,
        dtype=args.dtype or cfg.model.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    predictor_tokens = args.predictor_tokens if args.predictor_tokens is not None else cfg.model.predictor_tokens
    layer = cfg.model.layer if args.layer is None else args.layer
    max_length = cfg.model.max_length if args.max_length is None else args.max_length
    torch = require_torch()

    payload = torch.load(args.train_span_hidden, map_location="cpu")
    train_examples = payload["examples"]
    train_groups = [example_group(example, args.group_field) for example in train_examples]
    source, target = source_target(payload, args.residual_kind)
    anchor_source = payload["source_anchor"].float()
    residual_target = target - source

    group_vectors, group_counts = build_group_profile_vectors(
        train_examples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        group_field=args.group_field,
        profile_max_length=args.profile_max_length,
        profile_batch_size=args.profile_batch_size,
        profile_items_per_group=args.profile_items_per_group,
    )
    profile_dim = next(iter(group_vectors.values())).shape[0]
    hidden_dim = residual_target.shape[-1]
    train_indices, eval_indices = stratified_example_split(train_groups, eval_fraction=args.eval_fraction, seed=args.seed)
    if not eval_indices:
        eval_indices = train_indices[: max(1, len(train_indices) // 10)]
        train_indices = train_indices[len(eval_indices) :]
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    eval_tensor = torch.tensor(eval_indices, dtype=torch.long)
    group_vec_tensor = torch.stack([group_vectors[group] for group in train_groups], dim=0).float()
    group_mean_residual = {}
    for group in sorted(set(train_groups)):
        group_idx = torch.tensor([idx for idx, value in enumerate(train_groups) if value == group], dtype=torch.long)
        group_mean_residual[group] = residual_target.index_select(0, group_idx).mean(dim=0)
    group_mean_tensor = torch.stack([group_mean_residual[group] for group in train_groups], dim=0).float()
    profile_train = group_vec_tensor.index_select(0, train_tensor).to(device)
    profile_eval = group_vec_tensor.index_select(0, eval_tensor).to(device)
    mean_residual_train = group_mean_tensor.index_select(0, train_tensor).to(device)
    mean_residual_eval = group_mean_tensor.index_select(0, eval_tensor).to(device)
    residual_train = residual_target.index_select(0, train_tensor).to(device)
    residual_eval = residual_target.index_select(0, eval_tensor).to(device)
    anchor_train = anchor_source.index_select(0, train_tensor).to(device)
    anchor_eval = anchor_source.index_select(0, eval_tensor).to(device)
    source_eval = source.index_select(0, eval_tensor).to(device)
    target_eval = target.index_select(0, eval_tensor).to(device)
    correction_train = residual_train - mean_residual_train
    correction_eval = residual_eval - mean_residual_eval

    profile_only_model = ProfileOnlyDecoder(profile_dim, hidden_dim)
    task_conditioned_model = TaskConditionedDecoder(profile_dim, hidden_dim, min(args.mlp_hidden_dim, hidden_dim))
    profile_only_eval = train_decoder(
        profile_only_model,
        (profile_train,),
        mean_residual_train,
        (profile_eval,),
        mean_residual_eval,
        epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    task_eval = train_decoder(
        task_conditioned_model,
        (anchor_train, profile_train),
        correction_train,
        (anchor_eval, profile_eval),
        correction_eval,
        epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    profile_only_model.eval()
    task_conditioned_model.eval()
    with torch.no_grad():
        profile_delta_eval = profile_only_model(profile_eval)
        task_delta_eval = profile_delta_eval + task_conditioned_model(anchor_eval, profile_eval)
    profile_only_alpha = calibrate_scalar(profile_delta_eval, residual_eval)
    task_conditioned_alpha = calibrate_scalar(task_delta_eval, residual_eval)
    profile_delta_eval_scaled = profile_delta_eval * profile_only_alpha
    task_delta_eval_scaled = task_delta_eval * task_conditioned_alpha
    summary = {
        "group_field": args.group_field,
        "residual_kind": args.residual_kind,
        "source_span_hidden": args.train_span_hidden,
        "profile_encoder_model": args.model_name or cfg.model.name,
        "profile_items_used": int(args.profile_items_per_group),
        "profile_embedding_dim": int(profile_dim),
        "profile_text_at_inference": False,
        "stored_latent": True,
        "uses_profile_semantics": True,
        "uses_behavioral_residual": False,
        "n_groups": len(group_vectors),
        "group_profile_counts": {group: int(count) for group, count in group_counts.items()},
        "profile_only": hidden_metrics(source_eval, target_eval, profile_delta_eval_scaled),
        "task_conditioned": hidden_metrics(source_eval, target_eval, task_delta_eval_scaled),
        "profile_only_alpha": profile_only_alpha,
        "task_conditioned_alpha": task_conditioned_alpha,
        "profile_only_eval_loss": profile_only_eval,
        "task_conditioned_eval_loss": task_eval,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_type": "dense_profile_vector_profile_only",
            "group_field": args.group_field,
            "profile_dim": profile_dim,
            "hidden_dim": hidden_dim,
            "state_dict": profile_only_model.cpu().state_dict(),
            "group_vectors": group_vectors,
            "summary": summary["profile_only"],
            "metadata": summary,
        },
        profile_only_ckpt,
    )
    torch.save(
        {
            "model_type": "dense_profile_vector_task_conditioned",
            "group_field": args.group_field,
            "profile_dim": profile_dim,
            "hidden_dim": hidden_dim,
            "mlp_hidden_dim": min(args.mlp_hidden_dim, hidden_dim),
            "state_dict": task_conditioned_model.cpu().state_dict(),
            "group_vectors": group_vectors,
            "summary": summary["task_conditioned"],
            "metadata": summary,
        },
        task_conditioned_ckpt,
    )
    profile_only_model.to(device)
    task_conditioned_model.to(device)

    steerer = DenseProfileSteerer(
        model,
        tokenizer,
        pred_token=cfg.model.pred_token,
        predictor_tokens=predictor_tokens,
        layer=layer,
        max_length=max_length,
        device=device,
        profile_only_model=profile_only_model,
        task_conditioned_model=task_conditioned_model,
    )
    reuse_by_id = load_jsonl_by_id(args.reuse_baselines_from)
    completed_ids = set(load_jsonl_by_id(str(output_path)).keys()) if args.resume else set()
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for example in tqdm(examples, desc="dense_profile_vector_eval"):
            if example.id in completed_ids:
                continue
            reused = reuse_by_id.get(example.id, {})
            generic_with_pred = append_pred_tokens(example.generic_prompt, cfg.model.pred_token, predictor_tokens)
            raw = reused.get("raw_generic_output")
            if raw is None:
                raw = generate_text(model, tokenizer, example.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
            pred = reused.get("generic_pred_output")
            if pred is None:
                pred = generate_text(model, tokenizer, generic_with_pred, device=device, max_new_tokens=args.max_new_tokens)
            personalized = reused.get("personalized_prompt_output")
            if personalized is None:
                personalized = generate_text(model, tokenizer, example.personalized_prompt, device=device, max_new_tokens=args.max_new_tokens)
            group = example_group(example.to_json(), args.group_field)
            profile_vec = group_vectors.get(group)
            if profile_vec is None:
                profile_vec = torch.stack(list(group_vectors.values()), dim=0).mean(dim=0)
            outputs = steerer.generate(
                example.generic_prompt,
                profile_vec.unsqueeze(0).to(device).float(),
                max_new_tokens=args.max_new_tokens,
            )
            profile_only_output = steerer._generate_with_delta(
                outputs["inputs"],
                outputs["positions"],
                outputs["profile_delta"] * (profile_only_alpha * args.profile_only_scale),
                max_new_tokens=args.max_new_tokens,
            )
            task_conditioned_output = steerer._generate_with_delta(
                outputs["inputs"],
                outputs["positions"],
                (outputs["profile_delta"] + outputs["task_correction"]) * (task_conditioned_alpha * args.task_conditioned_scale),
                max_new_tokens=args.max_new_tokens,
            )
            record = {
                **example.to_json(),
                "group_field": args.group_field,
                "group_id": group,
                "group_vector_available": group in group_vectors,
                "raw_generic_output": raw,
                "generic_pred_output": pred,
                "personalized_prompt_output": personalized,
                "dense_profile_vector_profile_only_output": profile_only_output,
                "dense_profile_vector_task_conditioned_output": task_conditioned_output,
                "inference_settings": {
                    "personalized_prompt_output": "profile_prompt_reference",
                    "dense_profile_vector_profile_only_output": "stored_latent_profile_free",
                    "dense_profile_vector_task_conditioned_output": "stored_latent_profile_free",
                },
                "dense_profile_vector_info": {
                    "encoder_name": args.model_name or cfg.model.name,
                    "profile_items_used": int(args.profile_items_per_group),
                    "profile_embedding_dim": int(profile_dim),
                    "profile_text_at_inference": False,
                    "stored_latent": True,
                    "uses_profile_semantics": True,
                    "uses_behavioral_residual": False,
                    "profile_only_alpha": profile_only_alpha,
                    "task_conditioned_alpha": task_conditioned_alpha,
                    "profile_only_scale": args.profile_only_scale,
                    "task_conditioned_scale": args.task_conditioned_scale,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    print(json.dumps({"output": str(output_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
