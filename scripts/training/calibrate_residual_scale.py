#!/usr/bin/env python
"""Automatic residual-scale calibration via a calibration-set
logit-space objective (teacher-forced next-token loss on the true answer,
under generic-prompt + scale*delta), instead of manually tuning scale on
downstream dev generation quality. Cheap: single forward passes, no
autoregressive generation needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.hidden import append_pred_tokens, require_torch
from persjepa.intervention import decoder_layer_module, final_norm_module, load_causal_lm


def answer_token_means(payload):
    torch = require_torch()
    source = payload["source_answer_tokens"].float()
    target = payload["target_answer_tokens"].float()
    mask = (payload["source_answer_token_mask"].bool() & payload["target_answer_token_mask"].bool()).float()
    denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return (source * mask.unsqueeze(-1)).sum(dim=1) / denom, (target * mask.unsqueeze(-1)).sum(dim=1) / denom


def hook_module(model, layer):
    if layer == -1:
        return final_norm_module(model)
    return decoder_layer_module(model, layer - 1)


def teacher_forced_loss(model, tokenizer, generic_prompt, target, delta, *, pred_token, predictor_tokens, layer, max_length, device):
    torch = require_torch()
    prompt_with_pred = append_pred_tokens(generic_prompt, pred_token, predictor_tokens)
    prompt_ids = tokenizer(prompt_with_pred, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    n_prompt_tokens = prompt_ids["input_ids"].shape[1]

    full_text = prompt_with_pred + "\nAnswer: " + target
    full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    if predictor_tokens == 0:
        anchor_pos = n_prompt_tokens - 1
    else:
        pred_id = tokenizer.convert_tokens_to_ids(pred_token)
        is_pred = input_ids[0].eq(pred_id)
        if not bool(is_pred.any().item()):
            return None
        anchor_pos = int(torch.arange(input_ids.shape[1], device=device).masked_fill(~is_pred, -1).max().item())

    if input_ids.shape[1] <= n_prompt_tokens:
        return None

    applied = {"value": False}

    def add_delta(hidden):
        if applied["value"] or hidden.ndim != 3 or hidden.shape[1] <= anchor_pos:
            return hidden
        patched = hidden.clone()
        patched[0, anchor_pos] = patched[0, anchor_pos] + delta.to(dtype=hidden.dtype, device=hidden.device)
        applied["value"] = True
        return patched

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (add_delta(output[0]), *output[1:])
        return add_delta(output)

    handle = hook_module(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
    finally:
        handle.remove()

    shift_logits = logits[0, :-1]
    shift_labels = input_ids[0, 1:]
    target_start = n_prompt_tokens  # first answer token index in input_ids
    losses = torch.nn.functional.cross_entropy(
        shift_logits[target_start - 1 :], shift_labels[target_start - 1 :], reduction="mean"
    )
    return float(losses.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--calibration-span-hidden", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--predictor-tokens", type=int, default=3)
    parser.add_argument("--pred-token", default="[PRED]")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch = require_torch()
    payload = torch.load(args.calibration_span_hidden, map_location="cpu")
    source, target = answer_token_means(payload)
    global_delta = (target - source).mean(dim=0)

    with open(args.calibration_jsonl, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle][: args.max_examples]

    model, tokenizer, device = load_causal_lm(
        args.model_name, pred_token=args.pred_token, device=args.device, dtype=args.dtype
    )
    model.eval()
    delta_device = global_delta.to(device=device)

    results = {}
    for scale in args.scales:
        losses = []
        for record in records:
            loss = teacher_forced_loss(
                model,
                tokenizer,
                record["prompt"],
                record["target"],
                delta_device * scale,
                pred_token=args.pred_token,
                predictor_tokens=args.predictor_tokens,
                layer=args.layer,
                max_length=args.max_length,
                device=device,
            )
            if loss is not None:
                losses.append(loss)
        mean_loss = sum(losses) / len(losses) if losses else float("nan")
        results[scale] = {"mean_teacher_forced_loss": mean_loss, "n": len(losses)}
        print(f"scale={scale}: mean_loss={mean_loss:.4f} (n={len(losses)})", flush=True)

    best_scale = min(results, key=lambda s: results[s]["mean_teacher_forced_loss"])
    output = {
        "calibration_jsonl": args.calibration_jsonl,
        "predictor_tokens": args.predictor_tokens,
        "results": results,
        "auto_calibrated_scale": best_scale,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Auto-calibrated scale: {best_scale}")


if __name__ == "__main__":
    main()
