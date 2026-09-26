#!/usr/bin/env python
from __future__ import annotations

import argparse
import torch
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.hidden import append_pred_tokens
from persjepa.judge import build_pairwise_judge_prompt, judge_with_openai


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pers-JEPA with required double baselines.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--steered-output-key", default="jepa_steered_output")
    parser.add_argument("--group-field", default=None, help="dotted record field naming the user/persona for routed checkpoints")
    parser.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--system", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--predictor-tokens", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--reuse-baselines-from",
        default=None,
        help="Optional JSONL evaluation file to reuse raw_generic_output and generic_pred_output by id.",
    )
    parser.add_argument("--judge-provider", choices=["none", "openai"], default="none")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file, skipping example ids already present.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        examples = examples[:args.max_examples]

    reused_baselines: dict[str, dict] = {}
    if args.reuse_baselines_from:
        baseline_path = Path(args.reuse_baselines_from)
        with baseline_path.open("r", encoding="utf-8") as baseline_file:
            for line in baseline_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                reused_baselines[str(record.get("id", ""))] = record

    from persjepa.intervention import ResidualSteerer, generate_text, load_causal_lm, load_steering_sae

    model, tokenizer, device = load_causal_lm(
        args.model_name or cfg.model.name,
        pred_token=cfg.model.pred_token,
        device=args.device or cfg.model.device,
        dtype=args.dtype or cfg.model.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    from persjepa.chat import apply_chat_formatting
    apply_chat_formatting(examples, tokenizer, args.model_name or cfg.model.name, args.chat_template, args.system)
    steering_sae, _payload = load_steering_sae(args.jepa_checkpoint, device=device)
    predictor_tokens = (
        args.predictor_tokens
        if args.predictor_tokens is not None
        else cfg.model.predictor_tokens
    )
    steerer = ResidualSteerer(
        model,
        tokenizer,
        steering_sae,
        pred_token=cfg.model.pred_token,
        predictor_tokens=predictor_tokens,
        layer=cfg.model.layer if args.layer is None else args.layer,
        max_length=cfg.model.max_length if args.max_length is None else args.max_length,
        residual_scale=args.residual_scale,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    completed_ids: set[str] = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                completed_ids.add(str(json.loads(line).get("id", "")))
        examples = [
            example
            for example in examples
            if str(example.id) not in completed_ids
        ]
        print(f"Resuming {args.output}; skipping {len(completed_ids)} existing records")
    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8", buffering=1) as handle:
        for example in tqdm(examples, desc="evaluate"):
            generic = example.generic_prompt
            with_pred = append_pred_tokens(
                generic,
                cfg.model.pred_token,
                predictor_tokens,
            )
            reused = reused_baselines.get(str(example.id))
            if reused is not None and "raw_generic_output" in reused:
                raw_output = reused["raw_generic_output"]
            else:
                raw_output = generate_text(
                    model,
                    tokenizer,
                    generic,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            if reused is not None and "generic_pred_output" in reused:
                pred_baseline_output = reused["generic_pred_output"]
            elif with_pred == generic:
                pred_baseline_output = raw_output
            else:
                pred_baseline_output = generate_text(
                    model,
                    tokenizer,
                    with_pred,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            if hasattr(steering_sae, "set_weights"):
                from persjepa.routing import group_of
                _table = getattr(steering_sae, "routing_table", None)
                _user = group_of({"metadata": example.metadata or {}}, args.group_field)
                _w = _table.get(_user) if _table is not None else None
                if _w is None and _table is not None:
                    _w = torch.full((_table.num_groups,), 1.0 / _table.num_groups)
                steering_sae.set_weights(_w)
            steered_output = steerer.generate(
                generic,
                max_new_tokens=args.max_new_tokens,
            )
            record = {
                **example.to_json(),
                "raw_generic_output": raw_output,
                "generic_pred_output": pred_baseline_output,
                args.steered_output_key: steered_output,
                "baselines": {
                    "raw_generic_prompt": generic,
                    "generic_prompt_with_pred": with_pred,
                    "residual_scale": args.residual_scale,
                },
            }
            prompt_raw_vs_steered = build_pairwise_judge_prompt(
                record,
                candidate_a=raw_output,
                candidate_b=steered_output,
            )
            prompt_pred_vs_steered = build_pairwise_judge_prompt(
                record,
                candidate_a=pred_baseline_output,
                candidate_b=steered_output,
            )
            record["judge_payloads"] = {
                "raw_generic_vs_jepa": prompt_raw_vs_steered,
                "generic_pred_vs_jepa": prompt_pred_vs_steered,
            }
            if args.judge_provider == "openai":
                record["judge_results"] = {
                    "raw_generic_vs_jepa": judge_with_openai(
                        prompt_raw_vs_steered,
                        model=args.judge_model,
                    ),
                    "generic_pred_vs_jepa": judge_with_openai(
                        prompt_pred_vs_steered,
                        model=args.judge_model,
                    ),
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Wrote evaluation records to {args.output}")


if __name__ == "__main__":
    main()
