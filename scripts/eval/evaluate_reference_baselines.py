#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.hidden import append_pred_tokens
from persjepa.intervention import ResidualSteerer, generate_text, load_causal_lm, load_steering_sae


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the standard prompt-side references: raw generic prompt, "
            "generic prompt plus predictor tokens, personalized prompt with "
            "profile/history included, and an optional profile-free steered method."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--predictor-tokens", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--reuse-baselines-from", default=None)
    parser.add_argument(
        "--profile-output-key",
        default="profile_prompt_output",
        help="Record key used for the personalized prompt with profile/history included.",
    )
    parser.add_argument("--jepa-checkpoint", default=None)
    parser.add_argument("--steered-output-key", default="jepa_steered_output")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file, skipping example ids already written.",
    )
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
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    reused_baselines: dict[str, dict] = {}
    if args.reuse_baselines_from:
        baseline_path = Path(args.reuse_baselines_from)
        with baseline_path.open("r", encoding="utf-8") as baseline_file:
            for line in baseline_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                reused_baselines[str(record.get("id", ""))] = record

    model, tokenizer, device = load_causal_lm(
        args.model_name or cfg.model.name,
        pred_token=cfg.model.pred_token,
        device=args.device or cfg.model.device,
        dtype=args.dtype or cfg.model.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    predictor_tokens = (
        args.predictor_tokens
        if args.predictor_tokens is not None
        else cfg.model.predictor_tokens
    )

    steerer = None
    if args.jepa_checkpoint:
        steering_sae, _payload = load_steering_sae(args.jepa_checkpoint, device=device)
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
        examples = [example for example in examples if str(example.id) not in completed_ids]
        print(f"Resuming {args.output}; skipping {len(completed_ids)} existing records")

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for example in tqdm(examples, desc="evaluate_reference_baselines"):
            generic = example.generic_prompt
            personalized = example.personalized_prompt
            with_pred = append_pred_tokens(generic, cfg.model.pred_token, predictor_tokens)
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

            if reused is not None and args.profile_output_key in reused:
                profile_output = reused[args.profile_output_key]
            else:
                profile_output = generate_text(
                    model,
                    tokenizer,
                    personalized,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )

            record = {
                **example.to_json(),
                "raw_generic_output": raw_output,
                "generic_pred_output": pred_baseline_output,
                args.profile_output_key: profile_output,
                "baselines": {
                    "raw_generic_prompt": generic,
                    "generic_prompt_with_pred": with_pred,
                    "personalized_prompt_with_profile": personalized,
                },
            }

            if steerer is not None:
                record[args.steered_output_key] = steerer.generate(
                    generic,
                    max_new_tokens=args.max_new_tokens,
                )
                record["baselines"]["residual_scale"] = args.residual_scale

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"Wrote reference baseline records to {args.output}")


if __name__ == "__main__":
    main()
