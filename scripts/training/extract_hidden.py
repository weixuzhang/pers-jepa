#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.hidden import HiddenStateExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Pers-JEPA hidden-state pairs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--predictor-tokens", type=int, default=None)
    parser.add_argument("--pred-token", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
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
    extractor = HiddenStateExtractor(
        model_name=args.model_name or cfg.model.name,
        pred_token=args.pred_token or cfg.model.pred_token,
        predictor_tokens=(
            args.predictor_tokens
            if args.predictor_tokens is not None
            else cfg.model.predictor_tokens
        ),
        layer=cfg.model.layer if args.layer is None else args.layer,
        max_length=args.max_length or cfg.model.max_length,
        device=args.device or cfg.model.device,
        dtype=args.dtype or cfg.model.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    extractor.extract_to_file(examples, args.output, batch_size=args.batch_size)
    print(f"Wrote hidden pairs for {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
