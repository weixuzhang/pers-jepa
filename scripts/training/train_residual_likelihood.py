#!/usr/bin/env python
"""Likelihood fine-tuning of a residual predictor through the frozen LM.

Objective (per example):  NLL( y | generic prompt, persistent injection of
delta = f(h_gen, w_user) at --layer )  +  aux_mse * || delta - r ||^2
where r is the answer-span residual target (from --span-hidden, matched by
example id) and the LM is frozen. Only the predictor's parameters receive
gradient; the LM forward is differentiable w.r.t. the injected delta.

Starts from an MSE-trained checkpoint (jepa_sae or routed_jepa_sae) and writes
a checkpoint of the same type, so generation/eval code is unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.intervention import load_causal_lm, load_steering_sae
from persjepa.persistent import PersistentSteerer, predictor_delta, target_nll
from persjepa.routing import answer_span_residuals, group_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="calibration jsonl (prompt/target/metadata)")
    ap.add_argument("--checkpoint", required=True, help="MSE-trained jepa_sae or routed_jepa_sae checkpoint")
    ap.add_argument("--output", required=True)
    ap.add_argument("--span-hidden", default=None, help="span-hidden file for the aux MSE residual targets")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--predictor-tokens", type=int, default=3)
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--aux-mse", type=float, default=0.1, help="weight of the residual-MSE auxiliary term (0 = pure likelihood)")
    ap.add_argument("--residual-scale", type=float, default=1.0, help="scale applied to delta during training injection")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-target-tokens", type=int, default=64)
    ap.add_argument("--freeze-offsets", action="store_true", help="routed: keep group offsets fixed, train experts only")
    ap.add_argument("--train-only-offsets", action="store_true", help="routed: train ONLY the per-group constants (control: is the per-example expert needed?)")
    ap.add_argument("--bridge-lora", type=int, default=0, help="rank of a shared LoRA on q/k/v/o trained jointly with the predictor (0 = frozen backbone; equal-budget composition with TAP-PER / PLUME)")
    ap.add_argument("--reinit-experts", choices=["none", "centroid"], default="none",
                    help="centroid: discard the MSE-pretrained expert weights (fresh random encoder shared across groups, zero decoder, "
                         "decoder bias = global mean residual, offsets kept) -> 'without predictor MSE pretraining, with centroid initialization'")
    ap.add_argument("--reinit-seed", type=int, default=None, help="seed of the fresh encoder init (default: --seed)")
    ap.add_argument("--top-k", type=int, default=None, help="override the experts' TopK (e.g. 128 = dense ReLU predictor); only meaningful with --reinit-experts")
    ap.add_argument("--constants-only", action="store_true",
                    help="checkpoint is a routed_mean_delta: train ONLY the K group vectors (genuine learned constants, prompt-independent by construction)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--system", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    cfg = ExperimentConfig()
    examples = load_persona_jsonl(
        args.input, profile_field=cfg.data.profile_field, prompt_field=cfg.data.text_field,
        target_field=cfg.data.target_field, id_field=cfg.data.id_field, profile_template=cfg.data.profile_template,
    )
    random.Random(args.seed).shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]

    # aux residual targets by example id
    residual_by_id: dict[str, torch.Tensor] = {}
    if args.span_hidden and args.aux_mse > 0:
        payload = torch.load(args.span_hidden, map_location="cpu")
        _, r = answer_span_residuals(payload)
        for rec, ri in zip(payload["examples"], r):
            residual_by_id[str(rec.get("id"))] = ri
        print(f"[likelihood] aux residual targets for {len(residual_by_id)} ids")

    model, tokenizer, device = load_causal_lm(args.model_name, pred_token=cfg.model.pred_token, device=args.device, dtype=args.dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = None
    if args.bridge_lora > 0:
        from persjepa.tapper import add_bridge_lora
        peft_model = add_bridge_lora(model, rank=args.bridge_lora)
        model = peft_model.base_model.model            # LoRA modules are injected in place; hooks resolve layers on the HF model
        lora_params = [p for n, p in peft_model.named_parameters() if "lora_" in n]
        for p in lora_params: p.requires_grad_(True)
        print(f"[likelihood] shared bridge LoRA r={args.bridge_lora}: {sum(p.numel() for p in lora_params)/1e6:.2f}M trainable")
    from persjepa.chat import apply_chat_formatting
    apply_chat_formatting(examples, tokenizer, args.model_name, args.chat_template, args.system)
    predictor, payload = load_steering_sae(args.checkpoint, device=device)
    if args.constants_only:
        from persjepa.models import RoutedConstants
        assert payload.get("model_type") == "routed_mean_delta", "--constants-only needs a routed_mean_delta checkpoint"
        table_obj = getattr(predictor, "routing_table", None)
        predictor = RoutedConstants(payload["group_means"], group_names=payload.get("group_names")).to(device)
        predictor.routing_table = table_obj
        print(f"[likelihood] constants-only control: {predictor.num_groups} trainable group vectors, prompt-independent")
    predictor.float().train()
    routed = hasattr(predictor, "set_weights")
    table = getattr(predictor, "routing_table", None)
    if args.reinit_experts == "centroid":
        assert hasattr(predictor, "experts"), "--reinit-experts needs a routed_jepa_sae checkpoint"
        assert args.span_hidden, "--reinit-experts needs --span-hidden (global mean residual for the decoder bias)"
        _pl = torch.load(args.span_hidden, map_location="cpu"); _, _r = answer_span_residuals(_pl); global_mean = _r.mean(0).float()
        g = torch.Generator().manual_seed(args.reinit_seed if args.reinit_seed is not None else args.seed)
        if args.top_k:
            predictor.top_k = int(args.top_k)
            for e in predictor.experts: e.top_k = int(args.top_k)
        fresh = torch.nn.Linear(predictor.input_dim, predictor.latent_dim)
        with torch.no_grad():
            bound = 1.0 / predictor.input_dim ** 0.5   # kaiming_uniform(a=sqrt(5)) on a Linear == U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            fresh.weight.uniform_(-bound, bound, generator=g); fresh.bias.zero_()
            for e in predictor.experts:
                e.encoder.weight.copy_(fresh.weight); e.encoder.bias.copy_(fresh.bias)   # one fresh init, copied across groups
                e.delta_decoder.weight.zero_()
                if e.delta_decoder.bias is not None: e.delta_decoder.bias.copy_(global_mean.to(e.delta_decoder.bias.device))
            # offsets stay (group mean - global mean): every expert initially predicts its group mean
        print(f"[likelihood] experts re-initialised (centroid init, top_k={predictor.top_k}); MSE pretraining discarded")
    if args.train_only_offsets:
        params = [p for n, p in predictor.named_parameters() if n == "group_offsets"]
        for n, p in predictor.named_parameters():
            p.requires_grad_(n == "group_offsets")
    else:
        params = [p for n, p in predictor.named_parameters() if not (args.freeze_offsets and n == "group_offsets")]
    if peft_model is not None:
        params = list(params) + lora_params
    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    print(f"[likelihood] trainable parameters: {n_trainable} ({n_trainable/1e6:.3f}M)")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    import time as _time; _t0 = _time.time()
    steerer = PersistentSteerer(model, tokenizer, predictor, layer=args.layer, pred_token=cfg.model.pred_token,
                                predictor_tokens=args.predictor_tokens, max_length=args.max_length, device=device)

    def encode_pair(ex):
        """prompt + target token ids, with a mask over target positions."""
        side = tokenizer.truncation_side; tokenizer.truncation_side = "left"
        try:
            p_ids = tokenizer(ex.generic_prompt, truncation=True, max_length=args.max_length - args.max_target_tokens)["input_ids"]
        finally:
            tokenizer.truncation_side = side
        t_ids = tokenizer(" " + ex.target.strip(), truncation=True, max_length=args.max_target_tokens, add_special_tokens=False)["input_ids"]
        t_ids = t_ids + [tokenizer.eos_token_id]
        ids = p_ids + t_ids
        tmask = [0] * len(p_ids) + [1] * len(t_ids)
        return ids, tmask

    def collate(batch):
        maxlen = max(len(ids) for ids, _ in batch)
        pad = tokenizer.pad_token_id
        input_ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        tmask = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, (ids, tm) in enumerate(batch):  # right padding
            input_ids[i, : len(ids)] = torch.tensor(ids); attn[i, : len(ids)] = 1; tmask[i, : len(ids)] = torch.tensor(tm)
        return input_ids.to(device), attn.to(device), tmask.to(device)

    step, log = 0, []
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(examples))); random.Random(args.seed + epoch).shuffle(order)
        pbar = tqdm(range(0, len(order), args.batch_size), desc=f"likelihood epoch {epoch}")
        for start in pbar:
            batch_ex = [examples[i] for i in order[start: start + args.batch_size]]
            # 1) anchor states (no grad through LM), 2) predictor deltas (grad)
            h = torch.cat([steerer.anchor_state(ex.generic_prompt) for ex in batch_ex]).float()
            if routed:
                ws = []
                for ex in batch_ex:
                    user = group_of({"metadata": ex.metadata or {}}, args.group_field)
                    w = table.get(user) if table is not None else None
                    ws.append(w if w is not None else torch.full((predictor.num_groups,), 1.0 / predictor.num_groups))
                delta = predictor(h, torch.stack(ws).to(device)).delta
            else:
                delta = predictor_delta(predictor, h)
            # 3) differentiable LM forward with persistent injection
            input_ids, attn, tmask = collate([encode_pair(ex) for ex in batch_ex])
            logits = steerer.forward_with_injection(input_ids, attn, (delta * args.residual_scale).to(next(model.parameters()).dtype))
            nll = target_nll(logits, input_ids, attn, tmask)
            loss = nll
            mse = torch.tensor(0.0, device=device)
            if residual_by_id and args.aux_mse > 0:
                targets = [residual_by_id.get(str(ex.id)) for ex in batch_ex]
                keep = [i for i, t in enumerate(targets) if t is not None]
                if keep:
                    tgt = torch.stack([targets[i] for i in keep]).to(device)
                    mse = ((delta[keep] - tgt) ** 2).mean()
                    loss = loss + args.aux_mse * mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                log.append({"step": step, "nll": float(nll), "aux_mse": float(mse), "loss": float(loss)})
                pbar.set_postfix(nll=f"{float(nll):.3f}", mse=f"{float(mse):.2f}")

    predictor.eval()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if routed:
        pay = predictor.export_payload()
        pay["routing_table"] = payload["routing_table"]
    else:
        pay = {**payload, "state_dict": predictor.state_dict()}
    pay["likelihood_training"] = {"args": vars(args), "log": log, "init_checkpoint": args.checkpoint,
                                  "n_trainable": int(n_trainable), "train_seconds": round(_time.time() - _t0, 1)}
    if peft_model is not None:
        from peft import get_peft_model_state_dict
        pay["lora_state_dict"] = {k: v.cpu() for k, v in get_peft_model_state_dict(peft_model).items()}; pay["lora_rank"] = args.bridge_lora
    torch.save(pay, out)
    print(f"[likelihood] wrote {out} after {step} steps; final nll={log[-1]['nll'] if log else float('nan'):.3f}")


if __name__ == "__main__":
    main()
