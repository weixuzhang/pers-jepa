#!/usr/bin/env python
"""Generate with persistent (all-position, middle-layer) residual injection.

Works with every steering checkpoint type (mean_delta, jepa_sae,
routed_jepa_sae, cluster_mean_delta, ...). For per-example predictors the
residual is predicted once from the generic prompt's [PRED]-anchored state at
--layer, then added at every position during the prompt pass and decoding.
Routed checkpoints route each example by its user/persona identity
(--group-field) through the checkpoint's routing table; unknown users get
uniform weights unless --cold-start-from provides residual examples.

Writes evaluate.py-compatible jsonl (raw_generic_output + steered key), so the
usual scorers (postprocess_generation, evaluate_text_metrics, persona
classifiers) apply unchanged.
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
from persjepa.intervention import generate_text, load_causal_lm, load_steering_sae
from persjepa.persistent import PersistentSteerer
from persjepa.routing import group_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steered-output-key", default="jepa_steered_output")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14, help="hidden-state index; hooks decoder block layer-1")
    ap.add_argument("--predictor-tokens", type=int, default=3)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-baselines-from", default=None)
    ap.add_argument("--resume", action="store_true", help="keep valid lines already in --output and append the missing examples")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--system", default=None)
    ap.add_argument("--fints-topk", type=int, default=5)
    ap.add_argument("--rag-topk", type=int, default=5)
    ap.add_argument("--rag-exclude-target", choices=["on", "off"], default="on",
                    help="drop history items containing the dev target (leak guard); off for label tasks (LaMP-2), where every prompt lists every tag and the guard empties the history")
    ap.add_argument("--calib", default=None, help="calibration jsonl (history for the rag_bm25 reference arm)")
    ap.add_argument("--fints-weighting", choices=["attn", "mean"], default="attn")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    examples = load_persona_jsonl(
        args.input, profile_field=cfg.data.profile_field, prompt_field=cfg.data.text_field,
        target_field=cfg.data.target_field, id_field=cfg.data.id_field, profile_template=cfg.data.profile_template,
    )
    random.Random(args.seed).shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]

    done_ids: set[str] = set()
    if args.resume and Path(args.output).exists():
        valid = []
        for line in open(args.output, encoding="utf-8"):
            try:
                rec = json.loads(line); valid.append(line if line.endswith("\n") else line + "\n"); done_ids.add(str(rec.get("id", "")))
            except json.JSONDecodeError:
                pass  # truncated last line from a killed process
        with open(args.output, "w", encoding="utf-8") as f: f.writelines(valid)
        examples = [ex for ex in examples if str(ex.id) not in done_ids]
        print(f"[resume] {len(done_ids)} examples already in {args.output}; {len(examples)} to go")
    out_mode = "a" if done_ids else "w"
    reused = {}
    if args.reuse_baselines_from and Path(args.reuse_baselines_from).exists():
        for line in open(args.reuse_baselines_from):
            if line.strip():
                rec = json.loads(line)
                reused[str(rec.get("id", ""))] = rec

    model, tokenizer, device = load_causal_lm(
        args.model_name, pred_token=cfg.model.pred_token, device=args.device, dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    from persjepa.chat import apply_chat_formatting, render_chat_prompt, wants_chat_template
    raw_prompts = {str(ex.id): ex.generic_prompt for ex in examples}
    chat_on = apply_chat_formatting(examples, tokenizer, args.model_name, args.chat_template, args.system)
    if args.checkpoint == "rag_bm25":
        # Reference arm (B6): BM25 top-k of the user's history in the prompt (LaMP protocol). Uses text history, not profile-free.
        from persjepa.data import read_jsonl
        from persjepa.rag import HistoryIndex, rag_prompt, raw_profile_examples
        hist = HistoryIndex.from_calibration(read_jsonl(args.calib), args.group_field) if args.calib else HistoryIndex()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True); n_empty = 0
        with open(args.output, out_mode, encoding="utf-8", buffering=1) as f:
            for ex in tqdm(examples, desc="rag_bm25"):
                base = reused.get(str(ex.id), {}); q = raw_prompts[str(ex.id)]
                raw = base.get("raw_generic_output") or generate_text(model, tokenizer, ex.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
                shots = raw_profile_examples(ex.metadata, q, args.rag_topk)
                if shots is None:
                    shots = hist.retrieve(group_of({"metadata": ex.metadata or {}}, args.group_field), q, args.rag_topk, exclude_text=(ex.target.strip()[:80] or None) if args.rag_exclude_target == "on" else None)
                n_empty += (not shots)
                text = rag_prompt(q, shots)
                text = render_chat_prompt(tokenizer, text, system=args.system) if chat_on else text
                out = generate_text(model, tokenizer, text, device=device, max_new_tokens=args.max_new_tokens)
                f.write(json.dumps({"id": ex.id, "generic_prompt": ex.generic_prompt, "profile": ex.profile, "target": ex.target,
                                    "metadata": ex.metadata or base.get("metadata"), "raw_generic_output": raw,
                                    args.steered_output_key: out, "rag_prompt": text}, ensure_ascii=False) + "\n")
        if n_empty: print(f"[rag] {n_empty} examples had no retrievable history (plain prompt)")
        print("wrote", args.output); return
    if args.checkpoint == "profile_prompt":
        # Reference arm: no injection; the profile text is in the prompt (upper reference, not profile-free).
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, out_mode, encoding="utf-8", buffering=1) as f:
            for ex in tqdm(examples, desc="profile_prompt"):
                base = reused.get(str(ex.id), {})
                raw = base.get("raw_generic_output") or generate_text(model, tokenizer, ex.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
                prof = generate_text(model, tokenizer, ex.personalized_prompt, device=device, max_new_tokens=args.max_new_tokens)
                f.write(json.dumps({"id": ex.id, "generic_prompt": ex.generic_prompt, "profile": ex.profile, "target": ex.target,
                                    "metadata": ex.metadata or base.get("metadata"), "raw_generic_output": raw,
                                    args.steered_output_key: prof, "personalized_prompt_output": prof}, ensure_ascii=False) + "\n")
        print("wrote", args.output); return
    _pay = torch.load(args.checkpoint, map_location="cpu")
    if _pay.get("model_type") == "fints_store":
        from persjepa.fints import FinTSStore, FinTSSteerer
        store = FinTSStore.from_payload(_pay)
        fs = FinTSSteerer(model, tokenizer, store, layer=args.layer, top_k=args.fints_topk, alpha=args.residual_scale,
                          weighting=args.fints_weighting, max_length=args.max_length, device=device)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True); n_unknown = 0
        with open(args.output, out_mode, encoding="utf-8", buffering=1) as f:
            for ex in tqdm(examples, desc="fints"):
                base = reused.get(str(ex.id), {})
                raw = base.get("raw_generic_output") or generate_text(model, tokenizer, ex.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
                user = group_of({"metadata": ex.metadata or {}}, args.group_field)
                steered, ok = fs.generate(user, ex.generic_prompt, max_new_tokens=args.max_new_tokens); n_unknown += (not ok)
                f.write(json.dumps({"id": ex.id, "generic_prompt": ex.generic_prompt, "profile": ex.profile, "target": ex.target,
                                    "metadata": ex.metadata or base.get("metadata"), "raw_generic_output": raw,
                                    args.steered_output_key: steered}, ensure_ascii=False) + "\n")
        if n_unknown: print(f"[fints] {n_unknown} examples had users absent from the store (unsteered)")
        print("wrote", args.output); return
    if _pay.get("model_type") == "tapper":
        from peft import set_peft_model_state_dict
        from persjepa.tapper import TapperPrefix, add_bridge_lora, build_inputs_with_prefix, embed_text_mean
        prefix = TapperPrefix.from_payload(_pay).to(device).eval()
        if not _pay.get("prompt_only"):
            model = add_bridge_lora(model, rank=int(_pay.get("lora_rank", 8)))
            set_peft_model_state_dict(model, _pay["lora_state_dict"]); model.eval()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True); n_unknown = 0
        with open(args.output, out_mode, encoding="utf-8", buffering=1) as f:
            for ex in tqdm(examples, desc="tapper"):
                base = reused.get(str(ex.id), {})
                raw = base.get("raw_generic_output") or generate_text(model if _pay.get("prompt_only") else (model.base_model.model if hasattr(model, "base_model") else model), tokenizer, ex.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
                user = group_of({"metadata": ex.metadata or {}}, args.group_field); n_unknown += user not in prefix.user_index
                with torch.no_grad():
                    if _pay.get("prompt_only"):
                        pre = prefix.user_prefix(user).to(model.get_input_embeddings().weight.dtype)
                    elif _pay.get("lora_only"):
                        pre = torch.zeros(0, prefix.d, device=device)
                    else:
                        pre = prefix.prefixes(user, embed_text_mean(model, tokenizer, ex.generic_prompt, device=device))
                    side = tokenizer.truncation_side; tokenizer.truncation_side = "left"
                    try: inputs = tokenizer(ex.generic_prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
                    finally: tokenizer.truncation_side = side
                    emb, attn = build_inputs_with_prefix(model, pre, inputs["input_ids"], inputs["attention_mask"])
                    out = model.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
                steered = tokenizer.decode(out[0], skip_special_tokens=True).strip()  # generate with inputs_embeds returns only new tokens
                f.write(json.dumps({"id": ex.id, "generic_prompt": ex.generic_prompt, "profile": ex.profile, "target": ex.target,
                                    "metadata": ex.metadata or base.get("metadata"), "raw_generic_output": raw,
                                    args.steered_output_key: steered}, ensure_ascii=False) + "\n")
        if n_unknown: print(f"[tapper] {n_unknown} examples had users absent from the prefix table (mean user state)")
        print("wrote", args.output); return
    if _pay.get("model_type") == "plume":
        from persjepa.plume import PlumeAdapter, attach_plume
        adapter = PlumeAdapter.from_payload(_pay).to(device).eval()
        attach_plume(model, adapter, tuple(_pay.get("targets") or ()))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True); n_unknown = 0
        with open(args.output, out_mode, encoding="utf-8", buffering=1) as f:
            for ex in tqdm(examples, desc="plume"):
                base = reused.get(str(ex.id), {})
                raw = base.get("raw_generic_output")
                if raw is None:
                    adapter.enabled = False
                    raw = generate_text(model, tokenizer, ex.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
                    adapter.enabled = True
                user = group_of({"metadata": ex.metadata or {}}, args.group_field)
                n_unknown += (not adapter.set_user(user))
                steered = generate_text(model, tokenizer, ex.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
                f.write(json.dumps({"id": ex.id, "generic_prompt": ex.generic_prompt, "profile": ex.profile, "target": ex.target,
                                    "metadata": ex.metadata or base.get("metadata"), "raw_generic_output": raw,
                                    args.steered_output_key: steered}, ensure_ascii=False) + "\n")
        if n_unknown: print(f"[plume] {n_unknown} examples had users absent from the adapter (shared task adapter only)")
        print("wrote", args.output); return
    if "lora_state_dict" in _pay:   # routed SAE + shared bridge LoRA (equal-budget composition)
        from peft import set_peft_model_state_dict
        from persjepa.tapper import add_bridge_lora
        _peft = add_bridge_lora(model, rank=int(_pay.get("lora_rank", 8)))
        set_peft_model_state_dict(_peft, _pay["lora_state_dict"]); model = _peft.base_model.model; model.eval()
        print(f"[lora] loaded shared bridge LoRA r={_pay.get('lora_rank', 8)} from the checkpoint")
    del _pay
    predictor, payload = load_steering_sae(args.checkpoint, device=device)
    routed = hasattr(predictor, "set_weights")
    table = getattr(predictor, "routing_table", None)
    steerer = PersistentSteerer(
        model, tokenizer, predictor, layer=args.layer, pred_token=cfg.model.pred_token,
        predictor_tokens=args.predictor_tokens, max_length=args.max_length, residual_scale=args.residual_scale,
        device=device,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_unknown = 0
    with open(args.output, out_mode, encoding="utf-8", buffering=1) as f:
        for ex in tqdm(examples, desc="persistent_steer"):
            base = reused.get(str(ex.id), {})
            raw = base.get("raw_generic_output") or generate_text(
                model, tokenizer, ex.generic_prompt, device=device, max_new_tokens=args.max_new_tokens)
            if routed:
                user = group_of({"metadata": ex.metadata or {}}, args.group_field)
                w = table.get(user) if table is not None else None
                if w is None:
                    n_unknown += 1
                    w = torch.full((predictor.num_groups,), 1.0 / predictor.num_groups)
                predictor.set_weights(w.to(device))
            steered = steerer.generate(ex.generic_prompt, max_new_tokens=args.max_new_tokens)
            f.write(json.dumps({
                "id": ex.id, "generic_prompt": ex.generic_prompt, "profile": ex.profile, "target": ex.target,
                "metadata": ex.metadata or base.get("metadata"),
                "raw_generic_output": raw, args.steered_output_key: steered,
            }, ensure_ascii=False) + "\n")
    if routed and n_unknown:
        print(f"[persistent_steer] {n_unknown} examples had users absent from the routing table (uniform weights)")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
