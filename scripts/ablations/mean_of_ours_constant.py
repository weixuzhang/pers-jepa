#!/usr/bin/env python
"""'Mean-of-ours' constant control.

For each group g: c_g = mean over group g's CALIBRATION records of the routed
likelihood SAE's predicted residual, computed exactly as at inference
(persjepa.persistent.PersistentSteerer.compute_delta: layer-L state at the last
[PRED] token of the chat-formatted generic prompt -> predictor with the user's
routing weights, scale 1). The constants are written as a routed_mean_delta
checkpoint with the routing table copied from ckpt/routed_mean_delta.pt, so
persistent_steer then injects the same fixed vector for every prompt of a user.

  python scripts/ablations/mean_of_ours_constant.py --run-dir runs/synper/Qwen_Qwen3.5_4B_L16 \
      --calib data/synper/train_10000.jsonl --model-name Qwen/Qwen3.5-4B --layer 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--predictor-tokens", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--source", default="routed_jepa_sae_lik")
    ap.add_argument("--output", default=None, help="default: <run-dir>/ckpt/mean_of_ours.pt")
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    from persjepa.chat import apply_chat_formatting
    from persjepa.config import ExperimentConfig
    from persjepa.data import load_persona_jsonl
    from persjepa.intervention import load_causal_lm, load_steering_sae
    from persjepa.persistent import PersistentSteerer
    from persjepa.routing import group_of

    run = Path(args.run_dir); ck = run / "ckpt"
    out = Path(args.output) if args.output else ck / "mean_of_ours.pt"
    cfg = ExperimentConfig()
    examples = load_persona_jsonl(
        args.calib, profile_field=cfg.data.profile_field, prompt_field=cfg.data.text_field,
        target_field=cfg.data.target_field, id_field=cfg.data.id_field, profile_template=cfg.data.profile_template,
    )
    model, tokenizer, device = load_causal_lm(args.model_name, pred_token=cfg.model.pred_token, device="cuda", dtype="bfloat16")
    chat_on = apply_chat_formatting(examples, tokenizer, args.model_name, "auto", None)
    predictor, _ = load_steering_sae(str(ck / f"{args.source}.pt"), device=device)
    table = predictor.routing_table
    steerer = PersistentSteerer(model, tokenizer, predictor, layer=args.layer, pred_token=cfg.model.pred_token,
                                predictor_tokens=args.predictor_tokens, max_length=args.max_length,
                                residual_scale=1.0, device=device)
    ref = torch.load(ck / "routed_mean_delta.pt", map_location="cpu", weights_only=False)
    G = predictor.num_groups
    assert G == ref["group_means"].shape[0] and list(ref["group_names"]) == list(getattr(predictor, "group_names", ref["group_names"])), "group mismatch"
    sums = torch.zeros(G, ref["group_means"].shape[1], dtype=torch.float64)
    counts = torch.zeros(G, dtype=torch.float64); skipped = 0
    for ex in tqdm(examples, desc="mean_of_ours"):
        user = group_of({"metadata": ex.metadata or {}}, args.group_field)
        w = table.get(user)
        if w is None:
            skipped += 1; continue
        predictor.set_weights(w.to(device))
        d = steerer.compute_delta(ex.generic_prompt).float().cpu().double()[0]
        g = int(torch.as_tensor(w).argmax())            # group of this record (hard routing on SynPer)
        sums[g] += d; counts[g] += 1
    means = sums / counts.clamp(min=1)[:, None]
    empty = [i for i in range(G) if counts[i] == 0]
    for i in empty:                                      # groups with no calibration record keep the reference constant
        means[i] = ref["group_means"][i].double()
    payload = {"model_type": "routed_mean_delta", "group_means": means.float(), "group_names": ref["group_names"],
               "routing_table": ref["routing_table"],
               "mean_of_ours": {"source": args.source, "calib": args.calib, "counts": counts.tolist(),
                                "skipped_unrouted": skipped, "empty_groups_kept_reference": empty, "chat_template": chat_on}}
    torch.save(payload, out)
    print(f"[mean_of_ours] wrote {out}: groups {G}, counts {[int(c) for c in counts]}, skipped {skipped}, "
          f"empty groups {empty}, chat {chat_on}")
    ref_means = ref["group_means"].double()
    for i, name in enumerate(ref["group_names"]):
        if counts[i] > 0:
            cos = torch.nn.functional.cosine_similarity(means[i], ref_means[i], dim=0).item()
            print(f"  {name:28s} n={int(counts[i]):5d} |c_g|={means[i].norm():.3f} |mean_delta_g|={ref_means[i].norm():.3f} cos={cos:.3f}")


if __name__ == "__main__":
    main()
