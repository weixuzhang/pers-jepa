#!/usr/bin/env python
"""Build LaMP-7 calibration pairs from profile tweets.

LaMP-7's task input is a neutral paraphrase of a user's tweet and the target
is the user's original tweet. Profile items are bare tweets, so to get
per-user calibration pairs we generate the neutral paraphrase ourselves with a
local model (no API cost):  prompt = LaMP-7 instruction + neutral paraphrase,
target = original tweet, metadata.user_id = the query's user id.

Input: the prepared LaMP-7 jsonl (from prepare_lamp.py --task 7, which keeps
the raw profile in metadata via --keep-raw-profile). Output: calibration jsonl.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

INSTR = "Paraphrase the following tweet without any explanation before or after it: "
NEUTRAL = "Rewrite the following tweet in plain, neutral language, keeping its meaning but removing personal style, slang, emojis and hashtags. Output only the rewritten text.\n\nTweet: {tweet}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="prepared LaMP-7 jsonl with metadata.raw_profile")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--pairs-per-user", type=int, default=8)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from persjepa.chat import render_chat_prompt, wants_chat_template

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("metadata", {}).get("raw_profile")]
    random.Random(a.seed).shuffle(rows)
    if a.max_users: rows = rows[: a.max_users]
    tok = AutoTokenizer.from_pretrained(a.model_name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_name, dtype=torch.bfloat16).to(a.device).eval()
    chat = wants_chat_template(a.model_name, tok)

    jobs = []  # (row, tweet)
    for r in rows:
        prof = r["metadata"]["raw_profile"]
        random.Random(a.seed).shuffle(prof)
        for it in prof[: a.pairs_per_user]:
            t = " ".join(str(it.get("text", "")).split())
            if len(t) > 15: jobs.append((r, t))
    print(f"[lamp7] {len(rows)} users, {len(jobs)} profile tweets to neutralize with {a.model_name}")
    out = open(a.output, "w", encoding="utf-8"); n = 0
    for i in range(0, len(jobs), a.batch_size):
        batch = jobs[i: i + a.batch_size]
        prompts = [render_chat_prompt(tok, NEUTRAL.format(tweet=t)) if chat else NEUTRAL.format(tweet=t) + "\n\nRewritten:" for _, t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(a.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for (r, t), g in zip(batch, gen):
            neutral = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().splitlines()[0].strip() if True else ""
            if not neutral or neutral.lower() == t.lower(): continue
            out.write(json.dumps({
                "id": f"{r['id']}__cal{n}", "profile": r["profile"], "prompt": INSTR + neutral, "target": t,
                "metadata": {**{k: v for k, v in r["metadata"].items() if k != "raw_profile"}, "kind": "profile_pair_neutralized"},
            }, ensure_ascii=False) + "\n"); n += 1
        if (i // a.batch_size) % 20 == 0: print(f"  {i + len(batch)}/{len(jobs)} -> {n} pairs")
    print(f"[lamp7] wrote {n} calibration pairs to {a.output}")


if __name__ == "__main__":
    main()
