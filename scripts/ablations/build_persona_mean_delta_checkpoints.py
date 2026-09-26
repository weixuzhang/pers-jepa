#!/usr/bin/env python
"""From a SynPer span-hidden file, write mean_delta checkpoints (global +
per-persona, answer-span residuals) and per-persona dev subsets for E3
(persistent-injection comparison: global vector vs persona vector)."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import torch


def persona_of(rec: dict) -> str:
    m = rec.get("metadata", {}) or {}
    inner = m.get("metadata", {}) if isinstance(m.get("metadata"), dict) else {}
    return str(inner.get("persona") or m.get("persona") or "unknown")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-hidden", required=True)
    ap.add_argument("--dev", required=True, help="dev jsonl to split per persona")
    ap.add_argument("--max-dev-per-persona", type=int, default=30)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    d = torch.load(a.span_hidden, map_location="cpu")
    mask = (d["source_answer_token_mask"].bool() & d["target_answer_token_mask"].bool()).float()
    r = ((d["target_answer_tokens"].float() - d["source_answer_tokens"].float()) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
    personas = [persona_of(e) for e in d["examples"]]
    torch.save({"model_type": "mean_delta", "mean_delta": r.mean(0), "n": int(r.shape[0]),
                "source": a.span_hidden, "residual": "answer_span_mean"}, out / "global_mean_delta.pt")
    by = defaultdict(list)
    for i, p in enumerate(personas): by[p].append(i)
    slug = lambda p: p.lower().replace("the ", "").replace(" ", "_").replace("-", "_")
    manifest = {}
    for p, idx in sorted(by.items()):
        torch.save({"model_type": "mean_delta", "mean_delta": r[idx].mean(0), "n": len(idx), "persona": p,
                    "source": a.span_hidden, "residual": "answer_span_mean"}, out / f"persona_{slug(p)}_mean_delta.pt")
        manifest[p] = {"slug": slug(p), "n_calibration": len(idx)}
    # per-persona dev subsets
    dev = [json.loads(l) for l in open(a.dev, encoding="utf-8") if l.strip()]
    dev_by = defaultdict(list)
    for rec in dev: dev_by[persona_of(rec)].append(rec)
    for p, recs in dev_by.items():
        if p not in manifest: continue
        sub = recs[: a.max_dev_per_persona]
        with open(out / f"dev_{manifest[p]['slug']}.jsonl", "w", encoding="utf-8") as h:
            for rec in sub: h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest[p]["n_dev"] = len(sub)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"wrote global + {len(manifest)} persona checkpoints to {out}; dev subsets:",
          {v['slug']: v.get('n_dev', 0) for v in manifest.values()})


if __name__ == "__main__":
    main()
