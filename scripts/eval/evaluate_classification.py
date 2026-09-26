#!/usr/bin/env python
"""Exact-match / accuracy scoring for classification-style targets (LaMP-2 tags,
LaMP-3 ratings): compares each candidate output's first line (normalised) to
the target; for numeric targets also reports MAE. Writes JSON per candidate."""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def norm(s: str) -> str:
    s = str(s or "").strip().splitlines()[0] if str(s or "").strip() else ""
    s = re.sub(r"^(tag|answer|score|rating)\s*[:：]\s*", "", s, flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--candidate-keys", nargs="+", default=["raw_generic_output", "generic_pred_output", "jepa_steered_output"])
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    out = {}
    for key in a.candidate_keys:
        if not any(key in r for r in rows):
            continue
        hits, n, abs_err, n_num = 0, 0, 0.0, 0
        for r in rows:
            tgt, pred = norm(r.get("target")), norm(r.get(key))
            n += 1; hits += int(pred == tgt or (tgt and pred.startswith(tgt)))
            try:
                abs_err += abs(float(pred.split()[0]) - float(tgt.split()[0])); n_num += 1
            except (ValueError, IndexError):
                pass
        out[key] = {"accuracy": hits / max(n, 1), "n": n, **({"mae": abs_err / n_num, "n_numeric": n_num} if n_num else {})}
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
