#!/usr/bin/env python
"""Cold-start routing: add held-out users to a routed checkpoint's routing table
from only k of their calibration residual examples (no retraining), so their
dev examples can be generated with the existing experts.

  coldstart_route.py --checkpoint ckpt/routed_jepa_sae_lik.pt --span-hidden span_hidden_calib.pt \
      --users "The Pirate,The Systems Engineer" --k 5 --output ckpt/routed_jepa_sae_lik_cold_k5.pt
"""
from __future__ import annotations

import argparse, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from persjepa.routing import RoutingTable, answer_span_residuals, group_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="routed_jepa_sae / routed_mean_delta checkpoint (with routing_table)")
    ap.add_argument("--span-hidden", default=None, help="calibration span-hidden file containing the held-out users' examples (cold start only)")
    ap.add_argument("--users", default="", help="comma-separated held-out identities (cold start); 'all' = every user in --span-hidden, or a path to a jsonl whose records' identities are used")
    ap.add_argument("--shuffle", action="store_true", help="control: permute the routing weights across users so every user gets another user's routing (derangement)")
    ap.add_argument("--k", type=int, default=5, help="number of residual examples per user used to route (0 = uniform)")
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--mode", choices=["soft", "hard", "keep"], default="keep", help="override routing mode (keep = as in checkpoint)")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    pay = torch.load(a.checkpoint, map_location="cpu")
    table = RoutingTable.from_payload(pay["routing_table"])
    if a.mode != "keep": table.mode = a.mode
    if a.tau is not None: table.tau = a.tau
    rng = random.Random(a.seed)
    if a.shuffle:
        names = sorted(table.weights)
        perm = names[:]
        for _ in range(100):
            rng.shuffle(perm)
            if all(x != y for x, y in zip(names, perm)): break
        old = {u: table.weights[u] for u in names}
        for u, v in zip(names, perm): table.weights[u] = old[v]
        moved = sum(int(old[u].argmax()) != int(table.weights[u].argmax()) for u in names)
        print(f"[shuffle] routing permuted for {len(names)} users; {moved} changed their top group")
        pay["routing_table"] = table.to_payload(); pay["shuffled_routing"] = {"seed": a.seed}
        torch.save(pay, a.output); print("wrote", a.output); return
    d = torch.load(a.span_hidden, map_location="cpu")
    _, r = answer_span_residuals(d)
    users = [group_of(rec, a.group_field) for rec in d["examples"]]
    if a.users == "all":
        wanted = sorted(set(users))
    elif a.users.endswith(".jsonl"):
        import json
        wanted = sorted({group_of(json.loads(l), a.group_field) for l in open(a.users) if l.strip()})
    else:
        wanted = [x.strip() for x in a.users.split(",") if x.strip()]
    print(f"[coldstart] routing {len(wanted)} users from k={a.k} examples")
    n_missing = 0
    for u in wanted:
        idx = [i for i, uu in enumerate(users) if uu == u]
        if not idx:
            n_missing += 1; continue
        if a.k <= 0:
            w = torch.full((table.num_groups,), 1.0 / table.num_groups); table.weights[u] = w
        else:
            pick = rng.sample(idx, k=min(a.k, len(idx)))
            w = table.add_user_from_examples(u, r[pick])
        if len(wanted) <= 12:
            print(f"[coldstart] {u}: k={a.k} -> weights {[round(float(x), 2) for x in w]} (top: {table.group_names[int(w.argmax())]})")
    if n_missing: print(f"[coldstart] {n_missing} users had no calibration examples (left as in the checkpoint)")
    if len(wanted) > 12:
        import collections
        tops = collections.Counter(table.group_names[int(table.weights[u].argmax())] for u in wanted if u in table.weights)
        print(f"[coldstart] top-group histogram over routed users: {dict(tops)}")
    pay["routing_table"] = table.to_payload()
    pay["coldstart"] = {"users": a.users, "k": a.k, "mode": table.mode, "tau": table.tau, "seed": a.seed}
    torch.save(pay, a.output); print("wrote", a.output)


if __name__ == "__main__":
    main()
