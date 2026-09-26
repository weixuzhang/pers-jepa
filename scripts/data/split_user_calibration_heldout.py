#!/usr/bin/env python
"""Warm-user calibration/heldout split at the jsonl level (stage 3, Amazon).

For every user (``--group-field``, default ``metadata.user_id``) the examples
are ordered by ``--time-field`` (target review timestamp) and the LAST
``--heldout-per-user`` become the held-out evaluation set; the earlier ones are
the calibration set (user latents z_u, routed-SAE training). Users with fewer
than ``--min-calibration`` calibration examples are dropped. Time ordering
means a held-out target never appears in a calibration example's history
prompt. A manifest with the counts is written next to the outputs.

    python scripts/data/split_user_calibration_heldout.py \
        --input data/amazon_movies_tv/all.jsonl \
        --calibration-out data/amazon_movies_tv/calibration.jsonl \
        --heldout-out data/amazon_movies_tv/heldout.jsonl --heldout-per-user 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import write_jsonl
from persjepa.routing import get_field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--calibration-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--manifest-out", default=None, help="default: <heldout dir>/split_manifest.json")
    parser.add_argument("--group-field", default="metadata.user_id")
    parser.add_argument("--time-field", default="metadata.target_timestamp")
    parser.add_argument("--heldout-per-user", type=int, default=5)
    parser.add_argument("--min-calibration", type=int, default=5)
    parser.add_argument("--keep-leaked", action="store_true",
                        help="keep held-out examples whose target text also appears in one of the user's "
                             "calibration history prompts (duplicate reviews); dropped by default")
    parser.add_argument("--leak-prefix-chars", type=int, default=80)
    args = parser.parse_args()

    by_user: dict[str, list[dict]] = defaultdict(list)
    n_in = 0
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            n_in += 1
            user = get_field(record, args.group_field)
            by_user["unknown" if user in (None, "") else str(user)].append(record)

    calibration: list[dict] = []
    heldout: list[dict] = []
    dropped_users = 0
    dropped_leaked = 0
    for user, records in sorted(by_user.items()):
        records = sorted(records, key=lambda r: int(get_field(r, args.time_field) or 0))
        if len(records) - args.heldout_per_user < args.min_calibration:
            dropped_users += 1
            continue
        user_calibration = records[: -args.heldout_per_user]
        user_heldout = records[-args.heldout_per_user :]
        if not args.keep_leaked:
            # Users sometimes post the same review text for several items; a held-out
            # target that is quoted verbatim in a calibration history prompt would let
            # profile prompting copy it. Drop those held-out examples.
            profiles = [str(r.get("profile", "")) for r in user_calibration]
            kept = []
            for r in user_heldout:
                key = str(r.get("target", ""))[: args.leak_prefix_chars]
                if key and any(key in p for p in profiles):
                    dropped_leaked += 1
                else:
                    kept.append(r)
            user_heldout = kept
        calibration.extend(user_calibration)
        heldout.extend(user_heldout)

    write_jsonl(args.calibration_out, calibration)
    write_jsonl(args.heldout_out, heldout)
    kept_users = len(by_user) - dropped_users
    manifest = {
        "input": args.input,
        "examples_in": n_in,
        "users_in": len(by_user),
        "users_kept": kept_users,
        "users_dropped_too_few": dropped_users,
        "heldout_dropped_leaked": dropped_leaked,
        "heldout_per_user": args.heldout_per_user,
        "min_calibration": args.min_calibration,
        "calibration_examples": len(calibration),
        "heldout_examples": len(heldout),
        "group_field": args.group_field,
        "time_field": args.time_field,
        "calibration_out": args.calibration_out,
        "heldout_out": args.heldout_out,
    }
    manifest_path = args.manifest_out or str(Path(args.heldout_out).parent / "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
