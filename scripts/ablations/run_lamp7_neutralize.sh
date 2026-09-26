#!/usr/bin/env bash
# LaMP-7 preparation (GPU step): neutral paraphrases of profile tweets -> per-user
# calibration pairs (scripts/data/neutralize_lamp7_profiles.py). Idempotent.
# Inputs (from prepare_lamp.py --task 7 --keep-raw-profile, CPU):
#   data/lamp7/dev_rawprofile.jsonl    (1500 dev users = the evaluation users)
#   data/lamp7/train_rawprofile.jsonl  (1500 sampled train users, disjoint; cold-start / extra calibration)
# Outputs: data/lamp7/calib_devusers.jsonl, data/lamp7/calib_trainusers.jsonl,
#          data/lamp7/train_calib.jsonl (= both, what run_stage_dataset.sh reads)
# Env: MODEL (Qwen/Qwen3.5-4B) PAIRS (8) BS (32)
set -uo pipefail
D="${PERSJEPA_ROOT:-$PWD}"; cd "$D"   # run from the repository root
PY="${PYTHON:-python}"; MODEL=${MODEL:-Qwen/Qwen3.5-4B}; PAIRS=${PAIRS:-8}; BS=${BS:-32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
for split in dev train; do
  out=data/lamp7/calib_${split}users.jsonl
  [ -f "$out" ] && { log "skip $out"; continue; }
  log "neutralize ${split} profiles ($MODEL, $PAIRS pairs/user)"
  $PY scripts/data/neutralize_lamp7_profiles.py --input data/lamp7/${split}_rawprofile.jsonl --output ${out}.tmp \
     --model-name $MODEL --pairs-per-user $PAIRS --batch-size $BS --seed 42 --device cuda && mv ${out}.tmp $out || { echo "FAILED $split"; exit 1; }
done
cat data/lamp7/calib_devusers.jsonl data/lamp7/calib_trainusers.jsonl > data/lamp7/train_calib.jsonl
wc -l data/lamp7/calib_devusers.jsonl data/lamp7/calib_trainusers.jsonl data/lamp7/train_calib.jsonl
log "LAMP7 NEUTRALIZE DONE"
