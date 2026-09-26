#!/usr/bin/env bash
# Seed replication (7, 13) of the likelihood routed SAE (2 epochs, aux 0.1) and the
# K curve (K=3, 8; k-means + soft routing) on the legacy setup. Idempotent.
set -uo pipefail
if [ -z "${PERSJEPA_FROZEN_DRIVER:-}" ]; then _f="$(mktemp "${TMPDIR:-/tmp}/persjepa_driver_XXXXXX.sh")"; cp "${BASH_SOURCE[0]}" "$_f"; PERSJEPA_FROZEN_DRIVER="$_f" exec bash "$_f" "$@"; fi
cd "${PERSJEPA_ROOT:-$PWD}"   # run from the repository root
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EX=${SRC:-runs/synper/Qwen_Qwen2.5_1.5B_L14_lik}; G=$EX/generation_seed42; A=$EX/ablations; mkdir -p $A
BASE=$G/persistent__routed_jepa_sae_lik__scale1.0.jsonl
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
complete() { [ -f "$1" ] && [ "$(wc -l < "$1")" -ge 989 ] && return 0; [ -f "$1" ] && rm -f "$1" "${1%.jsonl}"_*; return 1; }
score() { local b=${1%.jsonl}; [ -f ${b}_persona_strong.csv ] || python scripts/eval/evaluate_synper_persona_classifier_strong.py --train data/synper/train_10000.jsonl --generations $1 --output ${b}_persona_strong.csv --confusion-output ${b}_persona_strong_confusion.csv --candidate-keys raw_generic_output jepa_steered_output >/dev/null 2>&1; [ -f ${b}_text_metrics.json ] || python scripts/eval/evaluate_text_metrics.py --input $1 --output ${b}_text_metrics.json >/dev/null 2>&1; }
gen() { local ck=$1 out=$2; complete $out || { log "generate $(basename $out)"; python scripts/eval/persistent_steer.py --input $EX/dev_subset.jsonl --checkpoint $ck --output $out --layer 14 --predictor-tokens 3 --residual-scale 1.0 --max-new-tokens 64 --seed 42 --reuse-baselines-from $BASE 2>&1 | grep -aE "wrote|Traceback|Error"; }; score $out; }
# --- seeds: retrain routed SAE (MSE) + likelihood (2 ep, aux 0.1) with a new seed; dev subset unchanged
for SEED in 7 13; do
  CKS=$A/seed${SEED}_ckpt
  [ -f $CKS/routed_jepa_sae.pt ] || { log "routed SAE seed $SEED"; python scripts/training/train_routed_sae.py --span-hidden $EX/span_hidden_calib.pt --output-dir $CKS --num-groups 0 --routing-mode hard --epochs 80 --finetune-epochs 40 --seed $SEED --device cuda 2>&1 | grep -aE "wrote|Traceback"; }
  [ -f $A/seed${SEED}_lik.pt ] || { log "likelihood seed $SEED"; python scripts/training/train_residual_likelihood.py --input data/synper/train_10000.jsonl --checkpoint $CKS/routed_jepa_sae.pt --span-hidden $EX/span_hidden_calib.pt --output $A/seed${SEED}_lik.pt --layer 14 --batch-size 8 --lr 1e-4 --epochs 2 --aux-mse 0.1 --seed $SEED 2>&1 | grep -aE "wrote|Traceback"; }
  gen $A/seed${SEED}_lik.pt $G/persistent__abl_seed${SEED}_ep2__scale1.0.jsonl
done
# --- K curve: k-means groups, soft routing, likelihood (1 ep, aux 0.1) — comparable to abl_k5soft
for K in 3 8; do
  CKK=$A/k${K}soft_ckpt
  [ -f $CKK/routed_jepa_sae.pt ] || { log "routed SAE K=$K soft"; python scripts/training/train_routed_sae.py --span-hidden $EX/span_hidden_calib.pt --output-dir $CKK --num-groups $K --routing-mode soft --tau 1.0 --global-checkpoint $EX/ckpt/global_jepa_sae.pt --finetune-epochs 40 --seed 42 --device cuda 2>&1 | grep -aE "groups=|Traceback"; }
  [ -f $A/k${K}soft.pt ] || { log "likelihood K=$K"; python scripts/training/train_residual_likelihood.py --input data/synper/train_10000.jsonl --checkpoint $CKK/routed_jepa_sae.pt --span-hidden $EX/span_hidden_calib.pt --output $A/k${K}soft.pt --layer 14 --batch-size 8 --lr 1e-4 --epochs 1 --aux-mse 0.1 --seed 42 2>&1 | grep -aE "wrote|Traceback"; }
  gen $A/k${K}soft.pt $G/persistent__abl_k${K}soft__scale1.0.jsonl
done
python3 scripts/ablations/summarize_stage1.py $G | grep -E "raw|lik  *1.0|abl_"
log "SEEDS_AND_K DONE"
