#!/usr/bin/env bash
# Cheap ablations of the likelihood-trained routed SAE on the legacy setup
# (Qwen2.5-1.5B, layer 14, SynPer), reusing the finished extraction and MSE
# checkpoints. Each variant: likelihood fine-tune -> persistent generation at
# scale 1 -> strong persona classifier + text metrics. Idempotent.
set -uo pipefail
cd "${PERSJEPA_ROOT:-$PWD}"   # run from the repository root
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EX=${SRC:-runs/synper/Qwen_Qwen2.5_1.5B_L14_lik}; G=$EX/generation_seed42; A=$EX/ablations; mkdir -p $A
BASE=$G/persistent__routed_jepa_sae_lik__scale1.0.jsonl   # raw baseline reuse
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
run_variant() {  # name  init_ckpt  extra likelihood args...
  local name=$1 init=$2; shift 2
  local ck=$A/${name}.pt out=$G/persistent__abl_${name}__scale1.0.jsonl
  if [ ! -f $ck ]; then log "likelihood: $name ($*)"; python scripts/training/train_residual_likelihood.py --input data/synper/train_10000.jsonl --checkpoint $init --span-hidden $EX/span_hidden_calib.pt --output $ck --layer 14 --batch-size 8 --lr 1e-4 --seed 42 "$@" 2>&1 | grep -aE "wrote|Traceback|Error" || true; fi
  [ -f $ck ] || { echo "FAILED train $name"; return; }
  if [ ! -f $out ]; then log "generate: $name"; python scripts/eval/persistent_steer.py --input $EX/dev_subset.jsonl --checkpoint $ck --output $out --layer 14 --predictor-tokens 3 --residual-scale 1.0 --max-new-tokens 64 --seed 42 --reuse-baselines-from $BASE 2>&1 | grep -aE "wrote|Traceback|Error" || true; fi
  local b=${out%.jsonl}
  [ -f ${b}_persona_strong.csv ] || python scripts/eval/evaluate_synper_persona_classifier_strong.py --train data/synper/train_10000.jsonl --generations $out --output ${b}_persona_strong.csv --confusion-output ${b}_persona_strong_confusion.csv --candidate-keys raw_generic_output jepa_steered_output >/dev/null 2>&1
  [ -f ${b}_text_metrics.json ] || python scripts/eval/evaluate_text_metrics.py --input $out --output ${b}_text_metrics.json >/dev/null 2>&1
}
# objective ablations on the given-persona routed SAE
run_variant aux0   $EX/ckpt/routed_jepa_sae.pt --epochs 1 --aux-mse 0
run_variant aux1   $EX/ckpt/routed_jepa_sae.pt --epochs 1 --aux-mse 1.0
run_variant ep2    $EX/ckpt/routed_jepa_sae.pt --epochs 2 --aux-mse 0.1
# learned clustering: K=5 merged groups, soft routing, then likelihood
if [ ! -f $A/k5soft_ckpt/routed_jepa_sae.pt ]; then log "train routed SAE K=5 soft"; python scripts/training/train_routed_sae.py --span-hidden $EX/span_hidden_calib.pt --output-dir $A/k5soft_ckpt --num-groups 5 --routing-mode soft --tau 1.0 --global-checkpoint $EX/ckpt/global_jepa_sae.pt --finetune-epochs 40 --seed 42 --device cuda 2>&1 | grep -aE "groups=|routing acc|Traceback" || true; fi
run_variant k5soft $A/k5soft_ckpt/routed_jepa_sae.pt --epochs 1 --aux-mse 0.1
# frozen-offset variant: experts learn, per-group constants fixed (is the gain in the experts or the offsets?)
run_variant frozenoff $EX/ckpt/routed_jepa_sae.pt --epochs 1 --aux-mse 0.1 --freeze-offsets
python3 scripts/ablations/summarize_stage1.py $G | grep -E "raw|lik|abl_"
log "ABLATIONS DONE"
