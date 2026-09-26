#!/usr/bin/env bash
# Real-data cold start: route the dev users of a finished run_stage_dataset.sh
# run from only k of their own calibration examples (k = 0 uniform, 1, 5, 20),
# generate on the dev subset with the likelihood-trained routed SAE and the
# routed constant, and score with the dataset's recipe. Idempotent.
#   DATASET=lamp7 MODEL=Qwen/Qwen3.5-4B LAYER=16 RUN_TAG=_k5 GROUP_FIELD=metadata.user_id \
#   METRIC=userstyle bash scripts/ablations/run_coldstart_users.sh
# Optional: KS="0 1 5 20"  SCALE=1.0  SEED=42  MAX_NEW=64  CALIB / DEV (as in the main driver)
set -uo pipefail
if [ -z "${PERSJEPA_FROZEN_DRIVER:-}" ]; then
  _frozen="$(mktemp "${TMPDIR:-/tmp}/persjepa_driver_XXXXXX.sh")"; cp "${BASH_SOURCE[0]}" "$_frozen"
  PERSJEPA_FROZEN_DRIVER="$_frozen" exec bash "$_frozen" "$@"
fi
D="${PERSJEPA_ROOT:-$PWD}"; cd "$D"   # run from the repository root
PY="${PYTHON:-python}"
DATASET=${DATASET:?set DATASET}; MODEL=${MODEL:-Qwen/Qwen3.5-4B}; LAYER=${LAYER:-16}; RUN_TAG=${RUN_TAG:-}
KS=${KS:-"0 1 5 20"}; SCALE=${SCALE:-1.0}; SEED=${SEED:-42}; MAX_NEW=${MAX_NEW:-64}
case "$DATASET" in
  synper)       CALIB=${CALIB:-data/synper/train_10000.jsonl}; METRIC=${METRIC:-synper_persona} ;;
  lamp7)        CALIB=${CALIB:-data/lamp7/train_calib.jsonl};  METRIC=${METRIC:-userstyle} ;;
  lamp2)        CALIB=${CALIB:-data/lamp2/train_calib.jsonl};  METRIC=${METRIC:-classify} ;;
  lamp4)        CALIB=${CALIB:-data/lamp4/train_calib.jsonl};  METRIC=${METRIC:-userstyle} ;;
  lamp5)        CALIB=${CALIB:-data/lamp5/train_calib.jsonl};  METRIC=${METRIC:-userstyle} ;;
  amazon_movies_tv) CALIB=${CALIB:-data/amazon_movies_tv/calibration.jsonl}; METRIC=${METRIC:-userstyle} ;;
  *)            CALIB=${CALIB:?CALIB required}; METRIC=${METRIC:-text} ;;
esac
MSLUG=$(echo "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9_.\n' '_')
SRC=${SRC:-runs/${DATASET}/${MSLUG}_L${LAYER}${RUN_TAG}}
EX=$SRC/coldstart_users; CK=$EX/ckpt; GEN=$EX/generation_seed${SEED}; mkdir -p $CK $GEN logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
[ -f $SRC/span_hidden_calib.pt ] && [ -f $SRC/dev_subset.jsonl ] || { echo "need $SRC/{span_hidden_calib.pt,dev_subset.jsonl}"; exit 1; }
[ -f $SRC/ckpt/routed_jepa_sae_lik.pt ] || { echo "need $SRC/ckpt/routed_jepa_sae_lik.pt (LIKELIHOOD=1 run)"; exit 1; }
NDEVL=$(wc -l < $SRC/dev_subset.jsonl)
log "real-data cold start: dataset=$DATASET src=$SRC users=dev subset ($NDEVL examples) K=$KS"
BASE=$SRC/generation_seed${SEED}/persistent__global_mean_delta__scale1.0.jsonl
for K in $KS; do
  for base in routed_jepa_sae_lik routed_mean_delta; do
    ck=$CK/${base}_cold_k${K}.pt
    [ -f $ck ] || $PY scripts/eval/coldstart_route.py --checkpoint $SRC/ckpt/$base.pt --span-hidden $SRC/span_hidden_calib.pt \
        --users $SRC/dev_subset.jsonl --k $K ${GROUP_FIELD:+--group-field $GROUP_FIELD} --seed $SEED --output $ck 2>&1 | grep -aE "coldstart\]|Traceback"
    out=$GEN/persistent__${base}_cold_k${K}__scale${SCALE}.jsonl
    [ -f $out ] && [ "$(wc -l < $out)" -ge "$NDEVL" ] && continue
    log "generate $base k=$K"
    reuse=""; [ -f "$BASE" ] && reuse="--reuse-baselines-from $BASE"
    $PY scripts/eval/persistent_steer.py --input $SRC/dev_subset.jsonl --checkpoint $ck --output $out --model-name $MODEL --layer $LAYER \
        --predictor-tokens 3 --residual-scale $SCALE --max-new-tokens $MAX_NEW ${GROUP_FIELD:+--group-field $GROUP_FIELD} --seed $SEED $reuse --resume 2>&1 | grep -aE "wrote|Traceback|unknown"
  done
done
# scoring (same recipes as run_stage_dataset.sh)
for out in $GEN/*.jsonl; do b=${out%.jsonl}
  [ -f ${b}_text_metrics.json ] || $PY scripts/eval/evaluate_text_metrics.py --input $out --output ${b}_text_metrics.json >/dev/null 2>&1
  case "$METRIC" in
    synper_persona)
      [ -f ${b}_persona_strong.csv ] || $PY scripts/eval/evaluate_synper_persona_classifier_strong.py --train $CALIB --generations $out \
         --output ${b}_persona_strong.csv --confusion-output ${b}_persona_strong_confusion.csv --candidate-keys raw_generic_output jepa_steered_output >/dev/null 2>&1 ;;
    userstyle)
      [ -f ${b}_userstyle.csv ] || $PY scripts/eval/evaluate_group_style_classifier_strong.py --train $CALIB --generations $out \
         --label-field ${GROUP_FIELD:-metadata.user_id} --dataset $DATASET --output ${b}_userstyle.csv --confusion-output ${b}_userstyle_confusion.csv >/dev/null 2>&1 ;;
    classify)
      [ -f ${b}_classify.json ] || $PY scripts/eval/evaluate_classification.py --input $out --output ${b}_classify.json >/dev/null 2>&1 ;;
    text) : ;;
  esac
done
$PY scripts/ablations/summarize_stage1.py $GEN | tee $EX/summary_seed${SEED}.txt
log "COLDSTART-USERS DONE ($DATASET, $SRC)"
