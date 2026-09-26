#!/usr/bin/env bash
# Cold start: hold out personas entirely (no expert, no calibration in training),
# then route each held-out persona from k of its residual examples to the trained
# groups and generate its dev examples. Compares k in {0(uniform),1,5,20} against
# raw and the in-distribution routed model. Idempotent; self-freezing.
#   DATASET=synper MODEL=Qwen/Qwen3.5-4B LAYER=16 HOLDOUT="The Pirate,The Systems Engineer" bash scripts/ablations/run_coldstart.sh
set -uo pipefail
if [ -z "${PERSJEPA_FROZEN_DRIVER:-}" ]; then _f="$(mktemp "${TMPDIR:-/tmp}/persjepa_driver_XXXXXX.sh")"; cp "${BASH_SOURCE[0]}" "$_f"; PERSJEPA_FROZEN_DRIVER="$_f" exec bash "$_f" "$@"; fi
D="${PERSJEPA_ROOT:-$PWD}"; cd "$D"
PY="${PYTHON:-python}"
DATASET=${DATASET:-synper}; MODEL=${MODEL:-Qwen/Qwen3.5-4B}; LAYER=${LAYER:-16}; SEED=${SEED:-42}
HOLDOUT=${HOLDOUT:-"The Pirate,The Systems Engineer"}; KS=${KS:-"0 1 5 20"}; SCALE=${SCALE:-1.0}; MAX_NEW=${MAX_NEW:-64}
LIK_EPOCHS=${LIK_EPOCHS:-2}; AUX_MSE=${AUX_MSE:-0.1}; RUN_TAG=${RUN_TAG:-}
MSLUG=$(echo "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9_.\n' '_')
SRC=${SRC:-runs/${DATASET}/${MSLUG}_L${LAYER}${RUN_TAG}}      # a finished run_stage_dataset.sh dir (extraction + dev subset)
EX=runs/${DATASET}/${MSLUG}_L${LAYER}${RUN_TAG}_coldstart; CK=$EX/ckpt; GEN=$EX/generation_seed${SEED}; mkdir -p $CK $GEN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
[ -f $SRC/span_hidden_calib.pt ] || { echo "need $SRC/span_hidden_calib.pt (run run_stage_dataset.sh first)"; exit 1; }
# held-out dev subset + kept-persona calibration jsonl
$PY - <<PY
import json, sys; sys.path.insert(0, ".")
from persjepa.routing import group_of
hold = {x.strip() for x in "${HOLDOUT}".split(",")}
n1 = n2 = 0
with open("$EX/dev_holdout.jsonl", "w") as out:
    for l in open("$SRC/dev_subset.jsonl"):
        if l.strip() and group_of(json.loads(l)) in hold: out.write(l); n1 += 1
calib = "${CALIB:-data/synper/train_10000.jsonl}"
with open("$EX/calib_kept.jsonl", "w") as out:
    for l in open(calib):
        if l.strip() and group_of(json.loads(l)) not in hold: out.write(l); n2 += 1
print(f"held-out dev {n1}, kept calibration {n2}")
PY
# 1. routed SAE without the held-out personas
[ -f $CK/routed_jepa_sae.pt ] || { log "routed SAE excluding $HOLDOUT"; $PY scripts/training/train_routed_sae.py --span-hidden $SRC/span_hidden_calib.pt --output-dir $CK --num-groups 0 --routing-mode soft --tau 1.0 --exclude-groups "$HOLDOUT" --epochs 80 --finetune-epochs 40 --seed $SEED --device cuda 2>&1 | grep -aE "holding|groups=|wrote|Traceback"; }
# 2. likelihood fine-tune on kept personas only
[ -f $CK/routed_jepa_sae_lik.pt ] || { log "likelihood (kept personas)"; $PY scripts/training/train_residual_likelihood.py --input $EX/calib_kept.jsonl --checkpoint $CK/routed_jepa_sae.pt --span-hidden $SRC/span_hidden_calib.pt --output $CK/routed_jepa_sae_lik.pt --model-name $MODEL --layer $LAYER --epochs $LIK_EPOCHS --batch-size 8 --lr 1e-4 --aux-mse $AUX_MSE --seed $SEED 2>&1 | grep -aE "wrote|Traceback"; }
# 3. cold-start routing tables + generation on held-out dev
first=""
for K in $KS; do
  for base in routed_jepa_sae_lik routed_mean_delta; do
    ck=$CK/${base}_cold_k${K}.pt
    [ -f $ck ] || $PY scripts/eval/coldstart_route.py --checkpoint $CK/$base.pt --span-hidden $SRC/span_hidden_calib.pt --users "$HOLDOUT" --k $K --seed $SEED --output $ck 2>&1 | grep -aE "coldstart|Traceback"
    out=$GEN/persistent__${base}_cold_k${K}__scale${SCALE}.jsonl
    [ -f $out ] && [ "$(wc -l < $out)" -ge "$(wc -l < $EX/dev_holdout.jsonl)" ] && { [ -z "$first" ] && first=$out; continue; }
    log "generate $base k=$K"
    reuse=""; [ -n "$first" ] && reuse="--reuse-baselines-from $first"
    $PY scripts/eval/persistent_steer.py --input $EX/dev_holdout.jsonl --checkpoint $ck --output $out --model-name $MODEL --layer $LAYER --predictor-tokens 3 --residual-scale $SCALE --max-new-tokens $MAX_NEW --seed $SEED $reuse 2>&1 | grep -aE "wrote|Traceback"
    [ -z "$first" ] && first=$out
  done
done
# in-distribution reference: the full model (trained WITH the held-out personas) on the same held-out dev
if [ -f $SRC/ckpt/routed_jepa_sae_lik.pt ]; then
  out=$GEN/persistent__indist_routed_jepa_sae_lik__scale${SCALE}.jsonl
  [ -f $out ] || { log "in-distribution reference"; $PY scripts/eval/persistent_steer.py --input $EX/dev_holdout.jsonl --checkpoint $SRC/ckpt/routed_jepa_sae_lik.pt --output $out --model-name $MODEL --layer $LAYER --predictor-tokens 3 --residual-scale $SCALE --max-new-tokens $MAX_NEW --seed $SEED --reuse-baselines-from $first 2>&1 | grep -aE "wrote|Traceback"; }
fi
# 4. scoring
for out in $GEN/*.jsonl; do b=${out%.jsonl}
  [ -f ${b}_persona_strong.csv ] || $PY scripts/eval/evaluate_synper_persona_classifier_strong.py --train ${CALIB:-data/synper/train_10000.jsonl} --generations $out --output ${b}_persona_strong.csv --confusion-output ${b}_persona_strong_confusion.csv --candidate-keys raw_generic_output jepa_steered_output >/dev/null 2>&1
  [ -f ${b}_text_metrics.json ] || $PY scripts/eval/evaluate_text_metrics.py --input $out --output ${b}_text_metrics.json >/dev/null 2>&1
done
$PY scripts/ablations/summarize_stage1.py $GEN | tee $EX/summary_seed${SEED}.txt
log "COLDSTART DONE ($HOLDOUT)"
