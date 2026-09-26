#!/usr/bin/env bash
# Generic per-dataset driver for the routed JEPA-SAE campaign (stages 1-3).
# Idempotent (finished steps are skipped); one python process per step.
#
#   DATASET=synper MODEL=Qwen/Qwen3.5-4B LAYER=16 bash scripts/ablations/run_stage_dataset.sh
#
# Required env
#   DATASET     synper | lamp7 | lamp2 | amazon_movies_tv | <any name with CALIB/DEV set>
# Optional env (defaults)
#   MODEL (Qwen/Qwen3.5-4B)  LAYER (16)  SCALES ("1.0 2.0 4.0")  SEED (42)
#   NDEV (per-group cap on dev examples for grouped sets, 100)  MAX_NEW (64)
#   NUM_GROUPS (0 = given identities are the groups; >0 = k-means)  ROUTING (hard|soft)
#   LIKELIHOOD (0|1)  LIK_EPOCHS (2)  AUX_MSE (0.1)
#   ARMS (default: global_mean_delta routed_mean_delta global_jepa_sae routed_jepa_sae [+ routed_jepa_sae_lik]; add profile_prompt for the reference arm)
#   CALIB / DEV  jsonl paths (defaults per DATASET below)   GROUP_FIELD (auto)
#   METRIC       synper_persona | text | userstyle | classify  (scoring recipe)
#   RUN_TAG      suffix for the run dir (e.g. _k20)
#   BASELINES    1 = add the six baseline arms (profile_prompt rag_bm25 fints steerx tapper plume)
#   LIK_GLOBAL   1 = also likelihood-train the GLOBAL SAE (no-routing control arm global_jepa_sae_lik)
#   SKIP_ANCHOR  1 = skip the single-anchor contrast step (the contrast is established on SynPer)
#   CALIB_HOLDOUT fraction of CALIB held out (never trained on) for K selection by teacher-forced NLL
#                (0 = off; use a distinct RUN_TAG for held-out runs, extraction depends on CALIB)
set -uo pipefail
# Self-freeze: bash reads scripts incrementally, so editing this file while it
# runs can break the running job. Re-exec from a private copy.
if [ -z "${PERSJEPA_FROZEN_DRIVER:-}" ]; then
  _frozen="$(mktemp "${TMPDIR:-/tmp}/persjepa_driver_XXXXXX.sh")"; cp "${BASH_SOURCE[0]}" "$_frozen"
  PERSJEPA_FROZEN_DRIVER="$_frozen" exec bash "$_frozen" "$@"
fi
D="${PERSJEPA_ROOT:-$PWD}"; cd "$D"   # run from the repository root
PY="${PYTHON:-python}"
DATASET=${DATASET:?set DATASET}; MODEL=${MODEL:-Qwen/Qwen3.5-4B}; LAYER=${LAYER:-16}
SCALES=${SCALES:-"1.0 2.0 4.0"}; SEED=${SEED:-42}; NDEV=${NDEV:-100}; MAX_NEW=${MAX_NEW:-64}
NUM_GROUPS=${NUM_GROUPS:-0}; ROUTING=${ROUTING:-hard}; LIKELIHOOD=${LIKELIHOOD:-0}; RUN_TAG=${RUN_TAG:-}
case "$DATASET" in
  synper)       CALIB=${CALIB:-data/synper/train_10000.jsonl}; DEV=${DEV:-data/synper/dev_1000.jsonl}; METRIC=${METRIC:-synper_persona} ;;
  lamp7)        CALIB=${CALIB:-data/lamp7/train_calib.jsonl};  DEV=${DEV:-data/lamp7/dev.jsonl};        METRIC=${METRIC:-userstyle} ;;   # text metrics are always computed; userstyle adds the user-attribution classifier
  lamp2)        CALIB=${CALIB:-data/lamp2/train_calib.jsonl};  DEV=${DEV:-data/lamp2/dev.jsonl};        METRIC=${METRIC:-classify} ;;
  lamp4)        CALIB=${CALIB:-data/lamp4/train_calib.jsonl};  DEV=${DEV:-data/lamp4/dev.jsonl};        METRIC=${METRIC:-userstyle} ;;   # news headline generation
  lamp5)        CALIB=${CALIB:-data/lamp5/train_calib.jsonl};  DEV=${DEV:-data/lamp5/dev.jsonl};        METRIC=${METRIC:-userstyle} ;;   # scholarly title generation (prepare_lamp.py --task 5; not the legacy train_3000 files)
  amazon_movies_tv) CALIB=${CALIB:-data/amazon_movies_tv/calibration.jsonl}; DEV=${DEV:-data/amazon_movies_tv/heldout.jsonl}; METRIC=${METRIC:-userstyle} ;;
  *)            CALIB=${CALIB:?CALIB required for custom DATASET}; DEV=${DEV:?DEV required}; METRIC=${METRIC:-text} ;;
esac
MSLUG=$(echo "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9_.\n' '_')
EX=runs/${DATASET}/${MSLUG}_L${LAYER}${RUN_TAG}; CK=$EX/ckpt; GEN=$EX/generation_seed${SEED}; mkdir -p $CK $GEN logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
# optional calibration hold-out (K selection / objective check on data never trained on)
CALIB_HOLDOUT=${CALIB_HOLDOUT:-0}
if [ "$CALIB_HOLDOUT" != "0" ]; then
  if [ ! -f $EX/calib_holdout.jsonl ]; then
    log "hold out $CALIB_HOLDOUT of $CALIB -> $EX/calib_train.jsonl / calib_holdout.jsonl"
    $PY - <<PY
import json, random
recs=[l for l in open("$CALIB") if l.strip()]; random.Random($SEED).shuffle(recs)
n=int(len(recs)*float("$CALIB_HOLDOUT")); open("$EX/calib_holdout.jsonl","w").writelines(recs[:n]); open("$EX/calib_train.jsonl","w").writelines(recs[n:])
print(f"holdout {n} / train {len(recs)-n}")
PY
  fi
  CALIB=$EX/calib_train.jsonl
fi
log "dataset=$DATASET model=$MODEL layer=$LAYER calib=$CALIB dev=$DEV metric=$METRIC run=$EX"
# generation files are written incrementally: an interrupted job leaves a partial file, which must be redone
# persistent_steer.py and evaluate.py both resume a partial file (--resume); never delete a file another driver may be writing
complete() { [ -f "$1" ] && [ "$(wc -l < "$1")" -ge "$(wc -l < $EX/dev_subset.jsonl)" ] && return 0; [ -f "$1" ] && { log "incomplete $1 ($(wc -l < "$1") lines) -> resume/redo (dropping its stale scores)"; rm -f "${1%.jsonl}"_*; }; return 1; }

# 1. calibration extraction ---------------------------------------------------
if [ ! -f $EX/span_hidden_calib.pt ]; then
  log "extract calibration span hidden (layer $LAYER)"
  $PY scripts/training/extract_span_hidden.py --input $CALIB --output $EX/span_hidden_calib.pt \
     --model-name $MODEL --predictor-tokens 3 --layer $LAYER --max-length 1024 \
     --include-answer-spans --max-answer-tokens 32 --batch-size 8 --device cuda || exit 1
fi
# dev subset: cap per group, deterministic; robust identity lookup
if [ ! -f $EX/dev_subset.jsonl ]; then
  GF_PY="${GROUP_FIELD:+'$GROUP_FIELD'}"; GF_PY="${GF_PY:-None}"
  $PY - <<PY
import json, collections, random, sys
sys.path.insert(0, ".")
from persjepa.routing import group_of
random.seed(${SEED}); by = collections.defaultdict(list)
for l in open("${DEV}"):
    if l.strip(): r = json.loads(l); by[group_of(r, ${GF_PY})].append(r)
n = 0
keep = []
if True:
    for g, rs in sorted(by.items()):
        random.shuffle(rs)
        for r in rs[:${NDEV}]: keep.append(r)
dev_max = int("${DEV_MAX:-0}")
if dev_max and len(keep) > dev_max:
    random.Random(${SEED}).shuffle(keep); keep = keep[:dev_max]
with open("$EX/dev_subset.jsonl", "w") as out:
    for r in keep: out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"dev subset: {n} examples over {len(by)} groups (DEV_MAX=${DEV_MAX:-0})")
PY
fi

# 2. routed SAE ---------------------------------------------------------------
if [ ! -f $CK/routed_jepa_sae.pt ]; then
  log "train routed SAE (groups: ${NUM_GROUPS}=given/kmeans, routing $ROUTING)"
  $PY scripts/training/train_routed_sae.py --span-hidden $EX/span_hidden_calib.pt --output-dir $CK \
     --num-groups $NUM_GROUPS --routing-mode $ROUTING ${GROUP_FIELD:+--group-field $GROUP_FIELD} \
     --epochs 80 --finetune-epochs 40 --seed $SEED --device cuda || exit 1
fi

# 3. likelihood fine-tune (optional) -----------------------------------------
if [ "$LIKELIHOOD" = "1" ] && [ ! -f $CK/routed_jepa_sae_lik.pt ]; then
  log "likelihood fine-tune"
  $PY scripts/training/train_residual_likelihood.py --input $CALIB --checkpoint $CK/routed_jepa_sae.pt \
     --span-hidden $EX/span_hidden_calib.pt --output $CK/routed_jepa_sae_lik.pt --model-name $MODEL --layer $LAYER \
     ${GROUP_FIELD:+--group-field $GROUP_FIELD} --epochs ${LIK_EPOCHS:-2} --batch-size 8 --lr 1e-4 --aux-mse ${AUX_MSE:-0.1} --seed $SEED \
     || echo "likelihood FAILED"
fi

# 3b. global-SAE likelihood control (no routing) -------------------------------
if [ "${LIK_GLOBAL:-0}" = "1" ] && [ ! -f $CK/global_jepa_sae_lik.pt ]; then
  log "likelihood fine-tune of the GLOBAL SAE (no-routing control)"
  $PY scripts/training/train_residual_likelihood.py --input $CALIB --checkpoint $CK/global_jepa_sae.pt \
     --span-hidden $EX/span_hidden_calib.pt --output $CK/global_jepa_sae_lik.pt --model-name $MODEL --layer $LAYER \
     ${GROUP_FIELD:+--group-field $GROUP_FIELD} --epochs ${LIK_EPOCHS:-2} --batch-size 8 --lr 1e-4 --aux-mse ${AUX_MSE:-0.1} --seed $SEED \
     || echo "global likelihood FAILED"
fi

# 4. persistent generation grid ----------------------------------------------
ARMS=${ARMS:-"global_mean_delta routed_mean_delta global_jepa_sae routed_jepa_sae"}
[ "${LIK_GLOBAL:-0}" = "1" ] && [ -f $CK/global_jepa_sae_lik.pt ] && ARMS="$ARMS global_jepa_sae_lik"
[ "$LIKELIHOOD" = "1" ] && [ -f $CK/routed_jepa_sae_lik.pt ] && ARMS="$ARMS routed_jepa_sae_lik routed_jepa_sae_lik_shuffled"
# BASELINES=1 appends the main-table baseline arms (FinTS, SteerX, TAP-PER, PLUME, BM25-RAG reference, profile-prompt reference)
[ "${BASELINES:-0}" = "1" ] && ARMS="$ARMS profile_prompt rag_bm25 fints steerx tapper plume"
BASE=$GEN/persistent__global_mean_delta__scale1.0.jsonl
for SCALE in $SCALES; do
  for arm in $ARMS; do
    out=$GEN/persistent__${arm}__scale${SCALE}.jsonl
    [ "$arm" = profile_prompt ] && [ "$SCALE" != "$(echo $SCALES | awk '{print $1}')" ] && continue
    complete $out && continue
    log "persistent $arm scale $SCALE"
    reuse=""; [ -f "$BASE" ] && [ "$out" != "$BASE" ] && reuse="--reuse-baselines-from $BASE"
    ckarg=$CK/$arm.pt; [ "$arm" = profile_prompt ] && ckarg=profile_prompt
    ragarg=""; if [ "$arm" = rag_bm25 ]; then ckarg=rag_bm25; ragarg="--calib $CALIB"; [ "$SCALE" != "$(echo $SCALES | awk '{print $1}')" ] && continue; fi   # reference arm (BM25 top-5 history in the prompt)
    case "$arm" in tapper)  # baseline (TAP-PER): train prefixes + bridge LoRA once with the same objective/data
      ckarg=$CK/tapper.pt
      [ -f $ckarg ] || { log "train TAP-PER"; $PY scripts/training/train_tapper.py --input $CALIB --output $ckarg --model-name $MODEL ${GROUP_FIELD:+--group-field $GROUP_FIELD} --epochs ${LIK_EPOCHS:-2} --seed $SEED 2>&1 | grep -aE "tapper\]|Traceback"; }
      [ "$SCALE" != "$(echo $SCALES | awk '{print $1}')" ] && continue ;; esac   # scale-free arm: run once
    case "$arm" in routed_jepa_sae_lik_lora)  # ours + shared bridge LoRA (r=LORA_RANK) trained jointly with the likelihood objective
      ckarg=$CK/routed_jepa_sae_lik_lora.pt
      [ -f $ckarg ] || { log "likelihood + bridge LoRA"; $PY scripts/training/train_residual_likelihood.py --input $CALIB --checkpoint $CK/routed_jepa_sae.pt \
         --span-hidden $EX/span_hidden_calib.pt --output $ckarg --model-name $MODEL --layer $LAYER ${GROUP_FIELD:+--group-field $GROUP_FIELD} \
         --epochs ${LIK_EPOCHS:-2} --batch-size 8 --lr 1e-4 --aux-mse ${AUX_MSE:-0.1} --bridge-lora ${LORA_RANK:-8} --seed $SEED 2>&1 | grep -aE "likelihood\]|Traceback"; } ;; esac
    case "$arm" in tapper_lora_only)  # control for TAP-PER: the bridge LoRA alone (task adaptation, no user information)
      ckarg=$CK/tapper_lora_only.pt
      [ -f $ckarg ] || { log "train TAP-PER LoRA-only control"; $PY scripts/training/train_tapper.py --lora-only --input $CALIB --output $ckarg --model-name $MODEL ${GROUP_FIELD:+--group-field $GROUP_FIELD} --epochs ${LIK_EPOCHS:-2} --seed $SEED 2>&1 | grep -aE "tapper\]|Traceback"; }
      [ "$SCALE" != "$(echo $SCALES | awk '{print $1}')" ] && continue ;; esac
    case "$arm" in softprompt)  # baseline: per-user soft prompt (prompt tuning), backbone frozen, same data/objective/epochs
      ckarg=$CK/softprompt.pt
      [ -f $ckarg ] || { log "train soft prompt"; $PY scripts/training/train_tapper.py --prompt-only --lr ${SOFTPROMPT_LR:-1e-2} --input $CALIB --output $ckarg --model-name $MODEL ${GROUP_FIELD:+--group-field $GROUP_FIELD} --epochs ${LIK_EPOCHS:-2} --seed $SEED 2>&1 | grep -aE "tapper\]|Traceback"; }
      [ "$SCALE" != "$(echo $SCALES | awk '{print $1}')" ] && continue ;; esac
    case "$arm" in learned_vectors)  # baseline: one trainable steering vector per user group (init. group mean), likelihood objective, backbone frozen
      ckarg=$CK/learned_vectors.pt
      [ -f $ckarg ] || { log "train learned group steering vectors"; $PY scripts/training/train_residual_likelihood.py --input $CALIB --checkpoint $CK/routed_mean_delta.pt \
         --span-hidden $EX/span_hidden_calib.pt --output $ckarg --model-name $MODEL --layer $LAYER ${GROUP_FIELD:+--group-field $GROUP_FIELD} \
         --epochs ${LIK_EPOCHS:-2} --batch-size 8 --lr 1e-4 --aux-mse ${AUX_MSE:-0.1} --constants-only --seed $SEED 2>&1 | grep -aE "likelihood\]|Traceback"; }
      [ "$SCALE" != "$(echo $SCALES | awk '{print $1}')" ] && continue ;; esac
    case "$arm" in plume)   # baseline (PLUME): shared task LoRA + per-user mixers, same objective/data
      ckarg=$CK/plume.pt
      [ -f $ckarg ] || { log "train PLUME"; $PY scripts/training/train_plume.py --input $CALIB --output $ckarg --model-name $MODEL ${GROUP_FIELD:+--group-field $GROUP_FIELD} --user-epochs ${LIK_EPOCHS:-2} --seed $SEED 2>&1 | grep -aE "plume\]|Traceback"; }
      [ "$SCALE" != "$(echo $SCALES | awk '{print $1}')" ] && continue ;; esac   # scale-free arm: run once
    case "$arm" in steerx)  # baseline (SteerX style vector, training-free): build per-user vectors once; scale grid = gamma
      ckarg=$CK/steerx.pt
      [ -f $ckarg ] || { log "build SteerX vectors"; $PY scripts/eval/build_steerx_vectors.py --input $CALIB --output $ckarg --model-name $MODEL --layer $LAYER ${GROUP_FIELD:+--group-field $GROUP_FIELD} --max-per-user ${STEERX_MAX_PER_USER:-20} --seed $SEED 2>&1 | grep -aE "steerx\]|Traceback"; } ;; esac
    case "$arm" in fints)   # baseline (FinTS, training-free): build the per-user store once
      ckarg=$CK/fints_store.pt
      [ -f $ckarg ] || { log "build FinTS store"; $PY scripts/eval/build_fints_store.py --input $CALIB --output $ckarg --model-name $MODEL --layer $LAYER ${GROUP_FIELD:+--group-field $GROUP_FIELD} --max-per-user ${FINTS_MAX_PER_USER:-200} --seed $SEED 2>&1 | grep -aE "fints\]|Traceback"; } ;; esac
    case "$arm" in *_shuffled)  # control: same experts, routing permuted across users
      base=${arm%_shuffled}; [ -f $CK/$base.pt ] || { echo "skip $arm (no $base.pt)"; continue; }
      [ -f $ckarg ] || $PY scripts/eval/coldstart_route.py --checkpoint $CK/$base.pt --shuffle --seed $SEED --output $ckarg 2>&1 | grep -aE "shuffle|Traceback" ;; esac
    $PY scripts/eval/persistent_steer.py --input $EX/dev_subset.jsonl --checkpoint $ckarg --output $out \
       --model-name $MODEL --layer $LAYER --predictor-tokens 3 --residual-scale $SCALE --max-new-tokens $MAX_NEW \
       ${GROUP_FIELD:+--group-field $GROUP_FIELD} --seed $SEED $reuse $ragarg --resume || echo "FAILED $arm $SCALE"
  done
done

# 5. single-anchor contrast ---------------------------------------------------
for arm in routed_jepa_sae routed_mean_delta; do
  [ "${SKIP_ANCHOR:-0}" = "1" ] && { log "single-anchor skipped (SKIP_ANCHOR=1)"; break; }
  out=$GEN/anchor__${arm}__scale3.0.jsonl
  complete $out && continue
  log "single-anchor $arm"
  reuse=""; [ -f "$BASE" ] && reuse="--reuse-baselines-from $BASE"
  $PY scripts/eval/evaluate.py --input $EX/dev_subset.jsonl --jepa-checkpoint $CK/$arm.pt --output $out \
     --model-name $MODEL --predictor-tokens 3 --layer $LAYER --residual-scale 3.0 --max-new-tokens $MAX_NEW \
     ${GROUP_FIELD:+--group-field $GROUP_FIELD} --judge-provider none --seed $SEED --device cuda \
     $reuse --resume || echo "FAILED anchor $arm"
done

# 5b. held-out calibration NLL (K selection; only with CALIB_HOLDOUT) -----------
if [ "$CALIB_HOLDOUT" != "0" ] && [ "${SKIP_HOLDOUT_NLL:-0}" != "1" ] && [ ! -f $EX/holdout_nll.json ]; then
  cks=""; for c in global_mean_delta routed_mean_delta routed_jepa_sae routed_jepa_sae_lik; do [ -f $CK/$c.pt ] && cks="$cks $c=$CK/$c.pt"; done
  log "held-out calibration NLL:$cks"
  $PY scripts/eval/evaluate_teacher_forced_nll.py --input $EX/calib_holdout.jsonl --checkpoints $cks --output $EX/holdout_nll.json \
     --model-name $MODEL --layer $LAYER --predictor-tokens 3 --residual-scale 1.0 --max-examples 500 ${GROUP_FIELD:+--group-field $GROUP_FIELD} --seed $SEED || echo "FAILED holdout nll"
fi

# 6. scoring -------------------------------------------------------------------
for out in $GEN/*.jsonl; do
  b=${out%.jsonl}
  # never score a partial file: another process may still be writing it, and a stale score would never be recomputed
  [ "$(wc -l < "$out")" -ge "$(wc -l < $EX/dev_subset.jsonl)" ] || { echo "skip scoring incomplete $out"; continue; }
  [ -f ${b}_text_metrics.json ] || $PY scripts/eval/evaluate_text_metrics.py --input $out --output ${b}_text_metrics.json >/dev/null 2>&1
  case "$METRIC" in
    synper_persona)
      [ -f ${b}_persona_strong.csv ] || $PY scripts/eval/evaluate_synper_persona_classifier_strong.py --train $CALIB \
         --generations $out --output ${b}_persona_strong.csv --confusion-output ${b}_persona_strong_confusion.csv \
         --candidate-keys raw_generic_output jepa_steered_output >/dev/null 2>&1 ;;
    userstyle)
      [ -f ${b}_userstyle.csv ] || $PY scripts/eval/evaluate_group_style_classifier_strong.py --train $CALIB --generations $out \
         --label-field ${GROUP_FIELD:-metadata.user_id} --dataset $DATASET --output ${b}_userstyle.csv \
         --confusion-output ${b}_userstyle_confusion.csv >/dev/null 2>&1 ;;
    classify)
      [ -f ${b}_classify.json ] || $PY scripts/eval/evaluate_classification.py --input $out --output ${b}_classify.json >/dev/null 2>&1 ;;
    text) : ;;
  esac
done
$PY scripts/ablations/summarize_stage1.py $GEN | tee $EX/summary_seed${SEED}.txt
log "STAGE ALL DONE dataset=$DATASET model=$MODEL"
