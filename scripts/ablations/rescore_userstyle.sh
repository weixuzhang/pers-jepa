#!/usr/bin/env bash
# Scoring-only pass: user-attribution (METRIC=userstyle) scores for every COMPLETE
# generation file of the given run dirs, classifier trained ONCE per run dir
# (evaluate_group_style_classifier_strong.py multi-file mode). CPU only, no GPU;
# safe to run next to a still-running generation job (incomplete files are skipped,
# nothing is deleted). Idempotent: files with *_userstyle.csv are skipped.
#   DATASET=lamp7 RUN_TAGS=_k5,_k10 bash scripts/ablations/rescore_userstyle.sh   (comma or space separated)
set -uo pipefail
if [ -z "${PERSJEPA_FROZEN_DRIVER:-}" ]; then
  _frozen="$(mktemp "${TMPDIR:-/tmp}/persjepa_driver_XXXXXX.sh")"; cp "${BASH_SOURCE[0]}" "$_frozen"
  PERSJEPA_FROZEN_DRIVER="$_frozen" exec bash "$_frozen" "$@"
fi
D="${PERSJEPA_ROOT:-$PWD}"; cd "$D"   # run from the repository root
PY="${PYTHON:-python}"
DATASET=${DATASET:?set DATASET}; MODEL=${MODEL:-Qwen/Qwen3.5-4B}; LAYER=${LAYER:-16}; RUN_TAGS=${RUN_TAGS:?set RUN_TAGS}
GROUP_FIELD=${GROUP_FIELD:-metadata.user_id}; SEED=${SEED:-42}; RUN_TAGS=${RUN_TAGS//,/ }   # comma or space separated
# MAXFEAT: TF-IDF cap per vectorizer; scipy L-BFGS-B segfaults when n_classes x n_features exceeds ~1e8 (LaMP-7: 1500 users -> 20000)
case "$DATASET" in
  lamp7) DEFCALIB=data/lamp7/train_calib.jsonl ;; lamp2) DEFCALIB=data/lamp2/train_calib.jsonl ;; lamp4) DEFCALIB=data/lamp4/train_calib.jsonl ;; lamp5) DEFCALIB=data/lamp5/train_calib.jsonl ;;
  amazon_movies_tv) DEFCALIB=data/amazon_movies_tv/calibration.jsonl ;; *) DEFCALIB=${CALIB:?CALIB required} ;;
esac
MSLUG=$(echo "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9_.\n' '_')
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
for tag in $RUN_TAGS; do
  EX=runs/${DATASET}/${MSLUG}_L${LAYER}${tag}; GEN=$EX/generation_seed${SEED}
  [ -d "$GEN" ] || { log "skip $EX (no generation dir)"; continue; }
  CALIB=$DEFCALIB; [ -f $EX/calib_train.jsonl ] && CALIB=$EX/calib_train.jsonl   # CALIB_HOLDOUT runs trained on calib_train
  CALIB=${TRAIN:-$CALIB}   # TRAIN= overrides (e.g. lamp7: the 1500 dev users own pairs only -> 1500-way attribution, half the memory)
  n=$(wc -l < $EX/dev_subset.jsonl); todo=""
  for f in $GEN/*.jsonl; do
    b=${f%.jsonl}; [ -f ${b}_userstyle.csv ] && continue
    [ "$(wc -l < $f)" -ge "$n" ] || { log "incomplete (still generating?) $f — skipped"; continue; }
    todo="$todo $f"
  done
  [ -z "$todo" ] && { log "$EX: nothing to score"; continue; }
  log "$EX: scoring $(echo $todo | wc -w) files, train=$CALIB"
  $PY scripts/eval/evaluate_group_style_classifier_strong.py --train $CALIB --generations $todo \
     --label-field $GROUP_FIELD --dataset $DATASET ${MAXFEAT:+--max-features $MAXFEAT} 2>&1 | grep -vE "FutureWarning|warnings.warn"; first=$(echo $todo | awk '{print $1}'); [ -f "${first%.jsonl}_userstyle.csv" ] || echo "FAILED userstyle $EX"
  # coldstart_users dir, if present
  CG=$EX/coldstart_users/generation_seed${SEED}
  if [ -d "$CG" ]; then
    todo=""; for f in $CG/*.jsonl; do b=${f%.jsonl}; [ -f ${b}_userstyle.csv ] || [ "$(wc -l < $f)" -lt "$n" ] || todo="$todo $f"; done
    [ -n "$todo" ] && { log "$CG: scoring $(echo $todo | wc -w) files"; $PY scripts/eval/evaluate_group_style_classifier_strong.py --train $CALIB --generations $todo --label-field $GROUP_FIELD --dataset $DATASET ${MAXFEAT:+--max-features $MAXFEAT} 2>&1 | grep -vE "FutureWarning|warnings.warn"; }
  fi
done
log "RESCORE USERSTYLE DONE ($DATASET: $RUN_TAGS)"
