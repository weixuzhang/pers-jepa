#!/usr/bin/env bash
# Feature cards / expert divergence / logit lens on a routed JEPA-SAE
# likelihood checkpoint (scripts/interpretability/routed_expert_features.py).
# Idempotent; loads the ~5 GB span-hidden extraction, so run it on a machine with enough RAM. Env: DATASET (synper) MODEL LAYER RUN_TAG CK (routed_jepa_sae_lik) GROUP_FIELD
set -uo pipefail
if [ -z "${PERSJEPA_FROZEN_DRIVER:-}" ]; then
  _frozen="$(mktemp "${TMPDIR:-/tmp}/persjepa_driver_XXXXXX.sh")"; cp "${BASH_SOURCE[0]}" "$_frozen"
  PERSJEPA_FROZEN_DRIVER="$_frozen" exec bash "$_frozen" "$@"
fi
D="${PERSJEPA_ROOT:-$PWD}"; cd "$D"   # run from the repository root
PY="${PYTHON:-python}"
DATASET=${DATASET:-synper}; MODEL=${MODEL:-Qwen/Qwen3.5-4B}; LAYER=${LAYER:-16}; RUN_TAG=${RUN_TAG:-}; CKN=${CK:-routed_jepa_sae_lik}
MSLUG=$(echo "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9_.\n' '_')
SRC=${SRC:-runs/${DATASET}/${MSLUG}_L${LAYER}${RUN_TAG}}; OUT=$SRC/interp_${CKN#routed_jepa_sae_}
log() { echo "=== [$(date +%H:%M:%S)] $*"; }
[ -f "$OUT/DONE" ] && { log "skip: $OUT exists"; exit 0; }
log "interp: $SRC/ckpt/$CKN.pt -> $OUT"
$PY scripts/interpretability/routed_expert_features.py --checkpoint $SRC/ckpt/$CKN.pt --span-hidden $SRC/span_hidden_calib.pt \
   --output-dir $OUT --model-name $MODEL ${GROUP_FIELD:+--group-field $GROUP_FIELD} --device ${DEVICE:-cuda} && touch $OUT/DONE || { echo "FAILED interp"; exit 1; }
log "INTERP DONE ($OUT)"
