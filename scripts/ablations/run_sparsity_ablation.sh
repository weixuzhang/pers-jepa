#!/usr/bin/env bash
# Focused ablations on a finished SynPer run (reference A = its
# likelihood-trained routed SAE): sparse/dense x MSE-pretrained/centroid-init,
# plus genuine learned group constants. One arm per invocation (one per GPU):
#   E1  dense predictor (top_k = latent_dim = 128), MSE-pretrained, likelihood-trained
#   E2  sparse predictor, NO MSE pretraining (centroid init), likelihood-trained
#   E3  learned group constants only (delta = sum_g w_g v_g, v_g trainable)
#   E4  dense predictor, NO MSE pretraining (centroid init), likelihood-trained
#   E5  E4 with aux-MSE 0: no residual signal beyond routing + group-mean init (true no-residual-signal control)
#   E6  E2 with aux-MSE 0 (sparse counterpart of E5)
#   ARM=E1 SRC=runs/synper/Qwen_Qwen3.5_4B_L16 bash scripts/ablations/run_sparsity_ablation.sh
# Env: ARM (required)  ABL_DIR (output root, default $SRC/abl_sparsity)  DEVFILE (dev override, smoke only)
#       SRC (reference run dir)  MODEL (Qwen/Qwen3.5-4B)  LAYER (16)  SEED (42)
#      SCALE (1.0)  MAX_NEW (64)  LIK_EPOCHS (2)  AUX_MSE (0.1)  LATENT (128)  CALIB (data/synper/train_10000.jsonl)
# Reuses the reference extraction, dev subset, routing table and raw generations; writes to $SRC/abl_sparsity/$ARM/.
# Idempotent (skips finished steps); one python process per step; scale 1 only, no baselines.
set -uo pipefail
if [ -z "${PERSJEPA_FROZEN_DRIVER:-}" ]; then
  _frozen="$(mktemp "${TMPDIR:-/tmp}/persjepa_driver_XXXXXX.sh")"; cp "${BASH_SOURCE[0]}" "$_frozen"
  PERSJEPA_FROZEN_DRIVER="$_frozen" exec bash "$_frozen" "$@"
fi
D="${PERSJEPA_ROOT:-$PWD}"; cd "$D"   # run from the repository root
PY="${PYTHON:-python}"
ARM=${ARM:?set ARM=E1|E2|E3|E4|E5|E6}; SRC=${SRC:-runs/synper/Qwen_Qwen3.5_4B_L16}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}; LAYER=${LAYER:-16}; SEED=${SEED:-42}; SCALE=${SCALE:-1.0}; MAX_NEW=${MAX_NEW:-64}
LIK_EPOCHS=${LIK_EPOCHS:-2}; AUX_MSE=${AUX_MSE:-0.1}; LATENT=${LATENT:-128}; CALIB=${CALIB:-data/synper/train_10000.jsonl}
MAXEX=${MAXEX:-}   # smoke tests only: --max-examples for the likelihood stage
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
log() { echo "=== [$(date -u +%FT%TZ)] [$ARM] $*"; }
RCK=$SRC/ckpt; RGEN=$SRC/generation_seed${SEED}
for f in $SRC/span_hidden_calib.pt $SRC/dev_subset.jsonl $RCK/routed_jepa_sae.pt $RCK/routed_mean_delta.pt $RCK/routed_jepa_sae_lik.pt; do
  [ -f $f ] || { echo "missing reference artifact $f"; exit 1; }; done
A=${ABL_DIR:-$SRC/abl_sparsity}/$ARM; CK=$A/ckpt; GEN=$A/generation_seed${SEED}; mkdir -p $CK $GEN logs
DEVFILE=${DEVFILE:-$SRC/dev_subset.jsonl}; NDEVL=$(wc -l < $DEVFILE)   # DEVFILE override = smoke tests only
T=$A/timing.json; [ -f $T ] || echo "{}" > $T
stamp() { $PY - "$T" "$1" "$2" <<'PY'
import json,sys; p,k,v=sys.argv[1:]; d=json.load(open(p)); d[k]=float(v); json.dump(d,open(p,"w"),indent=1)
PY
}
log "reference $SRC ($NDEVL dev examples); output $A; commit $(git rev-parse --short HEAD 2>/dev/null)"
LIK="$PY scripts/training/train_residual_likelihood.py --input $CALIB --span-hidden $SRC/span_hidden_calib.pt --model-name $MODEL --layer $LAYER --epochs $LIK_EPOCHS --batch-size 8 --lr 1e-4 --aux-mse $AUX_MSE --seed $SEED --log-every ${LOG_EVERY:-20} ${MAXEX:+--max-examples $MAXEX}"
FINAL=$CK/final.pt
case "$ARM" in
  E1)  # dense, MSE-pretrained (same schedule as the reference: 80 global + 40 per-group epochs), then likelihood
    if [ ! -f $CK/routed_jepa_sae.pt ]; then
      log "MSE pretraining, dense (latent $LATENT, top_k $LATENT)"; t0=$(date +%s)
      $PY scripts/training/train_routed_sae.py --span-hidden $SRC/span_hidden_calib.pt --output-dir $CK --num-groups 0 --routing-mode soft \
         --latent-dim $LATENT --top-k $LATENT --epochs ${MSE_EPOCHS:-80} --finetune-epochs ${MSE_FT_EPOCHS:-40} --seed $SEED --device cuda || exit 1
      stamp mse_pretrain_seconds $(( $(date +%s) - t0 ))
      # reuse the reference routing table exactly (same latents -> same table, but make it byte-identical)
      $PY - $CK/routed_jepa_sae.pt $RCK/routed_jepa_sae.pt <<'PY'
import sys,torch; a,b=sys.argv[1:]; pa=torch.load(a,map_location="cpu"); pb=torch.load(b,map_location="cpu")
assert pa["group_names"]==pb["group_names"], (pa["group_names"],pb["group_names"])
assert int(pa["top_k"])==int(pa["latent_dim"]), pa["top_k"]
pa["routing_table"]=pb["routing_table"]; torch.save(pa,a); print("[E1] routing table copied from the reference;", "top_k", pa["top_k"], "latent", pa["latent_dim"])
PY
    fi
    [ -f $FINAL ] || { log "likelihood (dense)"; t0=$(date +%s); $LIK --checkpoint $CK/routed_jepa_sae.pt --output $FINAL 2>&1 | grep -aE "likelihood\]|Traceback|Error"; stamp lik_train_seconds $(( $(date +%s) - t0 )); } ;;
  E2)  # sparse, no MSE pretraining (centroid init)
    [ -f $FINAL ] || { log "likelihood (sparse, centroid init, no MSE pretraining)"; t0=$(date +%s); $LIK --checkpoint $RCK/routed_jepa_sae.pt --output $FINAL --reinit-experts centroid --reinit-seed $SEED 2>&1 | grep -aE "likelihood\]|Traceback|Error"; stamp lik_train_seconds $(( $(date +%s) - t0 )); } ;;
  E3)  # learned constants only
    [ -f $FINAL ] || { log "likelihood (learned group constants only)"; t0=$(date +%s); $LIK --checkpoint $RCK/routed_mean_delta.pt --output $FINAL --constants-only 2>&1 | grep -aE "likelihood\]|Traceback|Error"; stamp lik_train_seconds $(( $(date +%s) - t0 )); } ;;
  E4)  # dense, no MSE pretraining (same fresh init as E2)
    [ -f $FINAL ] || { log "likelihood (dense, centroid init, no MSE pretraining)"; t0=$(date +%s); $LIK --checkpoint $RCK/routed_jepa_sae.pt --output $FINAL --reinit-experts centroid --reinit-seed $SEED --top-k $LATENT 2>&1 | grep -aE "likelihood\]|Traceback|Error"; stamp lik_train_seconds $(( $(date +%s) - t0 )); } ;;
  E5)  # dense, no MSE pretraining, NO auxiliary residual loss: the no-residual-signal control (routing latents + group-mean init only)
    [ -f $FINAL ] || { log "likelihood (dense, centroid init, no MSE pretraining, aux-MSE 0)"; t0=$(date +%s); ${LIK/--aux-mse $AUX_MSE/--aux-mse 0} --checkpoint $RCK/routed_jepa_sae.pt --output $FINAL --reinit-experts centroid --reinit-seed $SEED --top-k $LATENT 2>&1 | grep -aE "likelihood\]|Traceback|Error"; stamp lik_train_seconds $(( $(date +%s) - t0 )); } ;;
  E6)  # sparse, no MSE pretraining, no auxiliary residual loss (E5's sparse counterpart)
    [ -f $FINAL ] || { log "likelihood (sparse, centroid init, no MSE pretraining, aux-MSE 0)"; t0=$(date +%s); ${LIK/--aux-mse $AUX_MSE/--aux-mse 0} --checkpoint $RCK/routed_jepa_sae.pt --output $FINAL --reinit-experts centroid --reinit-seed $SEED 2>&1 | grep -aE "likelihood\]|Traceback|Error"; stamp lik_train_seconds $(( $(date +%s) - t0 )); } ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
[ -f $FINAL ] || { log "FAILED: no checkpoint"; exit 1; }
# verification: reload behaviour, routing identity, prompt independence (E3), dense setting (E1/E4)
$PY - $FINAL $RCK/routed_jepa_sae_lik.pt $ARM $LATENT <<'PY' || exit 1
import sys,torch; sys.path.insert(0,".")
from persjepa.intervention import load_steering_sae
f,ref,arm,latent=sys.argv[1:]; latent=int(latent)
m,pay=load_steering_sae(f,device="cpu"); r,rpay=load_steering_sae(ref,device="cpu")
tw={k:v for k,v in pay["routing_table"]["weights"].items()}; rw=rpay["routing_table"]["weights"]
assert set(tw)==set(rw) and all(torch.allclose(tw[k].float(),rw[k].float()) for k in tw), "routing table differs from the reference"
assert list(pay["group_names"])==list(rpay["group_names"])
h=torch.randn(4,m.input_dim); w=torch.tensor([0.7,0.3]+[0.0]*(m.num_groups-2))
out=m(h,w); d=out.delta if hasattr(out,"delta") else out-h
if arm=="E3":
    assert pay["model_type"]=="routed_mean_delta" and pay.get("trained_constants"), pay["model_type"]
    assert torch.allclose(d[0],d[1],atol=1e-4) and torch.allclose(d[0],d[3],atol=1e-4), "E3 delta depends on the prompt state"   # (h+delta)-h rounding
    print(f"[verify] E3: routed_mean_delta with trained constants, prompt-independent OK; |v_g| mean {m.group_means.norm(dim=1).mean():.2f}")
else:
    assert int(pay["top_k"])==(latent if arm in ("E1","E4","E5") else int(rpay["top_k"])), (pay["top_k"], arm)
    assert m.top_k==int(pay["top_k"]) and all(e.top_k==m.top_k for e in m.experts)
    z=m.experts[0].encode(h)[0]; nz=(z>0).sum(-1).float().mean()
    print(f"[verify] {arm}: top_k={m.top_k} latent={pay['latent_dim']} reload OK; mean active latents {nz:.1f}; input-dependent: {not torch.allclose(d[0],d[1])}")
lt=pay.get("likelihood_training",{}); print(f"[verify] trainable={lt.get('n_trainable')} train_seconds={lt.get('train_seconds')} final_nll={lt['log'][-1]['nll'] if lt.get('log') else 'n/a'}")
PY
# generation at scale 1 (raw outputs reused from the reference run)
out=$GEN/persistent__${ARM}__scale${SCALE}.jsonl
if ! { [ -f $out ] && [ "$(wc -l < $out)" -ge "$NDEVL" ]; }; then
  [ -f $out ] && rm -f ${out%.jsonl}_*
  log "generate"; t0=$(date +%s)
  BASE=$RGEN/persistent__global_mean_delta__scale1.0.jsonl; reuse=""; [ -f "$BASE" ] && reuse="--reuse-baselines-from $BASE"
  $PY scripts/eval/persistent_steer.py --input $DEVFILE --checkpoint $FINAL --output $out --model-name $MODEL --layer $LAYER \
     --predictor-tokens 3 --residual-scale $SCALE --max-new-tokens $MAX_NEW --seed $SEED $reuse --resume 2>&1 | grep -aE "wrote|Traceback|unknown"
  stamp generation_seconds $(( $(date +%s) - t0 ))
fi
[ "$(wc -l < $out)" -ge "$NDEVL" ] || { log "FAILED: generation incomplete ($(wc -l < $out)/$NDEVL)"; exit 1; }
b=${out%.jsonl}
[ -f ${b}_text_metrics.json ] || $PY scripts/eval/evaluate_text_metrics.py --input $out --output ${b}_text_metrics.json >/dev/null 2>&1
[ -f ${b}_persona_strong.csv ] || $PY scripts/eval/evaluate_synper_persona_classifier_strong.py --train $CALIB --generations $out \
   --output ${b}_persona_strong.csv --confusion-output ${b}_persona_strong_confusion.csv --candidate-keys raw_generic_output jepa_steered_output >/dev/null 2>&1
$PY - ${b}_persona_strong.csv ${b}_text_metrics.json <<'PY'
import csv,json,sys; c,t=sys.argv[1:]; rows={r["candidate_key"]:r for r in csv.DictReader(open(c))}; tm=json.load(open(t))
print(f"[score] persona acc raw {float(rows['raw_generic_output']['persona_accuracy']):.3f} -> steered {float(rows['jepa_steered_output']['persona_accuracy']):.3f}; task F1 {tm['raw_generic_output']['token_f1']:.3f} -> {tm['jepa_steered_output']['token_f1']:.3f}; n={rows['jepa_steered_output']['n_examples']}")
PY
cp -f $FINAL $A/ckpt_${ARM}.pt 2>/dev/null; git rev-parse HEAD > $A/commit.txt 2>/dev/null
echo "{\"arm\": \"$ARM\", \"src\": \"$SRC\", \"model\": \"$MODEL\", \"layer\": $LAYER, \"seed\": $SEED, \"scale\": $SCALE, \"lik_epochs\": $LIK_EPOCHS, \"aux_mse\": $AUX_MSE, \"latent\": $LATENT, \"reference_ckpt\": \"$RCK/routed_jepa_sae_lik.pt\"}" > $A/config.json
log "ARM DONE ($ARM) -> $A"
