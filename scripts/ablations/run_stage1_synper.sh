#!/usr/bin/env bash
# Qwen2.5-1.5B (layer 14) SynPer setup used for the mechanism/ablation studies;
# a thin wrapper over the generic driver.
#   RUN_TAG=_lik LIKELIHOOD=1 bash scripts/ablations/run_stage1_synper.sh
export DATASET=synper MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}" LAYER="${LAYER:-14}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_stage_dataset.sh"
