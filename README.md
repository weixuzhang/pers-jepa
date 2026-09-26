# Pers-JEPA

Pers-JEPA personalizes a **frozen** LLM without showing any profile text at
inference time. It learns, in hidden-state space, the residual between a
generic view of a prompt and a profile-conditioned view of the same prompt,
and injects a predicted residual into the model during generation.

## Method overview

1. **Paired views.** For each example build a generic prompt (task only) and
   a profile-conditioned prompt (profile/history/persona + the same task).
2. **Residual extraction.** Append `k` `[PRED]` tokens to the generic prompt
   and read the layer-`ℓ` state at the anchor (`h_gen`); the
   profile-conditioned view gives the target, and the residual
   `r = h_pers − h_gen` is also averaged over answer-span positions under
   teacher forcing (`scripts/training/extract_span_hidden.py`).
3. **Pers-JEPA-SAE.** A TopK sparse autoencoder maps `h_gen` to a predicted
   residual and is trained with the JEPA-style objective
   `‖h_gen + Δ(h_gen) − h_pers‖²`. Restricting `Δ` to a constant recovers
   mean-difference steering (the constant-vector baselines).
4. **User-routed experts.** Each user gets a compact latent `z_u` (the
   task-centered mean of its calibration residuals, never profile text).
   Users are grouped by given labels (SynPer personas) or by k-means on
   `z_u`; one JEPA-SAE expert per group is initialised from the global SAE
   (offset so it starts at the per-group constant) and users are routed
   hard or softly (`softmax(−‖z_u − c_g‖² / τ)`).
5. **Persistent injection.** The predicted residual is added at a middle
   layer to every position, during the prompt pass and at every decoding step.
6. **Likelihood training.** The routed experts are fine-tuned through the
   frozen LM for the likelihood of the user's own answers, with the residual
   MSE as an auxiliary term (`λ = 0.1`).
7. **Cold start.** A new user is routed from `z_u` computed on `k` of its own
   examples (`k = 0` gives uniform routing), without retraining.

Code map:

| Component | Where |
|---|---|
| JEPA-SAE, routed SAE, constant predictors | `persjepa/models.py` |
| User latents, k-means groups, routing tables | `persjepa/routing.py` |
| Persistent (all-position) injection hooks | `persjepa/persistent.py` |
| Single-anchor injection (earlier estimator family) | `persjepa/intervention.py` |
| Chat-template handling (instruct models) | `persjepa/chat.py` |
| Baselines: FinTS / PLUME / TAP-PER / BM25 retrieval | `persjepa/{fints,plume,tapper,rag}.py` |
| Routed-SAE training | `scripts/training/train_routed_sae.py` |
| Likelihood fine-tuning through the frozen LM | `scripts/training/train_residual_likelihood.py` |
| Persistent-injection generation (all arms) | `scripts/eval/persistent_steer.py` |
| End-to-end driver (extraction → training → generation → scoring) | `scripts/ablations/run_stage_dataset.sh` |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e ".[judge]"
python scripts/smoke_check.py        # quick sanity check on tiny inputs
```

Python ≥ 3.10. The main model (Qwen3.5-4B) needs `transformers>=5`.
Llama-3.1-8B-Instruct is gated on the Hugging Face Hub and needs an access
token. The optional API judges read `OPENAI_API_KEY` (and optionally
`OPENAI_BASE_URL`) from `.env.local`; copy `.env.example` to start.

All commands below are run from the repository root. Data goes under
`data/` and all outputs under `runs/`; neither is tracked.

## Datasets

The driver expects these files:

| `DATASET` | calibration (`CALIB`) | evaluation (`DEV`) | default metric |
|---|---|---|---|
| `synper` | `data/synper/train_10000.jsonl` | `data/synper/dev_1000.jsonl` | persona classifier + task F1 |
| `lamp2` | `data/lamp2/train_calib.jsonl` | `data/lamp2/dev.jsonl` | tag accuracy |
| `lamp4`, `lamp5` | `data/lamp{4,5}/train_calib.jsonl` | `data/lamp{4,5}/dev.jsonl` | ROUGE + user attribution |
| `lamp7` | `data/lamp7/train_calib.jsonl` | `data/lamp7/dev.jsonl` | ROUGE + user attribution |
| `amazon_movies_tv` | `data/amazon_movies_tv/calibration.jsonl` | `data/amazon_movies_tv/heldout.jsonl` | ROUGE + user attribution |

Any other dataset works if you set `CALIB`/`DEV`. Records are JSON lines with
`id`, `profile`, `prompt`, `target` and `metadata` (for example
`metadata.user_id` or `metadata.persona`). `scripts/data/prepare_dataset.py`
converts a generic jsonl.

**SynPer** (10 personas, 10,000 train / 1,000 dev, plus a same-task
counterfactual split). The benchmark splits will be released with the paper.
To regenerate them with an OpenAI-compatible API:

```bash
python scripts/data/generate_synper.py --output data/synper/raw.jsonl --num-examples 11000
python scripts/data/split_synper_dataset.py --input data/synper/raw.jsonl \
  --train-output data/synper/train_10000.jsonl --dev-output data/synper/dev_1000.jsonl
python scripts/data/audit_clean_synper.py --help                 # cleaning / audit
python scripts/data/build_synper_counterfactual_split.py --help  # counterfactual split
```

**LaMP-2 / LaMP-4 / LaMP-5.** Download the official `{train,dev}_{questions,outputs}.json`
into `data/raw/lamp$T/`. Each user's calibration pairs are built from their
profile items:

```bash
T=2   # also 4, 5
python scripts/data/prepare_lamp.py --task $T --split dev --expand-profile-pairs 8 \
  --questions data/raw/lamp$T/dev_questions.json --outputs data/raw/lamp$T/dev_outputs.json \
  --output data/lamp$T/dev_expanded.jsonl
python scripts/data/prepare_lamp.py --task $T --split train --expand-profile-pairs 8 --max-examples 1500 \
  --questions data/raw/lamp$T/train_questions.json --outputs data/raw/lamp$T/train_outputs.json \
  --output data/lamp$T/train_expanded.jsonl
python scripts/data/split_lamp_expanded.py --dev-expanded data/lamp$T/dev_expanded.jsonl \
  --train-expanded data/lamp$T/train_expanded.jsonl --out-dir data/lamp$T
```

**LaMP-7.** Profile tweets have no paired input, so a neutral paraphrase is
generated with the local model:

```bash
python scripts/data/prepare_lamp.py --task 7 --split dev \
  --questions data/raw/lamp7/dev_questions.json --outputs data/raw/lamp7/dev_outputs.json \
  --output data/lamp7/dev.jsonl
python scripts/data/prepare_lamp.py --task 7 --split dev --keep-raw-profile \
  --questions data/raw/lamp7/dev_questions.json --outputs data/raw/lamp7/dev_outputs.json \
  --output data/lamp7/dev_rawprofile.jsonl
python scripts/data/prepare_lamp.py --task 7 --split train --keep-raw-profile --max-examples 1500 \
  --questions data/raw/lamp7/train_questions.json --outputs data/raw/lamp7/train_outputs.json \
  --output data/lamp7/train_rawprofile.jsonl
bash scripts/ablations/run_lamp7_neutralize.sh     # GPU; writes data/lamp7/train_calib.jsonl
```

**Amazon Reviews 2023, Movies & TV** (warm users; the last 5 reviews per user
are held out):

```bash
python scripts/data/prepare_amazon_reviews_raw_jsonl.py \
  --local-review-file data/raw/amazon/Movies_and_TV.jsonl.gz \
  --local-meta-file data/raw/amazon/meta_Movies_and_TV.jsonl.gz \
  --review-file raw/review_categories/Movies_and_TV.jsonl \
  --meta-file raw/meta_categories/meta_Movies_and_TV.jsonl \
  --min-reviews 50 --prefilter-users --max-users 800 --require-item-title \
  --examples-per-user 15 --max-examples 100000 --seed 42 --output data/amazon_movies_tv/all.jsonl
python scripts/data/split_user_calibration_heldout.py --input data/amazon_movies_tv/all.jsonl \
  --calibration-out data/amazon_movies_tv/calibration.jsonl \
  --heldout-out data/amazon_movies_tv/heldout.jsonl --heldout-per-user 5 --min-calibration 5
```

Without the `--local-*` flags the raw files are downloaded from
`McAuley-Lab/Amazon-Reviews-2023` on the Hugging Face Hub.

## Running the pipeline

`scripts/ablations/run_stage_dataset.sh` runs one dataset × model end to end
and is configured with environment variables (see the header of the script).
It skips steps that have already finished, so it is safe to re-run after an
interruption. The steps are:

1. span-hidden extraction on the calibration set at `LAYER`;
2. global JEPA-SAE and routed JEPA-SAE (+ constant baselines) training;
3. likelihood fine-tuning of the routed SAE (`LIKELIHOOD=1`), plus the
   no-routing control (`LIK_GLOBAL=1`);
4. persistent-injection generation for every arm and scale, including the
   permuted-routing control (`*_shuffled`) and the baselines (`BASELINES=1`);
5. an optional single-anchor contrast and a held-out-calibration NLL used to
   choose `K` (`CALIB_HOLDOUT`);
6. scoring and a summary table (`runs/<dataset>/<model>_L<layer><tag>/summary_seed<seed>.txt`).

**SynPer, main model** (the given personas are the groups):

```bash
DATASET=synper MODEL=Qwen/Qwen3.5-4B LAYER=16 NDEV=100 \
LIKELIHOOD=1 LIK_EPOCHS=2 LIK_GLOBAL=1 BASELINES=1 \
  bash scripts/ablations/run_stage_dataset.sh
```

Other backbones: `MODEL=Qwen/Qwen3.5-9B LAYER=16`,
`MODEL=meta-llama/Llama-3.1-8B-Instruct LAYER=16`, and the mechanism/ablation
model `MODEL=Qwen/Qwen2.5-1.5B LAYER=14` (`scripts/ablations/run_stage1_synper.sh`).
`python scripts/training/model_layer_info.py <model>` prints the number of layers.
Seeds: `SEED=7`, `SEED=13` with a distinct `RUN_TAG`.

**Real users** (k-means groups on the user latents, soft routing):

```bash
DATASET=lamp2 MODEL=Qwen/Qwen3.5-4B LAYER=16 NUM_GROUPS=20 ROUTING=soft \
GROUP_FIELD=metadata.user_id LIKELIHOOD=1 LIK_EPOCHS=2 LIK_GLOBAL=1 BASELINES=1 \
CALIB_HOLDOUT=0.05 SCALES=1.0 SKIP_ANCHOR=1 FINTS_MAX_PER_USER=50 STEERX_MAX_PER_USER=10 \
RUN_TAG=_k20 bash scripts/ablations/run_stage_dataset.sh
```

Use the same command for `lamp4`, `lamp5` and `lamp7` (`NUM_GROUPS=20`), and
for `amazon_movies_tv` use `NUM_GROUPS=10 NDEV=50 RUN_TAG=_k10`.

**K curve without labels (SynPer):** `NUM_GROUPS=3|5|8 ROUTING=soft CALIB_HOLDOUT=0.05 LIKELIHOOD=1`.
The held-out NLL is written to `holdout_nll.json` in the run directory.

**Additional trained baselines.** The per-user soft prompt (OPPU-style) and
the trained per-group steering vectors (BiPO-style) are extra arms. To add
them, append `softprompt learned_vectors` to `ARMS`. `tapper_lora_only` (the
bridge-LoRA-only control) and `routed_jepa_sae_lik_lora` (ours + a shared
bridge LoRA) are also available. Setting `ARMS` replaces the default arm
list, so include the arms you want.

### Ablations and analyses

| Experiment | Command |
|---|---|
| Sparse vs. dense, MSE pretraining, learned group constants | `ARM=E1..E6 SRC=<synper run dir> bash scripts/ablations/run_sparsity_ablation.sh`; `python scripts/ablations/compare_ablation_arms.py --help` |
| Objective / routing ablations and seeds (Qwen2.5-1.5B) | `SRC=<run dir> bash scripts/ablations/run_legacy_lik_ablations.sh`, `run_legacy_seeds_and_k.sh` |
| Cold start: held-out personas | `HOLDOUT="The Pirate,The Systems Engineer" bash scripts/ablations/run_coldstart.sh` |
| Cold start: new users of an existing population | `DATASET=lamp2 RUN_TAG=_k20 GROUP_FIELD=metadata.user_id bash scripts/ablations/run_coldstart_users.sh` |
| Mean-of-predictions constant control | `python scripts/ablations/mean_of_ours_constant.py --help` |
| Routed-expert feature cards / logit lens | `bash scripts/ablations/run_interp_experts.sh` |
| Bootstrap CIs on real users | `python scripts/eval/bootstrap_real_data.py --help` |
| Hidden-space estimator family, predictor-token and objective sweeps | `scripts/ablations/run_hidden_ablation.py`, `run_residual_variant_ablation.py`, `run_span_stp_ablation.py`, `analyze_personalization_residuals.py` |
| Residual-scale calibration | `python scripts/training/calibrate_residual_scale.py --help` |
| Sparse-latent interpretability (global SAE) | `scripts/interpretability/*.py` |

The earlier estimator family (single-anchor final-layer injection,
`Pers-JEPA-Latent`, dense profile vectors) is driven by
`scripts/training/{extract_hidden,train_saes}.py`, `scripts/eval/evaluate.py`,
`scripts/eval/evaluate_group_residual.py` and
`scripts/eval/evaluate_dense_profile_vector.py`.

### Scoring

The driver scores automatically. The scorers can also be run on their own:

- `scripts/eval/evaluate_text_metrics.py`: token F1 / ROUGE-1 / ROUGE-L.
- `scripts/eval/evaluate_synper_persona_classifier_strong.py`: the SynPer
  persona classifier (char/word TF-IDF). Always pass explicit
  `--output/--confusion-output` paths.
- `scripts/eval/evaluate_group_style_classifier_strong.py`: user attribution.
- `scripts/eval/evaluate_classification.py`: LaMP-2 accuracy.
- `scripts/eval/judge_generation.py`, `scripts/eval/judge_persona_style.py`:
  pairwise / style judges (local or OpenAI-compatible).

## Repository layout

```
persjepa/            core library
scripts/
  data/              dataset preparation (SynPer, LaMP-2/4/5/7, Amazon, generic)
  training/          hidden extraction, (routed) SAE training, likelihood training, baselines
  eval/              persistent / single-anchor generation, scorers, judges, cold start, bootstrap
  ablations/         end-to-end driver, ablation drivers, residual analyses
  interpretability/  sparse-latent and routed-expert analyses
  smoke_check.py     end-to-end sanity check
configs/             default configuration for the single-anchor pipeline
```

## License

MIT; see [LICENSE](LICENSE).
