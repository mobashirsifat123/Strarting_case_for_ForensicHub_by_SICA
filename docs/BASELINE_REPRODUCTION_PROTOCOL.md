# SICA Baseline Reproduction Protocol

**Status:** DEFINED ONLY. None of these experiments are run by this prep pass (no GPU job launched). Each experiment is specified so it can be executed verbatim when GPUs are available.

**Conventions**
- `REPO = /mnt/nas/users/moba/projects/SICA_OpenMMSec`
- `FH = $REPO/ForensicHub` (PYTHONPATH), `PKG = $FH/ForensicHub` (cwd)
- `PY = /home/moba/envs/sica_py310/bin/python`, `TORCHRUN = /home/moba/envs/sica_py310/bin/torchrun`
- Effective batch = `batch_size × accum_iter × world_size`.
- All outputs go under `reproduction_artifacts/` in run-specific directories (see `RUN_NAMING_STANDARD.md`).
- Always launch via `torchrun` (even 1 GPU) so DDP/evaluators initialize (`dist.all_gather` in `ImageAP`).
- Run on **idle** GPUs for any multi-GPU experiment (sharing crashes large NCCL collectives - see `SICA_4GPU_DDP_DIAGNOSIS.md`).

---

## A. Single-GPU engineering validation

| Field | Value |
|---|---|
| Purpose | Prove the full SICA pipeline (data → model → train step → validate → checkpoint) runs end-to-end on one GPU, quickly, on a cached subset. Confirms the runtime fix holds and the evaluator/test path does not crash. |
| GPUs | 1 (any free GPU; coexists with other users' jobs) |
| Batch size | 24 (per GPU) |
| Accumulation | 1 |
| Effective global batch | 24 |
| Epochs | 1 |
| Config | `reproduction_artifacts/configs/sica_train_sanity.yaml` (cached subset; 1 epoch) |
| Command/script | `bash reproduction_artifacts/scripts/run_sica_1gpu.sh` **after** editing the script's `CONFIG=` to `sica_train_sanity.yaml` (or run the config directly): <br>`CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FH" "$TORCHRUN" --nproc_per_node=1 --master_port=29557 training_scripts/train.py --config "$REPO/reproduction_artifacts/configs/sica_train_sanity.yaml"` (cwd = `$PKG`) |
| Expected outputs | `logs/sica_sanity/checkpoint-0.pth`; `logs/sica_sanity/log.txt` (one JSON line: train_lr, losses, test AUC/AP/Acc); TensorBoard event file; per-metric `*_Metric_validation/` dirs |
| Success criteria | Exit 0; 1 full epoch + 1 validation pass completed; `log.txt` contains finite AUC/AP/Acc; checkpoint saved; no `NameError`/`KeyError`/NCCL crash |
| Stop conditions | Exit 0 (success); or any uncaught exception / OOM / hang > 5 min on the subset (diagnose, do not retry blindly) |
| Status | ALREADY PASSED (2026-07-22): AUC 0.9249, AP 0.9431, Acc 0.8590 on the sanity subset. Re-run only if the runtime changes. |

---

## B. Four-GPU DDP smoke test

| Field | Value |
|---|---|
| Purpose | Verify the 4-rank DDP path (NCCL init, parameter broadcast, DistributedSampler sharding, gradient all-reduce, multi-rank validation, checkpoint) works end-to-end on **idle** GPUs before committing to the full 10-epoch run. |
| GPUs | 4 (must be **idle** - no other large-NCCL job sharing them) |
| Batch size | 24 (per GPU) |
| Accumulation | 1 |
| Effective global batch | 96 |
| Epochs | 1 |
| Config | `reproduction_artifacts/configs/sica_4gpu_smoke.yaml` (cached subset; 1 epoch) |
| Command/script | `bash reproduction_artifacts/scripts/run_sica_4gpu_smoke.sh` |
| Expected outputs | `logs/sica_4gpu_smoke/checkpoint-0.pth`; `log.txt`; TensorBoard event; all 4 ranks print "distributed init (rank 0..3)"; no illegal-memory-access |
| Success criteria | Exit 0; 4 ranks initialize; 1 epoch + validation completes; checkpoint written by rank 0; `log.txt` has finite metrics |
| Stop conditions | Exit 0 (success); or `CUDA error: an illegal memory access` at DDP construction ⇒ GPUs are still shared (wait for idle) **or** try `NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1` and re-diagnose; OOM ⇒ lower per-GPU batch |
| Prerequisite | GPUs confirmed idle (e.g., `nvidia-smi` shows no other compute processes) |

---

## C. Full four-GPU baseline

| Field | Value |
|---|---|
| Purpose | The faithful baseline reproduction: 10 epochs, effective batch 96, cosine 1e-4→1e-5, on the full 81,632-image train split; validate on full `id_test.json` each epoch. This is the run whose metrics are compared to the paper. |
| GPUs | 4 (idle) |
| Batch size | 24 (per GPU) |
| Accumulation | 1 |
| Effective global batch | 96 |
| Epochs | 10 |
| Config | `reproduction_artifacts/configs/sica_train_candidate.yaml` (confirm `accum_iter: 1`, `epochs: 10`, `lr: 1e-4`, `min_lr: 1e-5`) |
| Command/script | `bash reproduction_artifacts/scripts/run_sica_4gpu.sh` |
| Expected outputs | `logs/sica_train/checkpoint-0.pth`, `checkpoint-9.pth`, best-epoch `checkpoint-<best>.pth`; `log.txt` with 10 epoch lines (train_lr, losses, id_test AUC/AP/Acc per epoch); TensorBoard scalars incl. `Average` and per-metric curves; logged LR should decay 1e-4→1e-5 |
| Success criteria | Exit 0; all 10 epochs complete; cosine LR trajectory confirmed in logs; id_test AUC on the final epoch is finite and recorded; checkpoints for epoch 0, 9, and best exist |
| Stop conditions | Exit 0 (success); or NCCL illegal-memory-access (GPUs not idle - wait); OOM (lower batch); loss NaN (lower lr / disable AMP temporarily); divergence (stop and diagnose) |
| Prerequisite | Experiment B passed on idle GPUs; reproducibility snapshot captured (`scripts/capture_repro_snapshot.sh`) **before** launch |
| Note | If 4 GPUs cannot be obtained, run the 1-GPU equivalent with `accum_iter: 4` (effective batch 96) - same protocol, ~4× longer. The shipped `sica_train_candidate.yaml` has `accum_iter: 1`; **set it to 4** for the 1-GPU faithful run. |

---

## D. Official training-only checkpoint evaluation

| Field | Value |
|---|---|
| Purpose | Establish a reference anchor: evaluate the **official SICA "train" checkpoint** (trained only on OpenMMSec train, per the SICA README) under the identical protocol as our baseline, to (a) validate our eval pipeline against a known-good model and (b) compare our reproduced numbers to the paper's reference. |
| GPUs | 1 (idle or shared; inference only) |
| Batch size | test_batch_size 32 |
| Accumulation | n/a (eval) |
| Effective global batch | n/a |
| Epochs | n/a (eval) |
| Config | A **proposed** copy `research_prep/configs_proposed/sica_eval_official_train.yaml` (derived from `sica_eval_domains_candidate.yaml`) pointing `checkpoint_path` at a dir holding the official "train" checkpoint renamed `checkpoint-0.pth`. **Do not modify the existing eval config.** |
| Command/script | (1) Download the official "train" checkpoint from the SICA Google Drive folder (URL in `SICA_CHECKPOINT_AUDIT.md`); (2) verify it is a `CLIP_LORA_PURE` state dict (keys contain `...attn.in_proj_weight.parametrizations...`, `...out_proj...`, `fc.weight`); (3) place as `research_prep/checkpoints/official_train/checkpoint-0.pth`; (4) launch eval via `torchrun --nproc_per_node=1 ... test.py --config <proposed yaml>` for the 4 OOD domains + id_test. **Fallback** (if the checkpoint is a bare `state_dict` without a `'model'` key, which `test.py` requires): use the standalone `tools/evaluate_sica_predictions.py` after dumping per-sample predictions with a small read-only inference loop that loads the checkpoint via `inference.py`'s tolerant loader (supports raw `state_dict`, `'model'`, or `'state_dict'`). |
| Expected outputs | Per-domain `log.txt` (deepfake_ood, aigc_ood, iml_ood, doc_ood) + id_test; macro-domain average computed by the standalone evaluator |
| Success criteria | Checkpoint loads without shape mismatch; per-domain AUC/AP/Acc/F1 finite; macro-domain average computed; results recorded as the reference anchor |
| Stop conditions | Shape mismatch (wrong checkpoint / wrong model class - abort and re-verify); missing Google Drive access (blocker - request access); OOM (lower test_batch_size) |
| Prerequisite | Official "train" checkpoint downloaded & SHA-256 recorded. **Never use the "full" checkpoint** (trained incl. test/val) for fair baseline/OOD numbers. |
| Status | BLOCKED on checkpoint download. No local SICA checkpoint exists (`SICA_CHECKPOINT_AUDIT.md`). |

---

## E. Three-seed reproducibility run

| Field | Value |
|---|---|
| Purpose | Quantify seed variance of the headline metric (macro-domain AUC) to report reproduction robustly (mean ± std) and to confirm the run is not a lucky-seed fluke. |
| GPUs | 4 (idle) per run × 3 runs (sequential, or parallel on different GPU sets if available) |
| Batch size | 24 |
| Accumulation | 1 |
| Effective global batch | 96 |
| Epochs | 10 |
| Config | `sica_train_candidate.yaml` with `seed:` set to **42, 7, 123** for the three runs respectively (three **proposed** copies `sica_train_seed42/7/123.yaml`, or override via a small wrapper; **do not overwrite the candidate config**). `train.py` seeds via `misc.seed_torch(args.seed + rank)`. |
| Command/script | For each seed: `bash run_sica_4gpu.sh` with the seed-specific config (each writing to its own `logs/sica_train_seedN/`); then eval each checkpoint set on the 4 OOD domains. |
| Expected outputs | 3 independent `logs/sica_train_seedN/` dirs, each with checkpoints + `log.txt`; 3 sets of per-domain metrics; macro AUC mean ± std |
| Success criteria | All 3 runs complete 10 epochs; macro-domain AUC std is small (target: a few × 0.1 pp; if large, report and investigate determinism); no run diverges |
| Stop conditions | Exit 0 ×3 (success); if one run diverges, keep the other two and flag; OOM/NCCL as in C |
| Prerequisite | Experiment C completed (the protocol is validated); idle GPUs |

---

## F. Reduced-data diagnostic run

| Field | Value |
|---|---|
| Purpose | Diagnose data-scaling / convergence behavior and check for overfitting or domain-imbalance effects by training on a reduced subset of `train.json` (e.g., 10% / 20% stratified by domain and label). Helps interpret baseline results and motivates improvement research (e.g., domain-balanced sampling). |
| GPUs | 1 or 4 |
| Batch size | 24 |
| Accumulation | 1 (4-GPU) or 4 (1-GPU) |
| Effective global batch | 96 |
| Epochs | 10 (same as baseline, to isolate the data effect) |
| Config | **Proposed** `sica_train_subset10.yaml` / `sica_train_subset20.yaml` derived from the candidate, pointing `train_dataset.init_config.path` at a stratified subset manifest written under `reproduction_artifacts/data/` (derived from `train.json`; public data **not** modified). Keep `id_test.json` and the OOD manifests unchanged. |
| Command/script | `run_sica_1gpu.sh` (with `accum_iter:4`) or `run_sica_4gpu.sh` using the subset config. |
| Expected outputs | `logs/sica_train_subset10/` (etc.) with checkpoints + `log.txt`; OOD-domain eval for each subset size |
| Success criteria | Completes 10 epochs; metrics recorded; the curve of macro-AUC vs. fraction-of-data is monotonic-ish (sanity); a sharp gap between train and OOD metrics flags overfitting / domain imbalance |
| Stop conditions | Exit 0; divergence; OOM; subset manifest creation must be stratified (preserve per-domain real/fake ratios) to avoid confounding |
| Prerequisite | A stratified subset-creation step (CPU-only, JSON-only; can reuse the prefilter cache). Does **not** require idle GPUs if run single-GPU. |

---

## Execution order (recommended)

1. **A** (done) → 2. **B** (when 4 GPUs idle) → 3. **C** (immediately after B passes, same idle window) → 4. **D** (can start now if the checkpoint is downloaded; CPU/single-GPU) → 5. eval C's checkpoints on the 4 OOD domains + macro → 6. **E** (3 seeds, idle GPUs) → 7. **F** (diagnostic, flexible scheduling).

Do **not** claim reproduction until C completes at effective batch 96 **and** the OOD macro-domain evaluation (items in `EVALUATION_PROTOCOL.md`) is done. D provides the reference anchor; E provides the error bars; F motivates the research directions.
