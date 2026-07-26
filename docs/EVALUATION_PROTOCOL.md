# SICA Evaluation Protocol

This document defines exactly how SICA checkpoints are evaluated and how the headline numbers are computed and reported. It is the single source of truth for "what counts as the result." It does not launch any evaluation; it specifies the protocol.

All manifests live in `/mnt/nas/public/public_datasets/OpenMMSecV2/jsons_v4`. Counts below are from JSON parsing (no image scan).

---

## 1. Manifests

| Manifest | n | Role | Domains |
|---|---:|---|---|
| `train.json` | 81,632 | training only (never an evaluation target) | IMDL 30,585 / deepfake 25,484 / Doc 12,962 / AIGC 12,601 |
| `id_test.json` | 8,240 | **in-distribution validation** (used during training to pick the best epoch; also reported as the ID number) | IMDL 3,415 / deepfake 2,016 / Doc 1,410 / AIGC 1,399 |
| `deepfake_ood.json` | 67,136 | **OOD domain 1** | deepfake |
| `aigc_ood.json` | 77,048 | **OOD domain 2** | AIGC |
| `iml_ood.json` | 64,914 | **OOD domain 3** | IMDL |
| `doc_ood.json` | 34,613 | **OOD domain 4** | Doc |
| `total_ood_test.json` | 243,711 | union of the four OOD domains (sums exactly) | all four |

The four OOD domains are the **generalization** test. `total_ood_test.json` is their union - convenient for a single inference pass, but its aggregate metric is **sample-weighted**, not the paper's headline (see §6).

### Label distribution (real=0 / fake=1)

| Manifest | real (0) | fake (1) | % fake |
|---|---:|---:|---:|
| train.json | 34,736 | 46,896 | 57% |
| id_test.json | 3,864 | 4,376 | 53% |
| deepfake_ood.json | 19,000 | 48,136 | 72% |
| aigc_ood.json | 36,000 | 41,048 | 53% |
| iml_ood.json | 30,914 | 34,000 | 52% |
| doc_ood.json | 4,788 | 29,825 | 86% |
| total_ood_test.json | 90,702 | 153,009 | 63% |

The domains differ sharply in size and in real/fake balance. This is why aggregation method matters (§5–6).

---

## 2. Label semantics

- **`label` field**: `0` = real (authentic / non-manipulated); `1` = fake (manipulated / AI-generated / forged). Confirmed from the manifests and `CLIP_LORA_PURE.forward`.
- **Model output**: `logit = self.fc(self.model.encode_image(x))` (scalar per image); `pred_label = torch.sigmoid(logit)` is the **probability of fake** (class 1).
- Evaluators receive `pred_label` (probabilities), **not** raw logits.

## 3. Prediction threshold assumptions

- **Accuracy (ACC) and F1** are computed at a fixed decision threshold **τ = 0.5** on `pred_label`: predict fake iff `pred_label ≥ 0.5`.
- **AUC and AP are threshold-free** (they integrate over all thresholds), so they are unaffected by τ.
- The standalone evaluator (`reproduction_artifacts/tools/evaluate_sica_predictions.py`) defaults to τ = 0.5 and is configurable.
- **Caveat**: because the OOD domains are label-imbalanced (e.g., Doc is 86% fake), ACC and F1 at τ = 0.5 are sensitive to the balance; AUC and AP are more robust cross-domain comparisons. Report all four, but weight the comparison toward AUC/AP for OOD generalization.

## 4. Metric definitions

| Metric | Definition | Threshold | Computed by |
|---|---|---|---|
| **ACC** | (TP + TN) / N at τ | 0.5 | `ImageAccuracy` (IMDLBenCo) / standalone |
| **AUC** | area under the ROC curve | none | `ImageAUC` (IMDLBenCo) / standalone |
| **AP** | average precision = area under the precision-recall curve | none | `ImageAP` (ForensicHub) / standalone |
| **F1** | 2·P·R / (P + R) at τ (0 if P+R=0) | 0.5 | **not in current evaluator list** - add `ImageF1` or compute via standalone |

**Important**: the shipped evaluator list is `ImageAUC`, `ImageAP`, `ImageAccuracy` - **F1 is missing**. For the full paper metric suite, either add `ImageF1` to the config or compute F1 from saved per-sample predictions with the standalone evaluator. Until F1 is added, it cannot be reported.

### Single-class safety
If a domain (or a sub-group) contains only one class, AUC/AP are undefined. The standalone evaluator returns `None` with a warning rather than crashing. Do not silently drop such groups.

---

## 5. Per-domain reporting

For OOD generalization, evaluate **each of the four domains separately** and report a per-domain table:

| Domain | n | ACC | AUC | AP | F1 |
|---|---:|---|---|---|---|
| deepfake_ood | 67,136 | … | … | … | … |
| aigc_ood | 77,048 | … | … | … | … |
| iml_ood | 64,914 | … | … | … | … |
| doc_ood | 34,613 | … | … | … | … |

Run inference per domain (or once over `total_ood_test.json` and split predictions by the per-sample `domain` field), compute each metric within the domain, then aggregate (§6).

### Recommended evaluation path
1. Run `test.py` (via `torchrun --nproc_per_node=1`) with `sica_eval_domains_candidate.yaml` - it evaluates each of the four OOD manifests + `total_ood_test` and writes per-dataset `log.txt` under `log_dir/<dataset_name>/`. (`test.py` requires DDP because it uses `model.module`; launch via `torchrun` even for 1 GPU.)
2. **Additionally**, dump per-sample `(path, label, score, domain)` predictions and run `tools/evaluate_sica_predictions.py` to get the macro-domain average and per-`mani_type`/`sub_mani_type` breakdown, and to avoid any `drop_last=True` tail effect (§7).

## 6. Macro-domain average (the paper's headline)

The SICA paper reports the **overall** number as the **macro average across the four domains** - i.e., the unweighted mean of the four per-domain metric values:

```
macro_ACC = (ACC_deepfake + ACC_aigc + ACC_iml + ACC_doc) / 4
macro_AUC = (AUC_deepfake + AUC_aigc + AUC_iml + AUC_doc) / 4
macro_AP  = ( AP_deepfake +  AP_aigc +  AP_iml +  AP_doc) / 4
macro_F1  = ( F1_deepfake +  F1_aigc +  F1_iml +  F1_doc) / 4
```

### Why sample-weighted overall accuracy is NOT the paper's reported overall

Running a single metric over `total_ood_test.json` yields a **sample-weighted** number: each sample contributes equally, so larger domains dominate. With the domain sizes here:

```
AIGC    77,048  (31.6%)   ← dominates the sample-weighted average
deepfake 67,136 (27.5%)   ← dominates
IMDL    64,914  (26.6%)
Doc     34,613  (14.2%)   ← nearly silenced
```

A sample-weighted ACC/AUC/AP therefore ≈ `(0.316·AIGC + 0.275·deepfake + 0.266·IMDL + 0.142·Doc)`, **not** the unweighted `0.25·each`. The two can differ by several points when domain metrics differ (and they will: Doc is 86% fake and behaves differently from the ~52%-fake AIGC/IML). Additionally, because label balance differs per domain (§1), a sample-weighted ACC conflates "the model is accurate on big, balanced domains" with "the model generalizes across domains" - the paper's macro average deliberately weights each domain equally regardless of size.

**Rule:** always report the **macro-domain average** as the headline. Optionally also report the sample-weighted `total_ood_test` number, clearly labeled "sample-weighted (not the paper's headline)."

### Training-time "best" metric is also not the headline
During training, `train.py` selects the best checkpoint by the **mean of metric means across the configured test datasets** (id_test only, by default). That scalar is a checkpoint-selection heuristic, **not** the paper's OOD macro average. Do not report it as the result.

---

## 7. Evaluating the official "train" checkpoint fairly

The SICA authors release two checkpoints (per the SICA README, audited in `SICA_CHECKPOINT_AUDIT.md`):

- **"...train" checkpoint**: trained **only on the OpenMMSec training set**. This is the **fair** baseline/reference checkpoint.
- **"...full" checkpoint**: trained on the **entire OpenMMSec dataset, including the test and validation sets**. This is **not fair** for OOD/ID evaluation (see §8).

### Steps to evaluate the official "train" checkpoint

1. **Download** the official "...train" checkpoint from the SICA Google Drive folder (URL in `SICA_CHECKPOINT_AUDIT.md`). Record its SHA-256.
2. **Verify it is a `CLIP_LORA_PURE` state dict**: keys must include `model.visual.transformer.resblocks.*.attn.in_proj_weight.parametrizations.0.A/B`, `...out_proj...parametrizations...`, and `fc.weight`/`fc.bias`. (The local `univfd_train/checkpoint-9.pth` is **not** a SICA checkpoint - it has no LoRA/visual keys.)
3. **Load it tolerantly**: `inference.py`'s loader supports a bare `state_dict`, or one nested under `'model'`, or under `'state_dict'`. `test.py` only supports the `'model'`-nested form (`model.module.load_state_dict(ckpt['model'])`); if the official file is a bare `state_dict`, use the standalone evaluator with a small read-only inference loop instead.
4. **Evaluate under the identical protocol** as our reproduced checkpoint: the same four OOD domains + `id_test`, the same metrics (ACC/AUC/AP/F1), the same macro-domain averaging.
5. **Report as the reference anchor** alongside our reproduced numbers, so readers can see our pipeline matches a known-good model and how our reproduction compares to the paper's reference.

### Fairness conditions
- Same preprocessing, same evaluators, same threshold (0.5), same manifests.
- Do **not** fine-tune or adapt the official checkpoint before evaluating.
- Do **not** cherry-pick the threshold per domain.

---

## 8. Why the "full" checkpoint must NOT be used for fair baseline evaluation

The "...full" checkpoint was trained on the **entire OpenMMSec dataset, including the test and validation sets** (`id_test.json` and the OOD manifests). Therefore:

- Its ID-validation numbers are **contaminated** (it has seen `id_test.json` during training).
- Its OOD numbers are **contaminated** (it has seen the four OOD manifests during training).
- Reporting them as "baseline" or "reproduced" results would constitute **test-set leakage** and is not a valid evaluation of generalization.

**Permitted use (reference only):** the "full" checkpoint may be reported **only** as an explicit upper-bound / "trained-on-everything" reference, clearly labeled as leaky, never as the fair baseline or the reproduced result. It must never be the number compared to the paper's generalization tables.

**Rule:** for any fair baseline, OOD-generalization, or reproduction claim, use **only** (a) our own checkpoint trained solely on `train.json`, or (b) the official "...train" checkpoint. Never the "...full" checkpoint.

---

## 9. Aggregation checklist (for every reported number)

- [ ] Metric computed **per domain** (deepfake/aigc/iml/doc), not only over `total_ood_test`.
- [ ] **Macro-domain average** = unweighted mean of the 4 domain values, reported as the headline.
- [ ] All four metrics reported: ACC, AUC, AP, **F1** (add `ImageF1` if missing).
- [ ] Threshold τ = 0.5 stated for ACC/F1; AUC/AP noted as threshold-free.
- [ ] Checkpoint trained **only on `train.json`** (or the official "train" checkpoint) - never "full".
- [ ] `drop_last=True` tail effect excluded (recompute from saved per-sample predictions).
- [ ] Reproducibility snapshot attached (git/diff/pip-freeze/checksums/command).
- [ ] For the headline, mean ± std over ≥3 seeds (Task E).
