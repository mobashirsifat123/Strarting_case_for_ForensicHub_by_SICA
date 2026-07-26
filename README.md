# Starting Case for ForensicHub: SICA Baseline Reproduction

A beginner-friendly and auditable starting case for reproducing **SICA (Semantic-Induced Constrained Adaptation)** on **OpenMMSecV2** using the **ForensicHub** training framework.

This repository organizes the configuration, evaluation scripts, data-integrity procedures, and documentation used in a completed SICA baseline reproduction. It is intended to help new researchers understand and reproduce the workflow before attempting architectural improvements.

> **Status:** Close numerical reproduction completed  
> **Maintainer:** [Mobashir Sifat](https://github.com/mobashirsifat123)  
> **Affiliation:** Software Engineering, Sichuan University  
> **Research group:** Research Group of Prof. Ji-Zhe Zhou

---

## Overview

SICA addresses unified fake-image detection across four heterogeneous forensic domains:

- Deepfake detection
- AI-generated image detection
- Image manipulation detection
- Document forgery detection

The method uses a frozen **CLIP ViT-L/14** visual backbone with low-rank adaptation in the visual self-attention blocks. The goal is to learn forensic artifacts while preserving the semantic structure of the pretrained feature space.

This starter case provides:

- Portable single-GPU training configuration
- Four-GPU smoke-test configuration
- Full ID/OOD evaluation scripts
- Checkpoint verification utilities
- Paper-compatible metric aggregation
- Corrupted-image handling documentation
- An end-to-end reproduction report
- Reproduced baseline results

---

## Reproduction Result

A 10-epoch SICA baseline was trained and evaluated using the released OpenMMSecV2 manifests.

### Paper-style macro-domain comparison

| Metric | Reproduced | Paper | Difference |
|---|---:|---:|---:|
| ACC | **84.34%** | 85.40% | -1.06 pp |
| AUC | **86.65%** | 87.50% | -0.85 pp |
| AP | **79.19%** | 79.00% | +0.19 pp |
| F1 | **69.36%** | 69.70% | -0.34 pp |

**Verdict:** Close numerical reproduction, not exact reproduction.

### Paper-style per-domain accuracy

| Domain | Reproduced | Paper | Difference |
|---|---:|---:|---:|
| Deepfake | 88.46% | 88.40% | +0.06 pp |
| AIGC | 94.46% | 94.00% | +0.46 pp |
| IMDL | 82.74% | 85.30% | -2.56 pp |
| Document | 71.72% | 73.80% | -2.08 pp |

The largest remaining domain-level differences are concentrated in IMDL and Document.

### Direct split-level evaluation

These results use direct per-image evaluation and are reported separately from the paper-style macro-over-types protocol.

| Split | Evaluated | Skipped | ACC | AUC | AP | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ID test | 8,188 | 52 | 97.07% | 99.60% | 99.67% | 97.25% |
| Deepfake OOD | 67,129 | 7 | 83.40% | 92.13% | 97.06% | 87.24% |
| AIGC OOD | 75,774 | 1,274 | 88.77% | 95.22% | 96.71% | 88.76% |
| IMDL OOD | 64,914 | 0 | 75.78% | 86.76% | 89.98% | 71.14% |
| Document OOD | 34,613 | 0 | 73.39% | 85.70% | 97.36% | 82.44% |
| Total OOD micro | 242,430 | 1,281 | 81.61% | 92.19% | 95.88% | 83.49% |

Micro metrics and paper-style macro metrics use different sample groupings and should not be compared directly.

---

## Repository Structure

```text
.
├── configs/
│   ├── sica_train_1gpu.yaml
│   └── sica_4gpu_smoke.yaml
├── scripts/
│   ├── aggregate_sica_metrics.py
│   ├── dump_sica_predictions.py
│   ├── run_full_eval.sh
│   └── verify_checkpoint.py
├── docs/
│   ├── BASELINE_REPRODUCTION_PROTOCOL.md
│   ├── CORRUPTED_IMAGE_POLICY.md
│   ├── END_TO_END_REPRODUCTION_REPORT.md
│   └── EVALUATION_PROTOCOL.md
├── data/
├── outputs/
├── assets/
└── examples/
```

### Important files

| File | Purpose |
|---|---|
| `configs/sica_train_1gpu.yaml` | Single-GPU SICA training with gradient accumulation |
| `configs/sica_4gpu_smoke.yaml` | Small distributed pipeline smoke-test configuration |
| `scripts/dump_sica_predictions.py` | Generate image-level prediction CSV files |
| `scripts/aggregate_sica_metrics.py` | Aggregate ACC, AUC, AP, and F1 |
| `scripts/run_full_eval.sh` | Run full ID and four-domain OOD evaluation |
| `scripts/verify_checkpoint.py` | Verify checkpoint structure and checksum |
| `docs/BASELINE_REPRODUCTION_PROTOCOL.md` | Baseline reproduction workflow |
| `docs/EVALUATION_PROTOCOL.md` | Evaluation and metric definitions |
| `docs/CORRUPTED_IMAGE_POLICY.md` | Unreadable-image handling policy |
| `docs/END_TO_END_REPRODUCTION_REPORT.md` | Detailed reproduction record |

---

## What Is Not Included

This repository is a companion starter case rather than a standalone copy of ForensicHub.

It does not redistribute:

- The ForensicHub source repository
- OpenMMSec or OpenMMSecV2 images
- CLIP pretrained weights
- The trained SICA checkpoint
- Raw prediction files
- Private laboratory paths or credentials

Obtain these components from their official sources or your authorized laboratory storage.

---

## Prerequisites

### Required repositories

Clone ForensicHub and this starter repository into the same parent directory:

```bash
git clone https://github.com/scu-zjz/ForensicHub.git

git clone \
https://github.com/mobashirsifat123/Strarting_case_for_ForensicHub_by_SICA.git
```

Example layout:

```text
workspace/
├── ForensicHub/
└── Strarting_case_for_ForensicHub_by_SICA/
```

Define convenient paths:

```bash
export FORENSICHUB=/absolute/path/to/ForensicHub
export SICA_STARTER=/absolute/path/to/Strarting_case_for_ForensicHub_by_SICA
```

### Reference environment

The completed reproduction used:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| TorchVision | 0.22.1+cu128 |
| Backbone | CLIP ViT-L/14 |
| Input size | 224 × 224 |

These versions document the validated environment; they are not guaranteed to be the only compatible versions.

Follow the official ForensicHub installation instructions to prepare the remaining dependencies.

---

## Dataset Preparation

This starter case expects the released OpenMMSecV2 manifests.

Set the dataset root:

```bash
export OPENMMSEC_ROOT=/absolute/path/to/OpenMMSecV2
```

Create the local manifest links:

```bash
cd "$SICA_STARTER"

mkdir -p data

ln -s "$OPENMMSEC_ROOT/jsons_v4/train.json" \
      data/train.json

ln -s "$OPENMMSEC_ROOT/jsons_v4/id_test.json" \
      data/id_test.json
```

The repository configs use:

```text
./data/train.json
./data/id_test.json
```

Do not copy or modify public datasets unnecessarily. Symbolic links are preferred.

### Manifest path remapping

Some released manifests may contain absolute image paths that differ from the local dataset location.

The configs provide:

```yaml
path_prefix_from: ""
path_prefix_to: ""
```

Leave both values empty when the paths already resolve.

When remapping is required, set both values. For example:

```yaml
path_prefix_from: "/original/manifest/prefix/"
path_prefix_to: "/local/dataset/prefix/"
```

Only the leading prefix is replaced. The manifest file itself is not modified.

---

## Training

The reproduced single-GPU configuration uses:

| Setting | Value |
|---|---|
| Backbone | CLIP ViT-L/14 |
| Backbone parameters | Frozen |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA insertion | Attention QKV and output projection |
| Epochs | 10 |
| Per-step batch size | 24 |
| Gradient accumulation | 4 |
| Effective global batch | 96 |
| Optimizer | AdamW |
| Learning rate | Cosine, `1e-4` to `1e-5` |
| Loss | BCEWithLogitsLoss on the raw classifier logit |
| Seed | 42 |
| Input size | 224 × 224 |

From the starter repository root:

```bash
cd "$SICA_STARTER"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$FORENSICHUB" \
torchrun --nproc_per_node=1 \
"$FORENSICHUB/ForensicHub/training_scripts/train.py" \
--config configs/sica_train_1gpu.yaml
```

Checkpoints and logs are written under:

```text
outputs/sica_train_1gpu/
```

Use an available GPU and follow shared-server usage policies.

---

## Full Evaluation

The evaluation script requires:

- A trained SICA checkpoint
- The OpenMMSecV2 root
- The ForensicHub repository
- A Python executable with the required environment

Example:

```bash
cd "$SICA_STARTER"

OPENMMSEC_ROOT="$OPENMMSEC_ROOT" \
SICA_CHECKPOINT=/absolute/path/to/checkpoint-9.pth \
FORENSICHUB="$FORENSICHUB" \
PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_full_eval.sh
```

When manifest image paths require remapping:

```bash
OPENMMSEC_ROOT="$OPENMMSEC_ROOT" \
SICA_CHECKPOINT=/absolute/path/to/checkpoint-9.pth \
FORENSICHUB="$FORENSICHUB" \
PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0 \
PATH_PREFIX_FROM=/original/manifest/prefix/ \
PATH_PREFIX_TO=/local/dataset/prefix/ \
bash scripts/run_full_eval.sh
```

The script evaluates:

```text
id_test
deepfake_ood
aigc_ood
iml_ood
doc_ood
```

The total OOD result is derived from the verified union of the four pairwise-disjoint domain sets.

---

## Checkpoint Verification

Inspect a trained checkpoint without modifying it:

```bash
cd "$SICA_STARTER"

python3 scripts/verify_checkpoint.py \
  --checkpoint /absolute/path/to/checkpoint-9.pth \
  --out-json outputs/checkpoint_verification.json \
  --out-md outputs/checkpoint_verification.md \
  --repo "$(dirname "$FORENSICHUB")"
```

The completed reproduction checkpoint contained:

- Model state
- LoRA parameters
- Classification head
- Optimizer state
- AMP scaler
- Training arguments
- Final epoch information

The verified model contained:

| Parameter group | Count |
|---|---:|
| Total parameters | 428,796,930 |
| Trainable parameters | 1,180,417 |
| Frozen parameters | 427,616,513 |

---

## Evaluation Protocol

The evaluation pipeline uses:

1. RGB image conversion
2. PIL resize to 224 × 224
3. CLIP normalization
4. Raw binary classifier logit
5. Sigmoid only for probability output
6. Threshold 0.5 for ACC and F1
7. Continuous probabilities for AUC and AP

Metrics:

- **ACC:** accuracy at threshold 0.5
- **F1:** binary F1 at threshold 0.5
- **AUC:** `roc_auc_score`
- **AP:** `average_precision_score`

The paper-style aggregation protocol used in this reproduction is documented in:

```text
docs/EVALUATION_PROTOCOL.md
```

It was inferred from the released paper results and should be treated as a plausible reconstruction rather than a confirmed unpublished implementation detail.

---

## Data Integrity

The reproduction followed these rules:

- Public data were treated as read-only.
- No public image or manifest was repaired, overwritten, deleted, or replaced.
- Unreadable images were excluded deterministically.
- Every excluded image was recorded.
- Metrics use the effective evaluated denominator.
- Evaluation does not substitute corrupted files with unrelated images.

Effective counts:

| Manifest | Nominal | Evaluated / usable | Unreadable |
|---|---:|---:|---:|
| Train | 81,632 | 81,221 | 411 |
| ID test | 8,240 | 8,188 | 52 |
| Deepfake OOD | 67,136 | 67,129 | 7 |
| AIGC OOD | 77,048 | 75,774 | 1,274 |
| IMDL OOD | 64,914 | 64,914 | 0 |
| Document OOD | 34,613 | 34,613 | 0 |
| Total OOD | 243,711 | 242,430 | 1,281 |

---

## Limitations

This reproduction has several disclosed limitations:

- Only one random seed was evaluated.
- Run-to-run variance was not measured.
- Native four-GPU DDP parity was not established in an uncontended environment.
- Unreadable files changed the effective sample denominators.
- Some implementation details are not explicitly specified in the paper.
- The paper-style per-type aggregation protocol was inferred from the released results.
- The reproduction used one GPU with gradient accumulation rather than native four-GPU training.
- Residual performance gaps remain in IMDL Generative and Document Non-AIGC.

These limitations should be retained in reports, presentations, and future manuscripts.

---

## Validation

The completed reproduction package was checked for:

- Python syntax
- Shell-script syntax
- JSON validity
- CSV schema and row counts
- Checkpoint compatibility
- Artifact checksums
- Prediction probability ranges
- NaN and Inf values
- Duplicate image paths
- Public-data immutability
- Secret and credential leakage
- Separation of macro and micro metrics

The reproduction is suitable as a frozen baseline for controlled improvement experiments.

---

## Recommended Research Direction

The next stage should begin with failure analysis rather than immediately adding a large architecture.

The primary under-performing groups are:

- IMDL Generative
- Document Non-AIGC

A new method should be evaluated against the frozen baseline using controlled ablations, multiple seeds, consistent data splits, and unchanged metric definitions.

---

## Upstream Projects

- **SICA and OpenMMSec:**  
  https://github.com/scu-zjz/SICA_OpenMMSec

- **ForensicHub:**  
  https://github.com/scu-zjz/ForensicHub

- **SICA paper:**  
  https://arxiv.org/abs/2602.06676

This repository does not replace or claim ownership of the upstream projects.

---

## Citation

When using this starter case, please cite the original SICA and ForensicHub works.

```bibtex
@article{du2026can,
  title={Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction},
  author={Du, Bo and Ma, Xiaochen and Zhu, Xuekang and Yang, Zhe and Niu, Chaoqun and Qu, Chenfan and Fang, Mingqi and Wang, Zhenming and Liu, Jingjing and Liu, Jian and Zhou, Ji-Zhe},
  journal={arXiv preprint arXiv:2602.06676},
  year={2026}
}
```

Please also use the official ForensicHub citation provided by its upstream repository.

---

## Maintainer

**Mobashir Sifat**  
Software Engineering, Sichuan University  
Research Group of Prof. Ji-Zhe Zhou  
GitHub: [@mobashirsifat123](https://github.com/mobashirsifat123)

---

## Acknowledgments

This starter case was prepared from a completed SICA/OpenMMSecV2 baseline reproduction conducted in the Research Group of Prof. Ji-Zhe Zhou at Sichuan University.

The original SICA, OpenMMSec, and ForensicHub contributions belong to their respective authors and maintainers.# Starting Case for ForensicHub: SICA Baseline Reproduction

A beginner-friendly and auditable starting case for reproducing **SICA (Semantic-Induced Constrained Adaptation)** on **OpenMMSecV2** using the **ForensicHub** training framework.

This repository organizes the configuration, evaluation scripts, data-integrity procedures, and documentation used in a completed SICA baseline reproduction. It is intended to help new researchers understand and reproduce the workflow before attempting architectural improvements.

> **Status:** Close numerical reproduction completed  
> **Maintainer:** [Mobashir Sifat](https://github.com/mobashirsifat123)  
> **Affiliation:** Software Engineering, Sichuan University  
> **Research group:** Research Group of Prof. Ji-Zhe Zhou

---

## Overview

SICA addresses unified fake-image detection across four heterogeneous forensic domains:

- Deepfake detection
- AI-generated image detection
- Image manipulation detection
- Document forgery detection

The method uses a frozen **CLIP ViT-L/14** visual backbone with low-rank adaptation in the visual self-attention blocks. The goal is to learn forensic artifacts while preserving the semantic structure of the pretrained feature space.

This starter case provides:

- Portable single-GPU training configuration
- Four-GPU smoke-test configuration
- Full ID/OOD evaluation scripts
- Checkpoint verification utilities
- Paper-compatible metric aggregation
- Corrupted-image handling documentation
- An end-to-end reproduction report
- Reproduced baseline results

---

## Reproduction Result

A 10-epoch SICA baseline was trained and evaluated using the released OpenMMSecV2 manifests.

### Paper-style macro-domain comparison

| Metric | Reproduced | Paper | Difference |
|---|---:|---:|---:|
| ACC | **84.34%** | 85.40% | -1.06 pp |
| AUC | **86.65%** | 87.50% | -0.85 pp |
| AP | **79.19%** | 79.00% | +0.19 pp |
| F1 | **69.36%** | 69.70% | -0.34 pp |

**Verdict:** Close numerical reproduction, not exact reproduction.

### Paper-style per-domain accuracy

| Domain | Reproduced | Paper | Difference |
|---|---:|---:|---:|
| Deepfake | 88.46% | 88.40% | +0.06 pp |
| AIGC | 94.46% | 94.00% | +0.46 pp |
| IMDL | 82.74% | 85.30% | -2.56 pp |
| Document | 71.72% | 73.80% | -2.08 pp |

The largest remaining domain-level differences are concentrated in IMDL and Document.

### Direct split-level evaluation

These results use direct per-image evaluation and are reported separately from the paper-style macro-over-types protocol.

| Split | Evaluated | Skipped | ACC | AUC | AP | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ID test | 8,188 | 52 | 97.07% | 99.60% | 99.67% | 97.25% |
| Deepfake OOD | 67,129 | 7 | 83.40% | 92.13% | 97.06% | 87.24% |
| AIGC OOD | 75,774 | 1,274 | 88.77% | 95.22% | 96.71% | 88.76% |
| IMDL OOD | 64,914 | 0 | 75.78% | 86.76% | 89.98% | 71.14% |
| Document OOD | 34,613 | 0 | 73.39% | 85.70% | 97.36% | 82.44% |
| Total OOD micro | 242,430 | 1,281 | 81.61% | 92.19% | 95.88% | 83.49% |

Micro metrics and paper-style macro metrics use different sample groupings and should not be compared directly.

---

## Repository Structure

```text
.
├── configs/
│   ├── sica_train_1gpu.yaml
│   └── sica_4gpu_smoke.yaml
├── scripts/
│   ├── aggregate_sica_metrics.py
│   ├── dump_sica_predictions.py
│   ├── run_full_eval.sh
│   └── verify_checkpoint.py
├── docs/
│   ├── BASELINE_REPRODUCTION_PROTOCOL.md
│   ├── CORRUPTED_IMAGE_POLICY.md
│   ├── END_TO_END_REPRODUCTION_REPORT.md
│   └── EVALUATION_PROTOCOL.md
├── data/
├── outputs/
├── assets/
└── examples/
```

### Important files

| File | Purpose |
|---|---|
| `configs/sica_train_1gpu.yaml` | Single-GPU SICA training with gradient accumulation |
| `configs/sica_4gpu_smoke.yaml` | Small distributed pipeline smoke-test configuration |
| `scripts/dump_sica_predictions.py` | Generate image-level prediction CSV files |
| `scripts/aggregate_sica_metrics.py` | Aggregate ACC, AUC, AP, and F1 |
| `scripts/run_full_eval.sh` | Run full ID and four-domain OOD evaluation |
| `scripts/verify_checkpoint.py` | Verify checkpoint structure and checksum |
| `docs/BASELINE_REPRODUCTION_PROTOCOL.md` | Baseline reproduction workflow |
| `docs/EVALUATION_PROTOCOL.md` | Evaluation and metric definitions |
| `docs/CORRUPTED_IMAGE_POLICY.md` | Unreadable-image handling policy |
| `docs/END_TO_END_REPRODUCTION_REPORT.md` | Detailed reproduction record |

---

## What Is Not Included

This repository is a companion starter case rather than a standalone copy of ForensicHub.

It does not redistribute:

- The ForensicHub source repository
- OpenMMSec or OpenMMSecV2 images
- CLIP pretrained weights
- The trained SICA checkpoint
- Raw prediction files
- Private laboratory paths or credentials

Obtain these components from their official sources or your authorized laboratory storage.

---

## Prerequisites

### Required repositories

Clone ForensicHub and this starter repository into the same parent directory:

```bash
git clone https://github.com/scu-zjz/ForensicHub.git

git clone \
https://github.com/mobashirsifat123/Strarting_case_for_ForensicHub_by_SICA.git
```

Example layout:

```text
workspace/
├── ForensicHub/
└── Strarting_case_for_ForensicHub_by_SICA/
```

Define convenient paths:

```bash
export FORENSICHUB=/absolute/path/to/ForensicHub
export SICA_STARTER=/absolute/path/to/Strarting_case_for_ForensicHub_by_SICA
```

### Reference environment

The completed reproduction used:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| TorchVision | 0.22.1+cu128 |
| Backbone | CLIP ViT-L/14 |
| Input size | 224 × 224 |

These versions document the validated environment; they are not guaranteed to be the only compatible versions.

Follow the official ForensicHub installation instructions to prepare the remaining dependencies.

---

## Dataset Preparation

This starter case expects the released OpenMMSecV2 manifests.

Set the dataset root:

```bash
export OPENMMSEC_ROOT=/absolute/path/to/OpenMMSecV2
```

Create the local manifest links:

```bash
cd "$SICA_STARTER"

mkdir -p data

ln -s "$OPENMMSEC_ROOT/jsons_v4/train.json" \
      data/train.json

ln -s "$OPENMMSEC_ROOT/jsons_v4/id_test.json" \
      data/id_test.json
```

The repository configs use:

```text
./data/train.json
./data/id_test.json
```

Do not copy or modify public datasets unnecessarily. Symbolic links are preferred.

### Manifest path remapping

Some released manifests may contain absolute image paths that differ from the local dataset location.

The configs provide:

```yaml
path_prefix_from: ""
path_prefix_to: ""
```

Leave both values empty when the paths already resolve.

When remapping is required, set both values. For example:

```yaml
path_prefix_from: "/original/manifest/prefix/"
path_prefix_to: "/local/dataset/prefix/"
```

Only the leading prefix is replaced. The manifest file itself is not modified.

---

## Training

The reproduced single-GPU configuration uses:

| Setting | Value |
|---|---|
| Backbone | CLIP ViT-L/14 |
| Backbone parameters | Frozen |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA insertion | Attention QKV and output projection |
| Epochs | 10 |
| Per-step batch size | 24 |
| Gradient accumulation | 4 |
| Effective global batch | 96 |
| Optimizer | AdamW |
| Learning rate | Cosine, `1e-4` to `1e-5` |
| Loss | BCEWithLogitsLoss on the raw classifier logit |
| Seed | 42 |
| Input size | 224 × 224 |

From the starter repository root:

```bash
cd "$SICA_STARTER"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$FORENSICHUB" \
torchrun --nproc_per_node=1 \
"$FORENSICHUB/ForensicHub/training_scripts/train.py" \
--config configs/sica_train_1gpu.yaml
```

Checkpoints and logs are written under:

```text
outputs/sica_train_1gpu/
```

Use an available GPU and follow shared-server usage policies.

---

## Full Evaluation

The evaluation script requires:

- A trained SICA checkpoint
- The OpenMMSecV2 root
- The ForensicHub repository
- A Python executable with the required environment

Example:

```bash
cd "$SICA_STARTER"

OPENMMSEC_ROOT="$OPENMMSEC_ROOT" \
SICA_CHECKPOINT=/absolute/path/to/checkpoint-9.pth \
FORENSICHUB="$FORENSICHUB" \
PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_full_eval.sh
```

When manifest image paths require remapping:

```bash
OPENMMSEC_ROOT="$OPENMMSEC_ROOT" \
SICA_CHECKPOINT=/absolute/path/to/checkpoint-9.pth \
FORENSICHUB="$FORENSICHUB" \
PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0 \
PATH_PREFIX_FROM=/original/manifest/prefix/ \
PATH_PREFIX_TO=/local/dataset/prefix/ \
bash scripts/run_full_eval.sh
```

The script evaluates:

```text
id_test
deepfake_ood
aigc_ood
iml_ood
doc_ood
```

The total OOD result is derived from the verified union of the four pairwise-disjoint domain sets.

---

## Checkpoint Verification

Inspect a trained checkpoint without modifying it:

```bash
cd "$SICA_STARTER"

python3 scripts/verify_checkpoint.py \
  --checkpoint /absolute/path/to/checkpoint-9.pth \
  --out-json outputs/checkpoint_verification.json \
  --out-md outputs/checkpoint_verification.md \
  --repo "$(dirname "$FORENSICHUB")"
```

The completed reproduction checkpoint contained:

- Model state
- LoRA parameters
- Classification head
- Optimizer state
- AMP scaler
- Training arguments
- Final epoch information

The verified model contained:

| Parameter group | Count |
|---|---:|
| Total parameters | 428,796,930 |
| Trainable parameters | 1,180,417 |
| Frozen parameters | 427,616,513 |

---

## Evaluation Protocol

The evaluation pipeline uses:

1. RGB image conversion
2. PIL resize to 224 × 224
3. CLIP normalization
4. Raw binary classifier logit
5. Sigmoid only for probability output
6. Threshold 0.5 for ACC and F1
7. Continuous probabilities for AUC and AP

Metrics:

- **ACC:** accuracy at threshold 0.5
- **F1:** binary F1 at threshold 0.5
- **AUC:** `roc_auc_score`
- **AP:** `average_precision_score`

The paper-style aggregation protocol used in this reproduction is documented in:

```text
docs/EVALUATION_PROTOCOL.md
```

It was inferred from the released paper results and should be treated as a plausible reconstruction rather than a confirmed unpublished implementation detail.

---

## Data Integrity

The reproduction followed these rules:

- Public data were treated as read-only.
- No public image or manifest was repaired, overwritten, deleted, or replaced.
- Unreadable images were excluded deterministically.
- Every excluded image was recorded.
- Metrics use the effective evaluated denominator.
- Evaluation does not substitute corrupted files with unrelated images.

Effective counts:

| Manifest | Nominal | Evaluated / usable | Unreadable |
|---|---:|---:|---:|
| Train | 81,632 | 81,221 | 411 |
| ID test | 8,240 | 8,188 | 52 |
| Deepfake OOD | 67,136 | 67,129 | 7 |
| AIGC OOD | 77,048 | 75,774 | 1,274 |
| IMDL OOD | 64,914 | 64,914 | 0 |
| Document OOD | 34,613 | 34,613 | 0 |
| Total OOD | 243,711 | 242,430 | 1,281 |

---

## Limitations

This reproduction has several disclosed limitations:

- Only one random seed was evaluated.
- Run-to-run variance was not measured.
- Native four-GPU DDP parity was not established in an uncontended environment.
- Unreadable files changed the effective sample denominators.
- Some implementation details are not explicitly specified in the paper.
- The paper-style per-type aggregation protocol was inferred from the released results.
- The reproduction used one GPU with gradient accumulation rather than native four-GPU training.
- Residual performance gaps remain in IMDL Generative and Document Non-AIGC.

These limitations should be retained in reports, presentations, and future manuscripts.

---

## Validation

The completed reproduction package was checked for:

- Python syntax
- Shell-script syntax
- JSON validity
- CSV schema and row counts
- Checkpoint compatibility
- Artifact checksums
- Prediction probability ranges
- NaN and Inf values
- Duplicate image paths
- Public-data immutability
- Secret and credential leakage
- Separation of macro and micro metrics

The reproduction is suitable as a frozen baseline for controlled improvement experiments.

---

## Recommended Research Direction

The next stage should begin with failure analysis rather than immediately adding a large architecture.

The primary under-performing groups are:

- IMDL Generative
- Document Non-AIGC

A new method should be evaluated against the frozen baseline using controlled ablations, multiple seeds, consistent data splits, and unchanged metric definitions.

---

## Upstream Projects

- **SICA and OpenMMSec:**  
  https://github.com/scu-zjz/SICA_OpenMMSec

- **ForensicHub:**  
  https://github.com/scu-zjz/ForensicHub

- **SICA paper:**  
  https://arxiv.org/abs/2602.06676

This repository does not replace or claim ownership of the upstream projects.

---

## Citation

When using this starter case, please cite the original SICA and ForensicHub works.

```bibtex
@article{du2026can,
  title={Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction},
  author={Du, Bo and Ma, Xiaochen and Zhu, Xuekang and Yang, Zhe and Niu, Chaoqun and Qu, Chenfan and Fang, Mingqi and Wang, Zhenming and Liu, Jingjing and Liu, Jian and Zhou, Ji-Zhe},
  journal={arXiv preprint arXiv:2602.06676},
  year={2026}
}
```

Please also use the official ForensicHub citation provided by its upstream repository.

---

## Maintainer

**Mobashir Sifat**  
Software Engineering, Sichuan University  
Research Group of Prof. Ji-Zhe Zhou  
GitHub: [@mobashirsifat123](https://github.com/mobashirsifat123)

---

## Acknowledgments

This starter case was prepared from a completed SICA/OpenMMSecV2 baseline reproduction conducted in the Research Group of Prof. Ji-Zhe Zhou at Sichuan University.

The original SICA, OpenMMSec, and ForensicHub contributions belong to their respective authors and maintainers.
