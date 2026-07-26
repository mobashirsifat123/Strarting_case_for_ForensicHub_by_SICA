#!/usr/bin/env bash
# run_full_eval.sh - FINAL baseline evaluation of checkpoint-9.pth on all splits.
#
# Dumps per-image predictions for id_test + 4 OOD domain sets, then derives
# total_ood_test by concatenating the 4 domain CSVs (verified to be the exact
# path-union, pairwise disjoint), then aggregates ACC/AUC/AP/F1 per split.
#
# Single GPU, world_size=1 (no torchrun, no NCCL collectives) -> safe alongside
# other users' GPU jobs. Run inside tmux. ~12-15 min on a free GPU.
#
# All paths are portable. You MUST provide:
#   * SICA_CHECKPOINT : trained checkpoint file (e.g. .../sica_train_1gpu/checkpoint-9.pth)
#   * OPENMMSEC_ROOT  : OpenMMSecV2 dataset root (the dir that contains jsons_v4/)
# Optional overrides:
#   * FORENSICHUB          : path to the ForensicHub repo (default: $REPO/../ForensicHub)
#   * PYTHON               : Python executable (default: python)
#   * CUDA_VISIBLE_DEVICES : GPU index to use (default: 1)
#   * PATH_PREFIX_FROM / PATH_PREFIX_TO : remap the image-path prefix stored in
#     the manifests onto your local dataset root. Released manifests store image
#     paths under a fixed prefix (commonly /mnt/public/); if your local copy of
#     the dataset lives elsewhere, set BOTH to that prefix and your local root.
#     Leave both unset/empty if the manifest paths already resolve locally.
#
# Example:
#   OPENMMSEC_ROOT=/path/to/OpenMMSecV2 \
#   SICA_CHECKPOINT=/path/to/logs/sica_train_1gpu/checkpoint-9.pth \
#   PATH_PREFIX_FROM=/mnt/public/ PATH_PREFIX_TO=/path/to/local/dataset/ \
#   bash scripts/run_full_eval.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FH="${FORENSICHUB:-$REPO/../ForensicHub}"
PY="${PYTHON:-python}"
SCRIPTS="$REPO/scripts"

# ---- required: trained checkpoint -------------------------------------------
CKPT="${SICA_CHECKPOINT:-}"
if [[ -z "$CKPT" ]]; then
  echo "ERROR: SICA_CHECKPOINT is not set. Point it at the trained checkpoint, e.g." >&2
  echo "       export SICA_CHECKPOINT=/path/to/logs/sica_train_1gpu/checkpoint-9.pth" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

# ---- required: OpenMMSec dataset root (must contain jsons_v4/) ---------------
OPENMMSEC_ROOT="${OPENMMSEC_ROOT:-}"
if [[ -z "$OPENMMSEC_ROOT" ]]; then
  echo "ERROR: OPENMMSEC_ROOT is not set. Point it at the OpenMMSecV2 dataset root, e.g." >&2
  echo "       export OPENMMSEC_ROOT=/path/to/OpenMMSecV2" >&2
  exit 1
fi
MAN="$OPENMMSEC_ROOT/jsons_v4"
if [[ ! -d "$MAN" ]]; then
  echo "ERROR: manifest directory not found: $MAN (OPENMMSEC_ROOT=$OPENMMSEC_ROOT)" >&2
  exit 1
fi

# ---- ForensicHub (PYTHONPATH) ------------------------------------------------
if [[ ! -d "$FH" ]]; then
  echo "ERROR: ForensicHub not found at: $FH" >&2
  echo "       set FORENSICHUB to your ForensicHub checkout, e.g." >&2
  echo "       export FORENSICHUB=/path/to/ForensicHub" >&2
  exit 1
fi

# ---- optional: manifest image-path prefix remap -----------------------------
# Both must be set together (or both empty); the dumper enforces the same.
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"
if [[ -n "$PATH_PREFIX_FROM" || -n "$PATH_PREFIX_TO" ]]; then
  if [[ -z "$PATH_PREFIX_FROM" || -z "$PATH_PREFIX_TO" ]]; then
    echo "ERROR: PATH_PREFIX_FROM and PATH_PREFIX_TO must be set together (both or neither)." >&2
    exit 1
  fi
fi
PREFIX_ARGS=(--path-prefix-from "$PATH_PREFIX_FROM" --path-prefix-to "$PATH_PREFIX_TO")

# ---- output dirs (under this repo) ------------------------------------------
SUB="$REPO/reproduction_artifacts/submission_ready"
PRED="$SUB/predictions"
MET="$SUB/metrics"
LOGDIR="$SUB/logs"
mkdir -p "$PRED" "$MET" "$LOGDIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
GPU_LOCAL=0
THRESHOLD=0.5
HOST="$(hostname)"
DATE="$(date +%Y%m%d_%H%M%S)"
RUNLOG="$LOGDIR/full_eval_${HOST}_${DATE}.log"

echo "===== SICA full evaluation =====" | tee "$RUNLOG"
echo "host       : $HOST" | tee -a "$RUNLOG"
echo "date       : $(date -Iseconds)" | tee -a "$RUNLOG"
echo "ckpt       : $CKPT" | tee -a "$RUNLOG"
echo "manifests  : $MAN" | tee -a "$RUNLOG"
echo "ForensicHub: $FH" | tee -a "$RUNLOG"
echo "GPU        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (local cuda:$GPU_LOCAL)" | tee -a "$RUNLOG"
echo "threshold  : $THRESHOLD" | tee -a "$RUNLOG"
if [[ -n "$PATH_PREFIX_FROM" ]]; then
  echo "prefix     : $PATH_PREFIX_FROM -> $PATH_PREFIX_TO" | tee -a "$RUNLOG"
else
  echo "prefix     : (no remap)" | tee -a "$RUNLOG"
fi
echo "=================================" | tee -a "$RUNLOG"

# split -> manifest
declare -a SPLITS=(id_test deepfake_ood aigc_ood iml_ood doc_ood)
declare -A MANIFEST=(
  [id_test]="id_test.json"
  [deepfake_ood]="deepfake_ood.json"
  [aigc_ood]="aigc_ood.json"
  [iml_ood]="iml_ood.json"
  [doc_ood]="doc_ood.json"
)

FAIL=0
for SPLIT in "${SPLITS[@]}"; do
  echo "" | tee -a "$RUNLOG"
  echo "[$(date +%T)] >>> dump $SPLIT (${MANIFEST[$SPLIT]})" | tee -a "$RUNLOG"
  PYTHONPATH="$FH" "$PY" "$SCRIPTS/dump_sica_predictions.py" \
    --checkpoint "$CKPT" \
    --manifest  "$MAN/${MANIFEST[$SPLIT]}" \
    --out       "$PRED/${SPLIT}.csv" \
    --gpu "$GPU_LOCAL" --batch-size 64 --num-workers 8 \
    "${PREFIX_ARGS[@]}" \
    2>&1 | tee -a "$RUNLOG" || { echo "[FAIL] dump $SPLIT" | tee -a "$RUNLOG"; FAIL=1; }
done

# ---- derive total_ood_test as concatenation of the 4 OOD domain CSVs ---------
echo "" | tee -a "$RUNLOG"
echo "[$(date +%T)] >>> build total_ood_test.csv (concat of 4 OOD domain CSVs)" | tee -a "$RUNLOG"
BUILD_TOTAL=1
for S in deepfake_ood aigc_ood iml_ood doc_ood; do
  if [[ ! -s "$PRED/${S}.csv" ]]; then
    echo "[WARN] missing or empty $PRED/${S}.csv - total_ood_test will be skipped" | tee -a "$RUNLOG"
    BUILD_TOTAL=0
  fi
done
if [[ "$BUILD_TOTAL" -eq 1 ]]; then
  head -1 "$PRED/deepfake_ood.csv" > "$PRED/total_ood_test.csv"
  for S in deepfake_ood aigc_ood iml_ood doc_ood; do
    tail -n +2 "$PRED/${S}.csv" >> "$PRED/total_ood_test.csv"
  done
  wc -l "$PRED/total_ood_test.csv" | tee -a "$RUNLOG"
else
  FAIL=1
fi

# ---- aggregate every split ---------------------------------------------------
for SPLIT in id_test deepfake_ood aigc_ood iml_ood doc_ood total_ood_test; do
  echo "" | tee -a "$RUNLOG"
  echo "[$(date +%T)] >>> aggregate $SPLIT" | tee -a "$RUNLOG"
  "$PY" "$SCRIPTS/aggregate_sica_metrics.py" \
    --input "$PRED/${SPLIT}.csv" \
    --out-json "$MET/${SPLIT}.json" \
    --out-md "$MET/${SPLIT}.md" \
    --threshold "$THRESHOLD" --split-name "$SPLIT" 2>&1 | tee -a "$RUNLOG" || FAIL=1
done

echo "" | tee -a "$RUNLOG"
echo "[$(date +%T)] ===== DONE (FAIL=$FAIL) =====" | tee -a "$RUNLOG"
exit $FAIL
