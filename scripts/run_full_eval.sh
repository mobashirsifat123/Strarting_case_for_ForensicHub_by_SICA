#!/usr/bin/env bash
# run_full_eval.sh - FINAL baseline evaluation of checkpoint-9.pth on all splits.
#
# Dumps per-image predictions for id_test + 4 OOD domain sets, then derives
# total_ood_test by concatenating the 4 domain CSVs (verified to be the exact
# path-union, pairwise disjoint), then aggregates ACC/AUC/AP/F1 per split.
#
# Single GPU, world_size=1 (no torchrun, no NCCL collectives) -> safe alongside
# other users' GPU jobs. Run inside tmux. ~12-15 min on a free GPU.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FH=$REPO/ForensicHub
PY="${PYTHON:-python}"
SUB=$REPO/reproduction_artifacts/submission_ready
CKPT=$REPO/reproduction_artifacts/logs/sica_train_1gpu/checkpoint-9.pth
MAN=/mnt/nas/public/public_datasets/OpenMMSecV2/jsons_v4
PRED=$SUB/predictions
MET=$SUB/metrics
LOGDIR=$SUB/logs
mkdir -p "$PRED" "$MET" "$LOGDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
GPU_LOCAL=0
THRESHOLD=0.5
HOST=$(hostname)
DATE=$(date +%Y%m%d_%H%M%S)
RUNLOG=$LOGDIR/full_eval_${HOST}_${DATE}.log

echo "===== SICA full evaluation =====" | tee "$RUNLOG"
echo "host      : $HOST" | tee -a "$RUNLOG"
echo "date      : $(date -Iseconds)" | tee -a "$RUNLOG"
echo "ckpt      : $CKPT" | tee -a "$RUNLOG"
echo "GPU       : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (local cuda:$GPU_LOCAL)" | tee -a "$RUNLOG"
echo "threshold : $THRESHOLD" | tee -a "$RUNLOG"
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
  PYTHONPATH="$FH" "$PY" "$SUB/scripts/dump_sica_predictions.py" \
    --checkpoint "$CKPT" \
    --manifest  "$MAN/${MANIFEST[$SPLIT]}" \
    --out       "$PRED/${SPLIT}.csv" \
    --gpu "$GPU_LOCAL" --batch-size 64 --num-workers 8 \
    2>&1 | tee -a "$RUNLOG" || { echo "[FAIL] dump $SPLIT" | tee -a "$RUNLOG"; FAIL=1; }
done

# ---- derive total_ood_test as concatenation of the 4 OOD domain CSVs ---------
echo "" | tee -a "$RUNLOG"
echo "[$(date +%T)] >>> build total_ood_test.csv (concat of 4 OOD domain CSVs)" | tee -a "$RUNLOG"
head -1 "$PRED/deepfake_ood.csv" > "$PRED/total_ood_test.csv"
for S in deepfake_ood aigc_ood iml_ood doc_ood; do
  tail -n +2 "$PRED/${S}.csv" >> "$PRED/total_ood_test.csv"
done
wc -l "$PRED/total_ood_test.csv" | tee -a "$RUNLOG"

# ---- aggregate every split ---------------------------------------------------
for SPLIT in id_test deepfake_ood aigc_ood iml_ood doc_ood total_ood_test; do
  echo "" | tee -a "$RUNLOG"
  echo "[$(date +%T)] >>> aggregate $SPLIT" | tee -a "$RUNLOG"
  "$PY" "$SUB/scripts/aggregate_sica_metrics.py" \
    --input "$PRED/${SPLIT}.csv" \
    --out-json "$MET/${SPLIT}.json" \
    --out-md "$MET/${SPLIT}.md" \
    --threshold "$THRESHOLD" --split-name "$SPLIT" 2>&1 | tee -a "$RUNLOG" || FAIL=1
done

echo "" | tee -a "$RUNLOG"
echo "[$(date +%T)] ===== DONE (FAIL=$FAIL) =====" | tee -a "$RUNLOG"
exit $FAIL
