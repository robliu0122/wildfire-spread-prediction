#!/bin/bash
set -e

cd /root/ECE228-Project-WildfirePredict
source /root/autodl-tmp/wsts/venv/bin/activate
source /etc/network_turbo 2>/dev/null || true
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

LOG_DIR=/root/autodl-tmp/wsts/logs
MASTER_LOG=$LOG_DIR/run_phase1_master.log
mkdir -p $LOG_DIR

# Phase 1 = angle bug fix + ImageNet pretraining + Focal loss + val_AP early stopping
# Comparison anchor: baseline_fold*.log (12-fold mean test_AP = 0.349)

# Suggested run order: easy years first (2021), then 2020/2018, then hardest (2019)
# fold 0,1,2 test=2021 | fold 3,5,10 test=2019 | fold 4,6,7 test=2020 | fold 8,9,11 test=2018
FOLDS=(0 1 2 4 6 7 8 9 11 3 5 10)

TOTAL_START=$(date +%s)
echo "[$(date)] phase1 full sweep starting on $(nvidia-smi -L | head -1)"

for FOLD in "${FOLDS[@]}"; do
    LOG="$LOG_DIR/phase1_fold${FOLD}.log"

    if [ -f "$LOG" ] && grep -q "test_AP" "$LOG"; then
        AP=$(grep -oP "test_AP\s*\xe2\x94\x82\s*\K[0-9.]+" "$LOG" | tail -1)
        echo "[skip] Fold $FOLD already done, test_AP = $AP"
        continue
    fi

    FOLD_START=$(date +%s)
    echo
    echo "========================================"
    echo "[$(date)] phase1 fold $FOLD"
    echo "========================================"

    python src/train.py \
        --config=cfgs/unet/res18_monotemporal.yaml \
        --trainer=cfgs/trainer_single_gpu.yaml \
        --data=cfgs/data_monotemporal_full_features.yaml \
        --seed_everything=0 \
        --trainer.max_epochs=100 \
        --do_test=True \
        --data.data_fold_id=$FOLD \
        --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
        --data.num_workers=2 \
        2>&1 | tee "$LOG"

    FOLD_DUR=$(( $(date +%s) - FOLD_START ))
    AP=$(grep -oP "test_AP\s*\xe2\x94\x82\s*\K[0-9.]+" "$LOG" | tail -1)
    echo "[$(date)] phase1 fold $FOLD done in $((FOLD_DUR/60))min, test_AP = $AP"
done

TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))
echo
echo "========================================"
echo "Total time: $((TOTAL_DUR/3600))h$(((TOTAL_DUR%3600)/60))m"
echo "========================================"

echo
echo "=== phase1 vs baseline ==="
printf "%-6s %-12s %-12s %-10s\n" "Fold" "baseline_AP" "phase1_AP" "delta"
SUM_P=0; COUNT=0
for F in 0 1 2 3 4 5 6 7 8 9 10 11; do
    BASE_LOG="$LOG_DIR/baseline_fold${F}.log"
    PHASE_LOG="$LOG_DIR/phase1_fold${F}.log"
    BASE_AP=$(grep -oP "test_AP\s*\xe2\x94\x82\s*\K[0-9.]+" "$BASE_LOG" 2>/dev/null | tail -1)
    PHASE_AP=$(grep -oP "test_AP\s*\xe2\x94\x82\s*\K[0-9.]+" "$PHASE_LOG" 2>/dev/null | tail -1)
    if [ -n "$PHASE_AP" ]; then
        DELTA=$(python -c "print(f\"{$PHASE_AP - $BASE_AP:+.4f}\")" 2>/dev/null || echo "n/a")
        printf "%-6s %-12s %-12s %-10s\n" "$F" "${BASE_AP:-n/a}" "$PHASE_AP" "$DELTA"
        SUM_P=$(python -c "print($SUM_P + $PHASE_AP)")
        COUNT=$((COUNT + 1))
    fi
done
if [ $COUNT -gt 0 ]; then
    MEAN_P=$(python -c "print(f\"{$SUM_P / $COUNT:.4f}\")")
    echo "phase1 mean across $COUNT folds: $MEAN_P  (baseline 12-fold mean = 0.349)"
fi
