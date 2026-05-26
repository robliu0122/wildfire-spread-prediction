#!/bin/bash
set -e
cd /root/ECE228-Project-WildfirePredict
source /root/autodl-tmp/wsts/venv/bin/activate
source /etc/network_turbo 2>/dev/null || true
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

LOG_DIR=/root/autodl-tmp/wsts/logs
mkdir -p $LOG_DIR

# Phase 2A = T=5 multi-temporal + Phase 1 + cosine LR + weight_decay + grad clip
FOLDS=(0 1 2 4 6 7 8 9 11 3 5 10)
TOTAL_START=$(date +%s)
echo "[$(date)] phase2a sweep starting on $(nvidia-smi -L | head -1)"

for FOLD in "${FOLDS[@]}"; do
    LOG="$LOG_DIR/phase2a_fold${FOLD}.log"
    if [ -f "$LOG" ] && grep -q "test_AP" "$LOG"; then
        AP=$(grep -oP "test_AP\s*\xe2\x94\x82\s*\K[0-9.]+" "$LOG" | tail -1)
        echo "[skip] Fold $FOLD already done, test_AP = $AP"
        continue
    fi

    FOLD_START=$(date +%s)
    echo
    echo "========================================"
    echo "[$(date)] phase2a fold $FOLD"
    echo "========================================"

    python src/train.py \
        --config=cfgs/unet/res18_monotemporal.yaml \
        --trainer=cfgs/trainer_single_gpu.yaml \
        --data=cfgs/data_multitemporal_full_features.yaml \
        --seed_everything=0 \
        --trainer.max_epochs=100 \
        --trainer.accumulate_grad_batches=1 \
        --do_test=True \
        --data.data_fold_id=$FOLD \
        --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
        2>&1 | tee "$LOG"

    FOLD_DUR=$(( $(date +%s) - FOLD_START ))
    AP=$(grep -oP "test_AP\s*\xe2\x94\x82\s*\K[0-9.]+" "$LOG" | tail -1)
    echo "[$(date)] phase2a fold $FOLD done in $((FOLD_DUR/60))min, test_AP = $AP"
done

TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))
echo
echo "Total: $((TOTAL_DUR/3600))h$(((TOTAL_DUR%3600)/60))m"
