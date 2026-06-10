#!/bin/bash
# Phase 3: UTAE (temporal self-attention) 12-fold leave-year-out sweep, no-PDSI.
# Mirrors run_phase2b_*.sh but swaps in the UTAE model + temporal data config.
# Skips folds already finished (log contains test_AP). Pass fold ids as args,
# e.g. `bash run_phase3_utae.sh 0 1 2 3` ; default = all 12.
set -e
cd /root/ECE228-Project-WildfirePredict
source /root/autodl-tmp/wsts/venv/bin/activate
export OMP_NUM_THREADS=4
export WANDB_MODE=online

LOG_DIR=/root/autodl-tmp/wsts/logs
mkdir -p "$LOG_DIR"

FOLDS=("$@")
if [ ${#FOLDS[@]} -eq 0 ]; then FOLDS=(0 1 2 3 4 5 6 7 8 9 10 11); fi

for FOLD in "${FOLDS[@]}"; do
    LOG="$LOG_DIR/phase3_utae_fold${FOLD}.log"
    if [ -f "$LOG" ] && grep -q "test_AP" "$LOG"; then
        echo "[skip] fold $FOLD already done"
        continue
    fi
    echo "[$(date)] starting UTAE fold $FOLD"
    python src/train.py \
        --config=cfgs/UTAE/utae_phase3_no_pdsi.yaml \
        --trainer=cfgs/trainer_single_gpu.yaml \
        --data=cfgs/data_multitemporal_utae_no_pdsi.yaml \
        --seed_everything=0 --trainer.max_epochs=100 \
        --trainer.accumulate_grad_batches=1 --do_test=True \
        --data.data_fold_id=$FOLD \
        --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
        2>&1 | tee "$LOG"
done
echo "[$(date)] UTAE sweep done"
