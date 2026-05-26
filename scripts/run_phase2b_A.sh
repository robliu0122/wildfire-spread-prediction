#!/bin/bash
set -e
cd /root/ECE228-Project-WildfirePredict
source /root/autodl-tmp/wsts/venv/bin/activate
source /etc/network_turbo 2>/dev/null || true
export OMP_NUM_THREADS=8
export WANDB_MODE=online

LOG_DIR=/root/autodl-tmp/wsts/logs
mkdir -p $LOG_DIR

FOLDS=(1 4 6 8)
for FOLD in "${FOLDS[@]}"; do
    LOG="$LOG_DIR/phase2b_fold${FOLD}.log"
    if [ -f "$LOG" ] && grep -q "test_AP" "$LOG"; then
        echo "[skip] fold $FOLD already done"
        continue
    fi
    echo "[$(date)] starting fold $FOLD"
    python src/train.py \
        --config=cfgs/unet/res18_monotemporal.yaml \
        --trainer=cfgs/trainer_single_gpu.yaml \
        --data=cfgs/data_multitemporal_no_pdsi.yaml \
        --seed_everything=0 --trainer.max_epochs=100 \
        --trainer.accumulate_grad_batches=1 --do_test=True \
        --data.data_fold_id=$FOLD \
        --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
        2>&1 | tee "$LOG"
done
echo "[$(date)] sweep done"
