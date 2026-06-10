#!/bin/bash
cd /root/ECE228-Project-WildfirePredict
source /root/autodl-tmp/wsts/venv/bin/activate
LOG=/root/autodl-tmp/wsts/logs
mkdir -p $LOG
for FOLD in 0 8; do
  echo "[$(date)] === SWA pilot fold $FOLD ==="
  python -u reports/train_swa.py $FOLD 90 0.6 2e-4 2>&1 | tee $LOG/swa_pilot_fold${FOLD}.log
done
echo "[$(date)] SWA pilot done"
grep -h "RESULT" $LOG/swa_pilot_fold*.log
