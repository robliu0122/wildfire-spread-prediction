# Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the 3 experiments specified in the Phase 2 design (T=5 multi-temporal + tuning, then + drop-PDSI, then ensemble) and update the report with new numbers.

**Architecture:** Configuration-driven: most changes are new/edited YAML files plus 2 shell driver scripts + 1 Python ensemble script. No model architecture changes.

**Tech Stack:** PyTorch Lightning 2.x (`LightningCLI`), Segmentation Models PyTorch (SMP), torchmetrics, AutoDL container with RTX 4090, SSH-based remote workflow.

---

## File Structure

### New files (on remote `~/ECE228-Project-WildfirePredict/`)

- `cfgs/data_multitemporal_full_features.yaml` — data config for T=5, all features
- `cfgs/data_multitemporal_no_pdsi.yaml` — same as above with `features_to_keep` excluding PDSI
- `reports/ensemble_inference.py` — load 12 fold checkpoints, average sigmoid probs by test-year, recompute AP
- `reports/analyze_phase2.py` — produce comparison table + fig12 (Phase 1 vs 2A vs 2B vs 2C ensemble)

### New files (on remote `/root/autodl-tmp/wsts/`)

- `run_phase2a_full.sh` — driver for sweep 2A
- `run_phase2b_full.sh` — driver for sweep 2B (differs by `--data=` arg)

### Modified files

- `cfgs/unet/res18_monotemporal.yaml` — add `lr_scheduler` (CosineAnnealingLR T_max=100) + `optimizer.init_args.weight_decay=1e-4`
- `cfgs/trainer_single_gpu.yaml` — add `gradient_clip_val: 1.0`

### Out of scope (no edits)

`src/dataloader/*.py`, `src/models/*.py`, `src/train.py` — Phase 1 already left these in working state for T>1.

---

## Stage 0: Pre-flight verification (BLOCKS everything else)

The spec marks `features_to_keep` semantics as HIGH risk. Verify it before committing to the design.

### Task 0: Verify `features_to_keep` index semantics

**Files:** none (read-only verification)

- [ ] **Step 1: SSH and run the fast_dev_run probe**

Run on remote:
```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com
source /root/autodl-tmp/wsts/venv/bin/activate
source /etc/network_turbo
cd ~/ECE228-Project-WildfirePredict
export WANDB_MODE=offline
python src/train.py \
  --config=cfgs/unet/res18_monotemporal.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
  --data.data_fold_id=0 \
  --data.num_workers=0 \
  --data.batch_size=2 \
  --data.features_to_keep='[0,1,2]' \
  --seed_everything=0 \
  --trainer.fast_dev_run=true 2>&1 | grep -E "n_channels|model.*Unet|Params|Total"
```

- [ ] **Step 2: Record observed `n_channels`**

Expected outcomes (write down which one happened):
- If model summary shows `n_channels=3` → semantics is **post-encoding indices** (0..42 after Phase 1) → safe to use `features_to_keep=[0..14, 16..42]` for drop-PDSI.
- If `n_channels` is anything else (likely 3 + landcover_one_hot = 19, or 3 + cos channels = 5+) → semantics differs → recompute the keep list before any 2B work.

- [ ] **Step 3: Document the result inline in the spec**

```bash
echo "## Verification result ($(date))" >> docs/superpowers/specs/2026-05-21-phase2-design.md
echo "features_to_keep=[0,1,2] gave n_channels=<RECORDED>" >> docs/superpowers/specs/2026-05-21-phase2-design.md
```

- [ ] **Step 4: Compute the drop-PDSI keep list**

Based on observed semantics:
- If post-encoding: `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]` (omit 15)
- If raw: `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22]` (omit 15)

Save the correct list into a variable for the rest of the plan.

- [ ] **Step 5: Commit verification result**

```bash
cd ~/ECE228-Project-WildfirePredict
git add docs/superpowers/specs/2026-05-21-phase2-design.md
git commit -m "phase2: verify features_to_keep semantics"
```

---

## Stage 1: Hyperparameter & multi-temporal config (Sweep 2A prerequisites)

### Task 1: Add cosine LR + weight decay to model yaml

**Files:**
- Modify: `cfgs/unet/res18_monotemporal.yaml`

- [ ] **Step 1: Read current yaml**

```bash
cat ~/ECE228-Project-WildfirePredict/cfgs/unet/res18_monotemporal.yaml
```

Expected: optimizer block with `class_path: torch.optim.AdamW`, `init_args.lr: 0.001`, no `weight_decay`, no `lr_scheduler` block.

- [ ] **Step 2: Edit yaml to add weight_decay and lr_scheduler**

Replace the optimizer block. The full new content:

```yaml
# pytorch_lightning==2.0.1
seed_everything: 0
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
    weight_decay: 1.0e-4
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 100
model:
  class_path: models.SMPModel
  init_args:
    encoder_name: resnet18
    n_channels: 40
    flatten_temporal_dimension: true
    pos_class_weight: 236
    loss_function: Focal

do_train: true
do_predict: false
do_test: true
```

- [ ] **Step 3: Smoke-test the yaml**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'source /root/autodl-tmp/wsts/venv/bin/activate; source /etc/network_turbo 2>/dev/null; cd ~/ECE228-Project-WildfirePredict && export WANDB_MODE=offline && python src/train.py \
  --config=cfgs/unet/res18_monotemporal.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
  --data.data_fold_id=0 \
  --data.num_workers=0 \
  --seed_everything=0 \
  --trainer.fast_dev_run=true 2>&1 | tail -20'
```

Expected: see `CosineAnnealingLR` in optimizer info; training completes 1 batch + 1 val; no schema errors.

- [ ] **Step 4: Commit**

```bash
cd ~/ECE228-Project-WildfirePredict
git add cfgs/unet/res18_monotemporal.yaml
git commit -m "phase2: add cosine LR schedule and weight_decay=1e-4"
```

### Task 2: Add gradient clipping to trainer yaml

**Files:** Modify `cfgs/trainer_single_gpu.yaml`

- [ ] **Step 1: Locate the line to edit**

```bash
grep -n "gradient_clip_val\|fast_dev_run" ~/ECE228-Project-WildfirePredict/cfgs/trainer_single_gpu.yaml
```

Expected: line showing `gradient_clip_val: null`.

- [ ] **Step 2: Edit the value**

Run on remote:
```bash
sed -i 's/gradient_clip_val: null/gradient_clip_val: 1.0/' ~/ECE228-Project-WildfirePredict/cfgs/trainer_single_gpu.yaml
grep gradient_clip ~/ECE228-Project-WildfirePredict/cfgs/trainer_single_gpu.yaml
```

Expected output: `gradient_clip_val: 1.0`

- [ ] **Step 3: Commit**

```bash
cd ~/ECE228-Project-WildfirePredict
git add cfgs/trainer_single_gpu.yaml
git commit -m "phase2: add gradient_clip_val=1.0"
```

### Task 3: Create multitemporal full-features data yaml

**Files:** Create `cfgs/data_multitemporal_full_features.yaml`

- [ ] **Step 1: Write the file**

Run on remote:
```bash
cat > ~/ECE228-Project-WildfirePredict/cfgs/data_multitemporal_full_features.yaml << 'EOF'
data_dir: DATA_DIR_PATH
batch_size: 16
n_leading_observations: 5
n_leading_observations_test_adjustment: 5
crop_side_length: 128
load_from_hdf5: true
num_workers: 2
remove_duplicate_features: true
features_to_keep: null
EOF
cat ~/ECE228-Project-WildfirePredict/cfgs/data_multitemporal_full_features.yaml
```

- [ ] **Step 2: Smoke-test with fast_dev_run**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'source /root/autodl-tmp/wsts/venv/bin/activate; source /etc/network_turbo 2>/dev/null; cd ~/ECE228-Project-WildfirePredict && export WANDB_MODE=offline && python src/train.py \
  --config=cfgs/unet/res18_monotemporal.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_multitemporal_full_features.yaml \
  --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
  --data.data_fold_id=0 \
  --data.num_workers=0 \
  --trainer.accumulate_grad_batches=4 \
  --seed_everything=0 \
  --trainer.fast_dev_run=true 2>&1 | tail -25'
```

Expected:
- Model summary shows `n_channels=131` (or close: `22*4 + 43 = 131`)
- Memory under 22 GB
- 1 batch + 1 val completes
- No worker crash

- [ ] **Step 3: If OOM, drop batch_size**

If memory exceeds 22 GB: edit `batch_size: 16` → `batch_size: 8` in the yaml and re-test.

- [ ] **Step 4: Commit**

```bash
cd ~/ECE228-Project-WildfirePredict
git add cfgs/data_multitemporal_full_features.yaml
git commit -m "phase2: multitemporal T=5 data config"
```

### Task 4: Create sweep 2A driver script

**Files:** Create `/root/autodl-tmp/wsts/run_phase2a_full.sh`

- [ ] **Step 1: Write the script (mirror run_phase1_full.sh structure)**

Run on remote:
```bash
cat > /root/autodl-tmp/wsts/run_phase2a_full.sh << 'EOF'
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
        --trainer.accumulate_grad_batches=4 \
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
EOF
chmod +x /root/autodl-tmp/wsts/run_phase2a_full.sh
head -25 /root/autodl-tmp/wsts/run_phase2a_full.sh
```

- [ ] **Step 2: Dry-run the script with one fold to verify**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'cd /root/autodl-tmp/wsts && FOLDS=(0) bash -c "echo using FOLDS=\${FOLDS[@]}"'
```

(verifies bash array syntax works in non-interactive shell)

### Task 5: Launch sweep 2A

**Files:** none (runtime operation)

- [ ] **Step 1: Verify GPU clean**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'pgrep -af "src/train.py\|run_phase" || echo CLEAN; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

Expected: `CLEAN` and < 2000 MiB.

- [ ] **Step 2: Launch via nohup**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'cd /root/autodl-tmp/wsts && nohup bash run_phase2a_full.sh > logs/run_phase2a_master.log 2>&1 & disown; sleep 10; pgrep -af "src/train.py" | head -3'
```

Expected: 3 lines showing python procs with `--data.data_fold_id=0`.

- [ ] **Step 3: Wait 10 min then sanity check first epoch**

```bash
sleep 600 && ssh -p 56165 root@connect.bjb1.seetacloud.com 'tail -3 /root/autodl-tmp/wsts/logs/phase2a_fold0.log | tr "\r" "\n" | grep -oP "val_AP=\K[0-9.]+" | tail -1; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

Expected: val_AP > 0 (e.g., 0.1 - 0.3 in early epochs); GPU memory under 22 GB; util > 30%.

- [ ] **Step 4: Schedule periodic checks**

Check every 30-60 min (depending on user availability) until master log shows "Total:" line. Each check runs Task 5's Step 3 commands.

### Task 6: Analyze sweep 2A results

**Files:** Create `reports/analyze_phase2.py`

- [ ] **Step 1: Verify all 12 folds completed**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'grep "done in" /root/autodl-tmp/wsts/logs/run_phase2a_master.log | wc -l'
```

Expected: `12`.

If less than 12, identify missing folds, delete partial logs, and re-launch via Task 5 (skip-if-done will resume).

- [ ] **Step 2: Write analyze_phase2.py**

Run on remote:
```bash
cat > ~/ECE228-Project-WildfirePredict/reports/analyze_phase2.py << 'EOF'
"""Compare baseline / phase1 / phase2a / phase2b / ensemble. Produces fig12 + phase2_metrics.csv."""
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = Path('/root/autodl-tmp/wsts/logs')
OUT_DIR = Path('/root/ECE228-Project-WildfirePredict/reports/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = OUT_DIR.parent

FOLDS = [
    (2018,2019,2020,2021),(2018,2019,2021,2020),(2018,2020,2019,2021),(2018,2020,2021,2019),
    (2018,2021,2019,2020),(2018,2021,2020,2019),(2019,2020,2018,2021),(2019,2020,2021,2018),
    (2019,2021,2018,2020),(2019,2021,2020,2018),(2020,2021,2018,2019),(2020,2021,2019,2018),
]

def parse(path):
    if not path.exists():
        return None
    m = re.search(r'\btest_AP\b\s*│\s*([0-9.]+)', path.read_text(errors='ignore'))
    return float(m.group(1)) if m else None

rows = []
for fid, (y1, y2, vy, ty) in enumerate(FOLDS):
    rows.append({
        'fold': fid, 'test_year': ty,
        'baseline': parse(LOG_DIR / f'baseline_fold{fid}.log'),
        'phase1':   parse(LOG_DIR / f'phase1_fold{fid}.log'),
        'phase2a':  parse(LOG_DIR / f'phase2a_fold{fid}.log'),
        'phase2b':  parse(LOG_DIR / f'phase2b_fold{fid}.log'),
    })
df = pd.DataFrame(rows)
df.to_csv(REPORT_DIR / 'phase2_metrics.csv', index=False)
print(df.to_string(index=False))

# Per-year means
for col in ['baseline', 'phase1', 'phase2a', 'phase2b']:
    if df[col].notna().any():
        print(f'\n{col} mean: {df[col].mean():.4f}')
        print(df.groupby('test_year')[col].mean().to_string())

# fig12: grouped bars
present = [c for c in ['baseline', 'phase1', 'phase2a', 'phase2b'] if df[c].notna().any()]
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(12)
w = 0.8 / len(present)
colors = {'baseline': '#888', 'phase1': '#4C72B0', 'phase2a': '#55A467', 'phase2b': '#C44E52'}
for i, col in enumerate(present):
    ax.bar(x + (i - (len(present)-1)/2) * w, df[col].fillna(0), w, label=col, color=colors[col])
ax.set_xticks(x)
ax.set_xticklabels([f'f{f}\n(t={t})' for f, t in zip(df.fold, df.test_year)], fontsize=8)
ax.set_ylabel('test AP')
ax.set_title('Phase 2 cumulative comparison')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig12_phase2_comparison.png', dpi=150)
plt.close()
print(f'\nSaved {OUT_DIR / "fig12_phase2_comparison.png"}')
EOF
cd ~/ECE228-Project-WildfirePredict && /root/autodl-tmp/wsts/venv/bin/python reports/analyze_phase2.py
```

- [ ] **Step 3: Decide if 2A target met**

Hard success criterion from spec: mean test AP ≥ 0.43.

If met → proceed to Task 7 (sweep 2B).
If not met → diagnose: check fold timings, look for any worker crash patterns, check whether lr scheduler actually engaged.

- [ ] **Step 4: Commit analysis script + results**

```bash
cd ~/ECE228-Project-WildfirePredict
git add reports/analyze_phase2.py reports/phase2_metrics.csv reports/figures/fig12_phase2_comparison.png
git commit -m "phase2a: results + analysis"
```

---

## Stage 2: Sweep 2B (drop PDSI)

### Task 7: Create no-PDSI data yaml

**Files:** Create `cfgs/data_multitemporal_no_pdsi.yaml`

- [ ] **Step 1: Use the verified keep list from Task 0**

Run on remote (substitute the correct list from Task 0 result):
```bash
cat > ~/ECE228-Project-WildfirePredict/cfgs/data_multitemporal_no_pdsi.yaml << 'EOF'
data_dir: DATA_DIR_PATH
batch_size: 16
n_leading_observations: 5
n_leading_observations_test_adjustment: 5
crop_side_length: 128
load_from_hdf5: true
num_workers: 2
remove_duplicate_features: true
features_to_keep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
EOF
```

(If Task 0 showed raw semantics, replace the list with `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22]`)

- [ ] **Step 2: Smoke-test**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'source /root/autodl-tmp/wsts/venv/bin/activate; source /etc/network_turbo 2>/dev/null; cd ~/ECE228-Project-WildfirePredict && export WANDB_MODE=offline && python src/train.py \
  --config=cfgs/unet/res18_monotemporal.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_multitemporal_no_pdsi.yaml \
  --data.data_dir=/root/autodl-tmp/wsts/data_hdf5 \
  --data.data_fold_id=0 \
  --data.num_workers=0 \
  --trainer.accumulate_grad_batches=4 \
  --seed_everything=0 \
  --trainer.fast_dev_run=true 2>&1 | grep -E "n_channels|Total estimated|val_AP" | head'
```

Expected: `n_channels=126` (one less per timestep than 2A's 131); no errors; val_AP logged.

- [ ] **Step 3: Commit yaml**

```bash
cd ~/ECE228-Project-WildfirePredict
git add cfgs/data_multitemporal_no_pdsi.yaml
git commit -m "phase2b: T=5 + drop PDSI data config"
```

### Task 8: Create sweep 2B driver script

**Files:** Create `/root/autodl-tmp/wsts/run_phase2b_full.sh`

- [ ] **Step 1: Copy 2A script and substitute data yaml + log names**

Run on remote:
```bash
cp /root/autodl-tmp/wsts/run_phase2a_full.sh /root/autodl-tmp/wsts/run_phase2b_full.sh
sed -i 's/phase2a/phase2b/g; s|cfgs/data_multitemporal_full_features.yaml|cfgs/data_multitemporal_no_pdsi.yaml|' /root/autodl-tmp/wsts/run_phase2b_full.sh
grep -E "phase2|cfgs/data" /root/autodl-tmp/wsts/run_phase2b_full.sh | head -5
```

Expected: all references show `phase2b` and `data_multitemporal_no_pdsi.yaml`.

### Task 9: Launch + monitor sweep 2B

**Files:** none (runtime)

- [ ] **Step 1: Verify GPU clean (after 2A finished)**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'pgrep -af "src/train.py" || echo CLEAN; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

- [ ] **Step 2: Launch**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'cd /root/autodl-tmp/wsts && nohup bash run_phase2b_full.sh > logs/run_phase2b_master.log 2>&1 & disown; sleep 10; pgrep -af "src/train.py" | head -3'
```

- [ ] **Step 3: Monitor as in Task 5 Step 4**

### Task 10: Analyze sweep 2B + verify PDSI hypothesis

**Files:** none (uses existing analyze_phase2.py)

- [ ] **Step 1: Re-run analysis after 2B completes**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'cd ~/ECE228-Project-WildfirePredict && /root/autodl-tmp/wsts/venv/bin/python reports/analyze_phase2.py'
```

- [ ] **Step 2: Check PDSI hypothesis**

From the per-year output, compare 2019 stratum:
- 2A's 2019 mean AP
- 2B's 2019 mean AP

Hypothesis confirmed if 2B 2019 mean ≥ 2A 2019 mean + 0.05.

- [ ] **Step 3: Commit**

```bash
cd ~/ECE228-Project-WildfirePredict
git add reports/phase2_metrics.csv reports/figures/fig12_phase2_comparison.png
git commit -m "phase2b: results + PDSI hypothesis test"
```

---

## Stage 3: Inference-time ensemble (Sweep 2C)

### Task 11: Write ensemble_inference.py

**Files:** Create `reports/ensemble_inference.py`

- [ ] **Step 1: Write the script**

Run on remote:
```bash
cat > ~/ECE228-Project-WildfirePredict/reports/ensemble_inference.py << 'PYEOF'
"""Load 12 phase2b checkpoints, run inference on each fold's test set, ensemble probs by test-year, recompute AP."""
import sys, json
from pathlib import Path
import torch
import torchmetrics
sys.path.insert(0, 'src')
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from models.SMPModel import SMPModel

CKPT_ROOT = Path('/root/ECE228-Project-WildfirePredict/lightning_logs/wildfire_progression')
RESULTS_PATH = Path('/root/ECE228-Project-WildfirePredict/reports/phase2c_ensemble.json')

# Map fold -> wandb run id from phase2b logs. We will resolve at runtime.
def find_best_ckpt(fold_id, prefix='phase2b'):
    log = Path(f'/root/autodl-tmp/wsts/logs/{prefix}_fold{fold_id}.log').read_text(errors='ignore')
    import re
    m = re.search(r'lightning_logs/wandb/offline-run-[0-9_]+-([a-z0-9]+)', log)
    if not m:
        raise RuntimeError(f'no run id in fold {fold_id}')
    runid = m.group(1)
    ckpts = list((CKPT_ROOT / runid / 'checkpoints').glob('best-*.ckpt'))
    if not ckpts:
        raise RuntimeError(f'no best ckpt for run {runid}')
    return ckpts[0]

FOLDS_TO_YEAR = {0:2021, 1:2020, 2:2021, 3:2019, 4:2020, 5:2019, 6:2021, 7:2018, 8:2020, 9:2018, 10:2019, 11:2018}
year_to_folds = {}
for f, y in FOLDS_TO_YEAR.items():
    year_to_folds.setdefault(y, []).append(f)

results = {}
for year, fold_list in year_to_folds.items():
    print(f'=== test year {year}: folds {fold_list} ===')
    fold_probs = {}
    fold_targets = None
    for fid in fold_list:
        ckpt = find_best_ckpt(fid)
        print(f'  loading fold {fid}: {ckpt.name}')
        model = SMPModel.load_from_checkpoint(str(ckpt), map_location='cuda').eval().cuda()
        dm = FireSpreadDataModule(
            data_dir='/root/autodl-tmp/wsts/data_hdf5',
            batch_size=1, n_leading_observations=5,
            n_leading_observations_test_adjustment=5,
            crop_side_length=128, load_from_hdf5=True, num_workers=0,
            remove_duplicate_features=True,
            features_to_keep=[i for i in range(43) if i != 15],
            data_fold_id=fid,
        )
        dm.setup('test')
        all_probs, all_targets = [], []
        for x, y in dm.test_dataloader():
            x = x.cuda()
            with torch.no_grad():
                # use the model's tiled inference via get_pred_and_gt logic if needed
                logits = model.model(x.flatten(start_dim=1, end_dim=2) if x.dim() == 5 else x)
            probs = torch.sigmoid(logits).squeeze(1).cpu()
            all_probs.append(probs)
            all_targets.append(y)
        fold_probs[fid] = torch.cat(all_probs, dim=0)
        if fold_targets is None:
            fold_targets = torch.cat(all_targets, dim=0)
    # Ensemble: average across folds
    avg_probs = torch.stack(list(fold_probs.values()), dim=0).mean(dim=0)
    ap_metric = torchmetrics.AveragePrecision('binary')
    ap_metric.update(avg_probs.flatten(), fold_targets.flatten().long())
    ap = ap_metric.compute().item()
    print(f'  ensemble AP: {ap:.4f}')
    results[year] = {'folds': fold_list, 'ensemble_AP': ap,
                     'individual_AP': {fid: 'see phase2b_metrics' for fid in fold_list}}

RESULTS_PATH.write_text(json.dumps(results, indent=2))
print(f'\nSaved to {RESULTS_PATH}')
mean_ensemble = sum(r['ensemble_AP'] for r in results.values()) / len(results)
print(f'Mean ensemble AP across 4 test years: {mean_ensemble:.4f}')
PYEOF
echo "written ensemble_inference.py"
```

- [ ] **Step 2: Smoke-test on one fold first**

Make a temporary single-year version:
```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'cd ~/ECE228-Project-WildfirePredict && /root/autodl-tmp/wsts/venv/bin/python -c "
import sys; sys.path.insert(0, \"src\")
from reports.ensemble_inference import find_best_ckpt
print(find_best_ckpt(0, prefix=\"phase2b\"))
"' 2>&1 | tail -5
```

Expected: path to a `.ckpt` file printed.

- [ ] **Step 3: Run full ensemble**

```bash
ssh -p 56165 root@connect.bjb1.seetacloud.com 'source /root/autodl-tmp/wsts/venv/bin/activate; source /etc/network_turbo 2>/dev/null; cd ~/ECE228-Project-WildfirePredict && export WANDB_MODE=disabled && /root/autodl-tmp/wsts/venv/bin/python reports/ensemble_inference.py 2>&1 | tail -20'
```

Expected: 4 years processed; "Mean ensemble AP across 4 test years: 0.XX" printed.

- [ ] **Step 4: Commit**

```bash
cd ~/ECE228-Project-WildfirePredict
git add reports/ensemble_inference.py reports/phase2c_ensemble.json
git commit -m "phase2c: inference ensemble across folds per test-year"
```

---

## Stage 4: Update report + presentation

### Task 12: Update Chinese report with Phase 2 section

**Files:** Modify `C:\Users\YongfengX\Desktop\WildfirePredict_report\experiment_report_zh.md`

- [ ] **Step 1: Read current TL;DR and identify insertion point**

```bash
grep -n "## " "$USERPROFILE/Desktop/WildfirePredict_report/experiment_report_zh.md"
```

Find the line numbers of "## 结果" and "## 下一步" sections.

- [ ] **Step 2: Update TL;DR with final number**

Replace the existing TL;DR (3 lines after `## TL;DR`) with one that includes Phase 2's final mean AP and the ensemble result.

The replacement text is determined by actual results — fill in real numbers from `phase2_metrics.csv` and `phase2c_ensemble.json`.

- [ ] **Step 3: Insert new section "## Phase 2 — multi-temporal + drop PDSI + ensemble" before "## 下一步"**

The section should include:
- Pipeline changes between Phase 1 and Phase 2 (new 4 changes: T=5, lr cosine, wd, clip; plus 2B drop-PDSI)
- New comparison table (baseline / phase1 / phase2a / phase2b / ensemble) — copy from fig12 caption
- Per-year stratified table (esp. 2019)
- PDSI hypothesis result: confirmed / partial / not confirmed
- vs WSTS+ 0.478 comparison

- [ ] **Step 4: Regenerate PDF**

```bash
cd "$USERPROFILE/Desktop/WildfirePredict_report" && pandoc experiment_report_zh.md -o experiment_report_zh.html --standalone --self-contained --css=style.css --metadata title="WildfireSpreadTS Phase 2 进展" && CHROME="C:\Program Files\Google\Chrome\Application\chrome.exe" && "$CHROME" --headless=new --disable-gpu --print-to-pdf-no-header "--print-to-pdf=C:\Users\YongfengX\Desktop\WildfirePredict_report\experiment_report_zh_v3.pdf" "file:///C:/Users/YongfengX/Desktop/WildfirePredict_report/experiment_report_zh.html"
```

Expected: `experiment_report_zh_v3.pdf` created.

### Task 13: Sync all artifacts local

**Files:** none (sync operation)

- [ ] **Step 1: Sync new figures + csv**

```bash
for f in fig12_phase2_comparison.png; do scp -P 56165 root@connect.bjb1.seetacloud.com:/root/ECE228-Project-WildfirePredict/reports/figures/$f "$USERPROFILE/Desktop/WildfirePredict_report/figures/$f"; done
scp -P 56165 root@connect.bjb1.seetacloud.com:/root/ECE228-Project-WildfirePredict/reports/phase2_metrics.csv "$USERPROFILE/Desktop/WildfirePredict_report/"
scp -P 56165 root@connect.bjb1.seetacloud.com:/root/ECE228-Project-WildfirePredict/reports/phase2c_ensemble.json "$USERPROFILE/Desktop/WildfirePredict_report/"
ls "$USERPROFILE/Desktop/WildfirePredict_report/"
```

- [ ] **Step 2: Commit on remote**

```bash
cd ~/ECE228-Project-WildfirePredict && git add reports/ docs/ cfgs/ && git status && git commit -m "phase2: full sweep + ensemble + report update"
```

### Task 14: Update presentation bullet points

**Files:** Create or edit `C:\Users\YongfengX\Desktop\WildfirePredict_report\presentation_outline.md`

- [ ] **Step 1: Write slide-by-slide outline using framework C from brainstorming**

The 8 slides established in the brainstorming session, with bullet points filled in from actual Phase 2 results:

1. Title
2. Problem + dataset (1 image: fire spread visualization)
3. WSTS+ paper SOTA + open problem (year heterogeneity)
4. Our two-track approach (slide title: "Replicate + Investigate")
5. Track 1: Phase 1 + Phase 2 changes (pipeline diagram fig11, updated to show 2A additions; result number)
6. Track 2: PDSI smoking gun (fig7 EDA + fig with 2A 2019 vs 2B 2019 if hypothesis confirmed)
7. Final results (fig12 comparison + ensemble number + vs WSTS+)
8. Limitations + next steps

- [ ] **Step 2: Validate presentation against final numbers**

Check that bullet points reference real numbers from `phase2_metrics.csv`.

---

## Self-Review

### Spec coverage check
- ✅ Sweep 2A (T=5 + cosine + wd + clip) → Tasks 1-6
- ✅ Sweep 2B (drop PDSI) → Tasks 7-10
- ✅ Sweep 2C (ensemble) → Task 11
- ✅ `features_to_keep` verification (HIGH risk) → Task 0
- ✅ Report update → Tasks 12-13
- ✅ Presentation bullet points → Task 14

### Risk address check
- ✅ T=5 OOM → Task 3 Step 3 has fallback to batch=8
- ✅ features_to_keep semantics → Task 0 is the gate
- ✅ Worker crashes → relies on Phase 1 mitigations (already applied)
- ✅ Cosine LR yaml syntax → Task 1 Step 3 smoke-tests it
- ✅ 2B partial improvement → still informative per spec; Task 10 Step 2 quantifies

### Success criteria check
- Hard (2A ≥ 0.43) → Task 6 Step 3 checks
- Soft (2B ≥ 0.46, 2019 ≥ 0.32) → Task 10 Step 2 checks
- Stretch (2C ≥ 0.478) → Task 11 Step 3 reports

### Type / name consistency
- `phase2a_fold{N}.log` / `phase2b_fold{N}.log` / `phase2c_ensemble.json` — used consistently
- `fig12_phase2_comparison.png` — used consistently
- `cfgs/data_multitemporal_full_features.yaml` / `cfgs/data_multitemporal_no_pdsi.yaml` — used consistently
