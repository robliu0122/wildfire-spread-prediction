"""Compare phase1 sweep against baseline. Produces phase1_metrics.csv + figs 8/9/10."""
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
    (2018, 2019, 2020, 2021), (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021), (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020), (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021), (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020), (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019), (2020, 2021, 2019, 2018),
]

METRIC_KEYS = ['test_AP', 'test_f1', 'test_iou',
               'test_loss', 'test_precision', 'test_recall']
YEAR_COLORS = {2018: '#4C72B0', 2019: '#DD8452', 2020: '#55A467', 2021: '#C44E52'}


def parse(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(errors='ignore')
    out = {}
    for key in METRIC_KEYS:
        m = re.search(rf'\b{key}\b\s*│\s*([0-9.]+)', text)
        if m:
            out[key] = float(m.group(1))
    return out


rows = []
for fold_id, (y1, y2, val_y, test_y) in enumerate(FOLDS):
    base = parse(LOG_DIR / f'baseline_fold{fold_id}.log')
    ph1 = parse(LOG_DIR / f'phase1_fold{fold_id}.log')
    if 'test_AP' not in base or 'test_AP' not in ph1:
        continue
    rows.append({
        'fold': fold_id,
        'train_years': f'{y1},{y2}',
        'val_year': val_y,
        'test_year': test_y,
        'baseline_AP': base['test_AP'],
        'phase1_AP': ph1['test_AP'],
        'delta': ph1['test_AP'] - base['test_AP'],
        'pct': (ph1['test_AP'] - base['test_AP']) / base['test_AP'] * 100,
        'phase1_f1': ph1.get('test_f1'),
        'phase1_iou': ph1.get('test_iou'),
        'phase1_precision': ph1.get('test_precision'),
        'phase1_recall': ph1.get('test_recall'),
    })

df = pd.DataFrame(rows)
df.to_csv(REPORT_DIR / 'phase1_metrics.csv', index=False)
print(df.to_string(index=False))
print()
print(f'baseline mean: {df.baseline_AP.mean():.4f}  std: {df.baseline_AP.std():.4f}')
print(f'phase1   mean: {df.phase1_AP.mean():.4f}  std: {df.phase1_AP.std():.4f}')
print(f'mean delta:   {df.delta.mean():+.4f}  ({df.pct.mean():+.1f}%)')
print()
yr_table = df.groupby('test_year').agg({'baseline_AP': 'mean', 'phase1_AP': 'mean'}).reset_index()
yr_table['delta'] = yr_table['phase1_AP'] - yr_table['baseline_AP']
yr_table['pct'] = yr_table['delta'] / yr_table['baseline_AP'] * 100
print('=== Per test-year ===')
print(yr_table.to_string(index=False))

# --- fig8: side-by-side bars per fold ---
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(df))
w = 0.38
b1 = ax.bar(x - w/2, df.baseline_AP, w, label='baseline (Dice + val_loss)', color='#888888')
b2 = ax.bar(x + w/2, df.phase1_AP, w, label='phase1 (Focal + val_AP + ImageNet + angle-fix)',
            color=[YEAR_COLORS[y] for y in df.test_year])
ax.set_xticks(x)
ax.set_xticklabels([f'f{f}\n(test={y})' for f, y in zip(df.fold, df.test_year)], fontsize=9)
ax.set_ylabel('test AP')
ax.set_title('Per-fold test AP: baseline vs phase1')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
for bx, (b, p) in zip(x, zip(df.baseline_AP, df.phase1_AP)):
    delta = p - b
    color = 'green' if delta > 0 else 'red'
    ax.annotate(f'{delta:+.03f}', xy=(bx + w/2, p), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=8, color=color)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig8_phase1_vs_baseline_per_fold.png', dpi=150)
plt.close()

# --- fig9: per-year mean comparison ---
fig, ax = plt.subplots(figsize=(7, 5))
years = sorted(yr_table.test_year)
x = np.arange(len(years))
base_means = [yr_table[yr_table.test_year == y].baseline_AP.iloc[0] for y in years]
ph1_means = [yr_table[yr_table.test_year == y].phase1_AP.iloc[0] for y in years]
ax.bar(x - 0.2, base_means, 0.4, label='baseline', color='#888888')
ax.bar(x + 0.2, ph1_means, 0.4, label='phase1', color=[YEAR_COLORS[y] for y in years])
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel('test year')
ax.set_ylabel('mean test AP (n=3 per year)')
ax.set_title('Per-test-year mean: baseline vs phase1')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for xi, (b, p) in zip(x, zip(base_means, ph1_means)):
    ax.annotate(f'{(p-b)/b*100:+.0f}%', xy=(xi + 0.2, p), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=10,
                color='green' if p > b else 'red', weight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig9_phase1_vs_baseline_per_year.png', dpi=150)
plt.close()

# --- fig10: delta distribution ---
fig, ax = plt.subplots(figsize=(7, 4.5))
df_sorted = df.sort_values('delta').reset_index(drop=True)
colors = [YEAR_COLORS[y] for y in df_sorted.test_year]
ax.barh(range(len(df_sorted)), df_sorted.delta, color=colors)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels([f'f{f} (test={y})' for f, y in zip(df_sorted.fold, df_sorted.test_year)])
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('phase1 - baseline test AP')
ax.set_title('Phase1 - baseline (sorted)')
ax.grid(axis='x', alpha=0.3)
for i, d in enumerate(df_sorted.delta):
    ax.annotate(f'{d:+.03f}', xy=(d, i), xytext=(4 if d > 0 else -4, 0),
                textcoords='offset points', va='center',
                ha='left' if d > 0 else 'right', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig10_phase1_delta_sorted.png', dpi=150)
plt.close()

print()
print('Saved:', OUT_DIR / 'fig8_phase1_vs_baseline_per_fold.png')
print('Saved:', OUT_DIR / 'fig9_phase1_vs_baseline_per_year.png')
print('Saved:', OUT_DIR / 'fig10_phase1_delta_sorted.png')
print('Saved:', REPORT_DIR / 'phase1_metrics.csv')
