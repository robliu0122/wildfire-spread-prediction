"""Parse baseline fold logs and generate charts for the experiment summary."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = Path("/root/autodl-tmp/wsts/logs")
OUT_DIR = Path("/root/ECE228-Project-WildfirePredict/reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Year split table (from src/dataloader/FireSpreadDataModule.py::split_fires)
FOLDS = [
    (2018, 2019, 2020, 2021),
    (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021),
    (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020),
    (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021),
    (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020),
    (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019),
    (2020, 2021, 2019, 2018),
]

METRIC_KEYS = ["test_AP", "test_f1", "test_iou",
               "test_loss", "test_precision", "test_recall"]


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    out = {}
    for key in METRIC_KEYS:
        # Match lines like "│          test_AP          │     0.4751...     │"
        m = re.search(rf"\b{key}\b\s*│\s*([0-9.]+)", text)
        if m:
            out[key] = float(m.group(1))
    return out


rows = []
for fold_id, (y1, y2, val_y, test_y) in enumerate(FOLDS):
    log = LOG_DIR / f"baseline_fold{fold_id}.log"
    if not log.exists():
        continue
    metrics = parse_log(log)
    if "test_AP" not in metrics:
        continue
    rows.append({
        "fold": fold_id,
        "train_years": f"{y1},{y2}",
        "val_year": val_y,
        "test_year": test_y,
        **metrics,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR.parent / "baseline_metrics.csv", index=False)
print(df.to_string(index=False))
print("\nMean test_AP =", df.test_AP.mean().round(4),
      " Std =", df.test_AP.std().round(4))

# Color per test year (consistent across charts)
YEAR_COLORS = {2018: "#4C72B0", 2019: "#DD8452",
               2020: "#55A467", 2021: "#C44E52"}

# ============================================================
# Chart 1: test_AP per fold, colored by test year
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [YEAR_COLORS[y] for y in df.test_year]
bars = ax.bar(df.fold.astype(str), df.test_AP, color=colors, edgecolor="black",
              linewidth=0.5)
mean_ap = df.test_AP.mean()
ax.axhline(mean_ap, ls="--", color="gray", lw=1,
           label=f"mean = {mean_ap:.3f}")
ax.set_xlabel("Fold ID")
ax.set_ylabel("test_AP")
ax.set_title("Baseline ResNet18-UNet — test_AP across 12 folds")
# Annotate value on top of each bar
for bar, ap in zip(bars, df.test_AP):
    ax.text(bar.get_x() + bar.get_width() / 2, ap + 0.008,
            f"{ap:.3f}", ha="center", va="bottom", fontsize=8)
# Legend by test year
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=YEAR_COLORS[y], edgecolor="black",
                      label=f"test year {y}")
                for y in sorted(YEAR_COLORS)]
legend_elems.append(plt.Line2D([0], [0], ls="--", color="gray",
                               label=f"mean = {mean_ap:.3f}"))
ax.legend(handles=legend_elems, loc="upper right", fontsize=9)
ax.set_ylim(0, max(df.test_AP) * 1.18)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_ap_per_fold.png", dpi=150)
plt.close()

# ============================================================
# Chart 2: test_AP grouped by test year (the headline finding)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
years = sorted(df.test_year.unique())
data = [df[df.test_year == y].test_AP.values for y in years]
positions = np.arange(len(years))

# Boxplot with individual points overlaid
bp = ax.boxplot(data, positions=positions, widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.5))
for patch, y in zip(bp["boxes"], years):
    patch.set_facecolor(YEAR_COLORS[y])
    patch.set_alpha(0.6)
# Scatter individual fold values
for i, (y, vals) in enumerate(zip(years, data)):
    jitter = np.random.RandomState(0).uniform(-0.06, 0.06, len(vals))
    ax.scatter(positions[i] + jitter, vals, color=YEAR_COLORS[y],
               edgecolor="black", s=60, zorder=3)
    # Mean line per group
    ax.hlines(np.mean(vals), positions[i] - 0.28, positions[i] + 0.28,
              color="black", lw=1.5, ls=":", alpha=0.7)

ax.set_xticks(positions)
ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel("Test year")
ax.set_ylabel("test_AP")
ax.set_title("test_AP grouped by test year (3 folds each)")
# Annotate means
for i, vals in enumerate(data):
    ax.text(positions[i] + 0.32, np.mean(vals),
            f"μ={np.mean(vals):.3f}", va="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(df.test_AP) * 1.15)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_ap_by_test_year.png", dpi=150)
plt.close()

# ============================================================
# Chart 3: Multi-metric per fold (AP, F1, IoU)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.5))
folds = df.fold.astype(str).values
x = np.arange(len(folds))
width = 0.26
ax.bar(x - width, df.test_AP, width, label="test_AP",
       color="#3a86ff", edgecolor="black", lw=0.4)
ax.bar(x,         df.test_f1, width, label="test_F1",
       color="#8338ec", edgecolor="black", lw=0.4)
ax.bar(x + width, df.test_iou, width, label="test_IoU",
       color="#ff006e", edgecolor="black", lw=0.4)
ax.set_xticks(x)
ax.set_xticklabels(folds)
# Mark test year underneath
for i, ty in enumerate(df.test_year.values):
    ax.text(x[i], -0.05, str(ty), ha="center", va="top",
            fontsize=8, color=YEAR_COLORS[ty])
ax.text(-0.85, -0.05, "test yr:", ha="right", va="top", fontsize=8,
        color="gray")
ax.set_xlabel("Fold ID")
ax.set_ylabel("metric value")
ax.set_title("Baseline metrics per fold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 0.75)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_multi_metric_per_fold.png", dpi=150)
plt.close()

# Save aggregated stats as JSON for the writeup
stats = {
    "n_folds_completed": len(df),
    "overall_mean_test_AP": float(df.test_AP.mean()),
    "overall_std_test_AP": float(df.test_AP.std()),
    "by_test_year": {
        int(y): {
            "n": int((df.test_year == y).sum()),
            "mean_test_AP": float(df[df.test_year == y].test_AP.mean()),
            "min_test_AP": float(df[df.test_year == y].test_AP.min()),
            "max_test_AP": float(df[df.test_year == y].test_AP.max()),
        }
        for y in sorted(df.test_year.unique())
    },
}
(OUT_DIR.parent / "baseline_stats.json").write_text(json.dumps(stats, indent=2))
print("\nWrote:", OUT_DIR.parent / "baseline_stats.json")
print("Figures in:", OUT_DIR)
