"""Phase 0 EDA: why is 2019 so much harder than other test years?

Computes for each year (2018-2021):
  1. Active-fire base rate (positive pixels / total pixels)
  2. Per-fire size distribution
  3. Geographic distribution of fires
  4. Distribution of key dynamic features (sampled), conditional on fire vs. no-fire

Outputs PNG charts + a CSV of per-fire stats to reports/figures/.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/root/autodl-tmp/wsts/data_hdf5")
OUT_DIR = Path("/root/ECE228-Project-WildfirePredict/reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Channel indices into the 23-channel HDF5 (base feature ordering)
CH = {
    "NDVI": 3, "EVI2": 4, "Precip": 5,
    "WindSpeed": 6, "WindDir": 7,
    "MinTemp": 8, "MaxTemp": 9, "ERC": 10, "Humidity": 11,
    "Slope": 12, "Aspect": 13, "Elevation": 14, "PDSI": 15,
    "LandCover": 16,
    "ActiveFire": 22,
}
YEARS = [2018, 2019, 2020, 2021]
YEAR_COLORS = {2018: "#4C72B0", 2019: "#DD8452",
               2020: "#55A467", 2021: "#C44E52"}
RNG = np.random.RandomState(0)
PIXELS_PER_FILE = 200  # cap on sampled pixels per file for feature distributions

# ============================================================
# Pass over all files: collect per-fire stats and sampled features
# ============================================================
per_fire_rows = []
# year -> feature_name -> list of (sampled value, is_fire 0/1)
feature_samples = defaultdict(lambda: defaultdict(list))

for year in YEARS:
    files = sorted(glob.glob(str(DATA_ROOT / str(year) / "*.hdf5")))
    print(f"[{year}] {len(files)} files")
    for path in files:
        with h5py.File(path, "r") as h:
            data = h["data"]
            T, C, H, W = data.shape
            attrs = dict(data.attrs)
            lng, lat = float(attrs["lnglat"][0]), float(attrs["lnglat"][1])

            # --- fire mask (across all days in this file) ---
            af = data[:, CH["ActiveFire"], :, :]  # (T, H, W)
            af = np.nan_to_num(af, nan=0.0)
            fire_mask = af > 0
            total_pixels = T * H * W
            fire_pixels = int(fire_mask.sum())

            per_fire_rows.append({
                "year": year,
                "fire_name": attrs["fire_name"],
                "T": T, "H": H, "W": W,
                "total_pixels": total_pixels,
                "fire_pixels": fire_pixels,
                "base_rate": fire_pixels / total_pixels,
                "lng": lng, "lat": lat,
            })

            # --- sample pixels for feature-distribution EDA ---
            # Sample some random (t, y, x) positions; stratify by fire/no-fire
            n_fire = min(PIXELS_PER_FILE // 2, fire_pixels)
            n_nofire = PIXELS_PER_FILE - n_fire

            fire_idx = np.argwhere(fire_mask)
            nofire_idx = np.argwhere(~fire_mask)

            if len(fire_idx) > 0 and n_fire > 0:
                pick = RNG.choice(len(fire_idx), size=n_fire, replace=False)
                fire_sel = fire_idx[pick]
            else:
                fire_sel = np.zeros((0, 3), dtype=int)
            if len(nofire_idx) > 0 and n_nofire > 0:
                pick = RNG.choice(len(nofire_idx), size=n_nofire, replace=False)
                nofire_sel = nofire_idx[pick]
            else:
                nofire_sel = np.zeros((0, 3), dtype=int)

            for name, ch in CH.items():
                if name in ("LandCover", "ActiveFire"):
                    continue
                channel_data = data[:, ch, :, :]  # (T, H, W)
                for sel, is_fire in [(fire_sel, 1), (nofire_sel, 0)]:
                    if len(sel) == 0:
                        continue
                    vals = channel_data[sel[:, 0], sel[:, 1], sel[:, 2]]
                    vals = vals[np.isfinite(vals)]
                    feature_samples[year][name].extend(
                        [(float(v), is_fire) for v in vals]
                    )

df = pd.DataFrame(per_fire_rows)
csv_path = OUT_DIR.parent / "per_fire_stats.csv"
df.to_csv(csv_path, index=False)
print(f"\nWrote {csv_path}")

# ============================================================
# Per-year summary
# ============================================================
agg = df.groupby("year").agg(
    n_fires=("fire_name", "count"),
    total_pixels=("total_pixels", "sum"),
    fire_pixels=("fire_pixels", "sum"),
    median_fire_size=("fire_pixels", "median"),
    mean_fire_size=("fire_pixels", "mean"),
).reset_index()
agg["base_rate"] = agg["fire_pixels"] / agg["total_pixels"]
agg["fires_with_zero_fire_pixels"] = df.groupby("year").apply(
    lambda g: int((g.fire_pixels == 0).sum())).values
print("\n=== Per-year summary ===")
print(agg.to_string(index=False))

# ============================================================
# Chart 4: Base rate per year + n_fires
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
years = agg.year.tolist()
colors = [YEAR_COLORS[y] for y in years]

bars = ax1.bar([str(y) for y in years], agg.base_rate * 100,
               color=colors, edgecolor="black")
ax1.set_ylabel("active-fire base rate (% of all pixels)")
ax1.set_xlabel("year")
ax1.set_title("Fire prevalence per year")
for b, v in zip(bars, agg.base_rate * 100):
    ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}%",
             ha="center", va="bottom", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

bars = ax2.bar([str(y) for y in years], agg.n_fires,
               color=colors, edgecolor="black")
ax2.set_ylabel("# fire events")
ax2.set_xlabel("year")
ax2.set_title("Number of fire events per year")
for b, v in zip(bars, agg.n_fires):
    ax2.text(b.get_x() + b.get_width() / 2, v, str(int(v)),
             ha="center", va="bottom", fontsize=9)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_base_rate_and_count.png", dpi=150)
plt.close()

# ============================================================
# Chart 5: Per-fire size distribution (log y)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
positions = np.arange(len(YEARS))
data_per_year = [df[df.year == y].fire_pixels.values for y in YEARS]
bp = ax.boxplot(data_per_year, positions=positions, widths=0.6,
                patch_artist=True, showfliers=True,
                medianprops=dict(color="black", lw=1.5),
                flierprops=dict(marker=".", markersize=4, alpha=0.5))
for patch, y in zip(bp["boxes"], YEARS):
    patch.set_facecolor(YEAR_COLORS[y])
    patch.set_alpha(0.6)
ax.set_yscale("symlog", linthresh=1)
ax.set_xticks(positions)
ax.set_xticklabels([str(y) for y in YEARS])
ax.set_xlabel("year")
ax.set_ylabel("fire pixels per file (symlog)")
ax.set_title("Per-fire size distribution")
ax.grid(axis="y", alpha=0.3)
# Annotate median
for i, y in enumerate(YEARS):
    m = np.median(data_per_year[i])
    ax.text(positions[i] + 0.32, max(m, 1), f"median={int(m)}",
            va="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_fire_size_distribution.png", dpi=150)
plt.close()

# ============================================================
# Chart 6: Geographic distribution
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))
for y in YEARS:
    sub = df[df.year == y]
    ax.scatter(sub.lng, sub.lat, color=YEAR_COLORS[y], s=20, alpha=0.6,
               edgecolor="black", linewidth=0.3, label=f"{y} (n={len(sub)})")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("Fire event locations by year")
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_geographic.png", dpi=150)
plt.close()

# ============================================================
# Chart 7: Dynamic feature distributions (fire-pixel only)
#   shows whether the conditions at fire pixels differ across years
# ============================================================
features_to_plot = ["NDVI", "Precip", "WindSpeed",
                    "MaxTemp", "Humidity", "ERC", "PDSI", "Elevation"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes = axes.ravel()
for i, fname in enumerate(features_to_plot):
    ax = axes[i]
    for y in YEARS:
        samples = feature_samples[y].get(fname, [])
        if not samples:
            continue
        vals_fire = np.array([v for v, isf in samples if isf == 1])
        if len(vals_fire) == 0:
            continue
        # Trim extreme outliers for plotting (keep 1st-99th percentile)
        lo, hi = np.percentile(vals_fire, [1, 99])
        vals_fire = vals_fire[(vals_fire >= lo) & (vals_fire <= hi)]
        ax.hist(vals_fire, bins=40, alpha=0.4, color=YEAR_COLORS[y],
                label=str(y), density=True)
    ax.set_title(fname)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)
fig.suptitle("Feature distributions at fire pixels, by year", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig7_feature_distributions_at_fire.png",
            dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# Persist agg + per-feature summary stats
# ============================================================
year_feature_means = {}
for y in YEARS:
    year_feature_means[y] = {}
    for fname in features_to_plot:
        samples = feature_samples[y].get(fname, [])
        fire_vals = [v for v, isf in samples if isf == 1]
        nofire_vals = [v for v, isf in samples if isf == 0]
        year_feature_means[y][fname] = {
            "fire_mean": float(np.mean(fire_vals)) if fire_vals else None,
            "fire_std": float(np.std(fire_vals)) if fire_vals else None,
            "nofire_mean": float(np.mean(nofire_vals)) if nofire_vals else None,
            "nofire_std": float(np.std(nofire_vals)) if nofire_vals else None,
        }
agg_dict = {
    "per_year": agg.to_dict(orient="records"),
    "feature_means_at_fire": year_feature_means,
}
(OUT_DIR.parent / "phase0_eda.json").write_text(json.dumps(agg_dict,
                                                           indent=2,
                                                           default=float))
print("\nWrote:", OUT_DIR.parent / "phase0_eda.json")
print("Figures in:", OUT_DIR)
