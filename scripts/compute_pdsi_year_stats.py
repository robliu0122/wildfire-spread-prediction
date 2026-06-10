"""Compute per-year PDSI statistics from HDF5 data.

Run from project root (venv active):
    python3 scripts/compute_pdsi_year_stats.py [--data_dir /path/to/data_hdf5]

Writes reports/pdsi_year_stats.json which is required when
year_normalize_pdsi=true in the data config.

PDSI is raw feature channel 15 (0-indexed) in the HDF5 files.
NaN values (missing observations) are excluded from statistics.
"""
import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np

PDSI_CHANNEL = 15
YEARS = [2018, 2019, 2020, 2021]
OUT_PATH = Path('reports/pdsi_year_stats.json')


def compute_year_stats(data_dir: str, year: int) -> tuple[float, float, int]:
    files = sorted(glob.glob(f'{data_dir}/{year}/*.hdf5'))
    if not files:
        raise FileNotFoundError(f'No HDF5 files found for year {year} in {data_dir}')

    chunks = []
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            pdsi = f['data'][:, PDSI_CHANNEL, :, :]  # (T, H, W)
        valid = pdsi[~np.isnan(pdsi)].astype(np.float64)
        if valid.size > 0:
            chunks.append(valid)

    if not chunks:
        return 0.0, 1.0, 0

    combined = np.concatenate(chunks)
    return float(np.mean(combined)), float(np.std(combined)), int(combined.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/root/autodl-tmp/wsts/data_hdf5',
                        help='Root directory containing year subdirectories of HDF5 files')
    args = parser.parse_args()

    stats: dict = {}
    for year in YEARS:
        print(f'Processing {year}...', end=' ', flush=True)
        mean, std, n = compute_year_stats(args.data_dir, year)
        stats[str(year)] = {'mean': round(mean, 6), 'std': round(std, 6)}
        print(f'mean={mean:.4f}  std={std:.4f}  n_pixels={n:,}')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(stats, indent=2))
    print(f'\nSaved to {OUT_PATH}')


if __name__ == '__main__':
    main()
