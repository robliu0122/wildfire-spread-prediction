"""Windows-safe HDF5 conversion for the WildfireSpreadTS subset.

Equivalent to src/preprocess/CreateHDF5Dataset.py but avoids that file's
POSIX-only `split("/src")` path hack. Converts every fire folder present under
<root>/data_raw/<year>/fire_*/ into <root>/data_hdf5/<year>/<fire>.hdf5.
Resumable: existing hdf5 files are skipped.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'ECE228-Project-WildfirePredict-main',
                   'ECE228-Project-WildfirePredict-main', 'src')
sys.path.insert(0, SRC)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
from pathlib import Path
from dataloader.FireSpreadDataset import FireSpreadDataset

DATA_RAW = os.path.join(ROOT, 'data_raw')
TARGET = os.path.join(ROOT, 'data_hdf5')
YEARS = [2018, 2019, 2020, 2021]


def main():
    ds = FireSpreadDataset(data_dir=DATA_RAW, included_fire_years=YEARS,
                           n_leading_observations=1, crop_side_length=128,
                           load_from_hdf5=False, is_train=True,
                           remove_duplicate_features=False, stats_years=(2018, 2019))
    for y in YEARS:
        Path(os.path.join(TARGET, str(y))).mkdir(parents=True, exist_ok=True)

    n = 0
    for year, fire_name, img_dates, lnglat, imgs in ds.get_generator_for_hdf5():
        h5_path = os.path.join(TARGET, str(year), f"{fire_name}.hdf5")
        if Path(h5_path).is_file():
            continue
        with h5py.File(h5_path, "w") as f:
            dset = f.create_dataset("data", imgs.shape, data=imgs)
            dset.attrs["year"] = year
            dset.attrs["fire_name"] = fire_name
            dset.attrs["img_dates"] = img_dates
            dset.attrs["lnglat"] = lnglat
        n += 1
        if n % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] converted {n} fires (last {year}/{fire_name}, shape {imgs.shape})", flush=True)
    print(f"DONE. Converted {n} new fires into {TARGET}")


if __name__ == '__main__':
    main()
