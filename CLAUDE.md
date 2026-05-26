# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Fork of [SebastianGer/WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS) — a PyTorch Lightning baseline for next-day wildfire spread prediction on multi-modal satellite time series (NeurIPS 2023 Datasets & Benchmarks). Being re-run as a UCSD ECE228 course project on an AutoDL container with a single RTX 4090.

## Environment activation (AutoDL-specific)

Two non-obvious commands every fresh shell needs (originals live in `指令.txt`):

```bash
source /root/autodl-tmp/wsts/venv/bin/activate    # project venv — NOT the system miniconda
source /etc/network_turbo                          # AutoDL outbound proxy — needed for pip, git, wandb
```

The WandB API key is also stored in `指令.txt`; that file must not be committed.

## Running training

Entry point is `src/train.py`, driven by `LightningCLI`. Configs in `cfgs/` are composed:

```bash
# from repo root, venv active
python3 src/train.py \
    --config=cfgs/unet/res18_monotemporal.yaml \
    --trainer=cfgs/trainer_single_gpu.yaml \
    --data=cfgs/data_monotemporal_full_features.yaml \
    --seed_everything=0 --trainer.max_epochs=200 --do_test=True \
    --data.data_dir /path/to/hdf5_dataset
```

- Configs merge in order; later `--config` files and inline flags override earlier ones.
- Set `WANDB_MODE=disabled` to skip WandB and log locally to `./lightning_logs/`.
- Hyperparameter sweeps live in `cfgs/**/wandb_*.yaml`; run them with `wandb sweep` + `wandb agent` (see upstream README).

## Preprocessing TIF → HDF5

Required for any reasonable training throughput; raw TIF mode (`--data.load_from_hdf5=False`) works but is very slow.

```bash
python3 src/preprocess/CreateHDF5Dataset.py --data_dir <TIF_ROOT> --target_dir <HDF5_OUT>
```

Import-path quirk: this script uses `from src.dataloader...` and prepends the repo root to `sys.path` itself, so always invoke it from the repo root. (`train.py` in contrast uses `from dataloader...` and only works because `python3 src/train.py` adds `src/` to `sys.path[0]`.)

## Architecture — things you can't see from one file

### Custom `LightningCLI` overrides two model args at runtime

`src/train.py::MyLightningCLI.before_instantiate_classes` rewrites whatever the model YAML sets for:

- `model.init_args.n_channels` — recomputed from `data.n_leading_observations`, `data.features_to_keep`, `data.remove_duplicate_features` via `FireSpreadDataset.get_n_features`.
- `model.init_args.pos_class_weight` — recomputed from the train fold's fire-pixel rate via `get_means_stds_missing_values`.

So editing those two values in a model YAML has no effect. Change the data config instead.

### Model hierarchy

All models extend `src/models/BaseModel.py` (`pl.LightningModule`, abstract). It owns:

- Train/val/test loop, F1/precision/recall/AUPRC metrics, and WandB image logging.
- Loss selection: `BCE | Focal | Lovasz | Jaccard | Dice` (the latter three come from `segmentation_models_pytorch.losses`; `pos_class_weight` is honored only for BCE/Focal).
- A tiled-inference wrapper triggered by `required_img_size` — needed for ConvLSTM/UTAE which can't handle arbitrary spatial sizes. The test loader uses `batch_size=1` for the same reason.

Concrete models in `src/models/`:
- `SMPModel` — U-Net family via `segmentation_models_pytorch` (encoder_name configurable).
- `ConvLSTMLightning`, `UTAELightning` — temporal models wrapping the upstream `utae_paps_models/` code.
- `LogisticRegression`, `PersistenceModel` — baselines.

### Data folds are hardcoded year permutations

`FireSpreadDataModule.split_fires` defines 12 train/val/test splits (`data.data_fold_id=0..11`), each (train=2 years, val=1, test=1) drawn from {2018, 2019, 2020, 2021}. To reproduce paper tables, sweep `data_fold_id` over the relevant range.

### Known upstream issues (not fixed here)

- Angle features (wind direction etc.) only get sin-transformed, losing information — see `src/dataloader/FireSpreadDataset.py:339`.
- A dataset-class bug existed before commit `ab3c8f35c5ec8c52c306a4488eaeb71a5a13d0de`; metrics after that commit run slightly higher than the paper's reported numbers.
