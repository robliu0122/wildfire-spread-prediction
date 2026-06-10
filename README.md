# Wildfire Spread Prediction (WildfireSpreadTS, extended)

This repository extends the **WildfireSpreadTS (WSTS)** next-day wildfire-spread
benchmark (Gerard et al., NeurIPS 2023 Datasets & Benchmarks) with a sequence of
reproducible improvements. Starting from the original repository baseline, the
**12-fold leave-one-year-out mean test AP improves from 0.349 to 0.473**, matching
the published ResNet18-UNet state of the art (WSTS+, ~0.468) and surpassing it on
the hardest test year (2019).

> Fork of <https://github.com/SebastianGer/WildfireSpreadTS>. The original
> dataset, model code, and paper are credited under [Acknowledgements](#acknowledgements).

## Headline results

12-fold mean test AP (Average Precision; the design has exactly 3 folds per test
year, so the 12-fold mean equals the 4-year-stratum mean):

| Stage | Configuration | T | Mean AP | Δ |
|-------|---------------|---|---------|---|
| Baseline | original repo (Dice loss, no pretrain, val-loss selection) | 1 | 0.349 | — |
| Phase 1 | + sin/cos angle fix, ImageNet pretrain, Focal loss, val-AP selection | 1 | 0.397 | +13.8% |
| Phase 2A | + 5-day temporal stack, cosine LR, weight decay, grad-clip | 5 | 0.386 | −2.8% |
| Phase 2B | Phase 2A − PDSI channel | 5 | 0.429 | +11.2% |
| Phase 2C | per-year 3-fold inference ensemble (no retrain) | 5 | 0.470 | +9.4% |
| + D4-TTA | dihedral test-time augmentation | 5 | **0.473** | +0.7% |

Per-year breakdown (Phase 2C ensemble vs. published ResNet18-UNet):

| Test year | Phase 2C | + TTA | WSTS+ (paper) |
|-----------|----------|-------|---------------|
| 2018 | 0.480 | 0.483 | ~0.49 |
| 2019 | 0.337 | 0.340 | ~0.31 |
| 2020 | 0.484 | 0.488 | ~0.42 |
| 2021 | 0.577 | 0.581 | ~0.57 |

Two findings not present in the original benchmark:
- **Dropping the PDSI (drought-index) channel** removes a strongly year-shifting
  covariate and rescues a catastrophic fold (fold 10: 0.040 → 0.310).
- **A per-year inference ensemble** adds +0.040 mean AP at zero extra training
  cost, helping the highest-variance stratum (2019) the most.

## Repository layout

```
src/                      original WSTS code (+ our bug fixes)
  dataloader/             FireSpreadDataset, FireSpreadDataModule
  models/                 SMPModel (U-Net), UTAE, ConvLSTM, ...
  preprocess/             CreateHDF5Dataset.py
  train.py                LightningCLI entrypoint
cfgs/                     YAML configs (model / data / trainer / sweeps)
scripts/                  per-phase sweep drivers (skip-if-done, idempotent)
reports/                  analysis scripts + result JSON/CSV (numbers only)
```

Large artifacts (the dataset, `.ckpt` checkpoints, `lightning_logs/`, W&B logs)
are intentionally **not** tracked — see `.gitignore`. The committed `reports/*.json`
and `reports/*.csv` hold the result numbers so the tables above are verifiable
without rerunning anything.

## Setup

Python 3.10. We use [`uv`](https://github.com/astral-sh/uv) (pip also works):

```bash
uv venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Data

The dataset is on Zenodo (CC-BY-4.0): <https://doi.org/10.5281/zenodo.8006177>.
Download it, then convert to HDF5 for fast training:

```bash
python src/preprocess/CreateHDF5Dataset.py \
    --data_dir  YOUR_RAW_DATA_DIR \
    --target_dir YOUR_HDF5_DIR
```

You can skip HDF5 and pass `--data.load_from_hdf5=False`, but training will be
much slower. Point experiments at your data with `--data.data_dir YOUR_HDF5_DIR`
(or edit the `data_dir` field in the `cfgs/data_*.yaml` files).

## Reproduce

Training uses [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html);
each experiment is a `--config` (model) + `--data` + `--trainer` triple. W&B
logging can be disabled with `WANDB_MODE=disabled`.

A single fold, by hand:

```bash
python src/train.py \
    --config=cfgs/unet/res18_monotemporal.yaml \
    --trainer=cfgs/trainer_single_gpu.yaml \
    --data=cfgs/data_monotemporal_full_features.yaml \
    --seed_everything=0 --do_test=True \
    --data.data_dir YOUR_HDF5_DIR
```

The full 12-fold sweeps are wrapped in idempotent driver scripts (edit
`DATA_DIR` / `LOG_DIR` near the top of each before running):

```bash
bash scripts/run_phase1_full.sh        # Phase 1, T=1            (~6-10 h GPU)
bash scripts/run_phase2a_full.sh       # Phase 2A, T=5 all feats (~12 h GPU)
bash scripts/run_phase2b_A.sh          # Phase 2B, no-PDSI, folds subset A
bash scripts/run_phase2b_B.sh          # Phase 2B, no-PDSI, folds subset B
```

Aggregate per-fold / per-year / 12-fold means:

```bash
python reports/analyze_phase2.py       # -> reports/phase2_summary.json
```

Phase 2C inference ensemble (reuses the Phase 2B checkpoints, no training):

```bash
python reports/ensemble_inference.py        # -> reports/phase2c_ensemble.json
python reports/tta_ensemble_inference.py     # + D4-TTA -> reports/phase2c_tta_ensemble.json
```

### Optional / exploratory

```bash
bash scripts/run_swa_pilot.sh          # SWA pilot (reported as a negative ablation)
python reports/utae_cpu_check.py       # sanity-check the UTAE temporal model wiring
bash scripts/run_phase3_utae.sh        # UTAE sweep scaffold (not in headline results)
python scripts/compute_pdsi_year_stats.py   # PDSI cross-year drift statistics
```

## Key changes vs. upstream

| File | Change | Phase |
|------|--------|-------|
| `src/dataloader/FireSpreadDataset.py` | angle features: `sin` → (`sin`, `cos`) | 1 |
| `src/models/SMPModel.py` | encoder weights `None` → `imagenet` | 1 |
| `src/models/BaseModel.py` | add `val_AP` metric; fix Focal-loss `alpha` | 1 |
| `cfgs/trainer_single_gpu.yaml` | `gradient_clip_val=1.0`; early-stop / checkpoint on `val_AP` | 1–2A |
| `cfgs/unet/res18_monotemporal.yaml` | AdamW + cosine LR (`T_max=100`) + weight decay `1e-4` | 2A |
| `cfgs/data_multitemporal_full_features.yaml` | `T=5`, batch 64 | 2A |
| `cfgs/data_multitemporal_no_pdsi.yaml` | drop PDSI (raw channel index 15) | 2B |
| `reports/analyze_phase2.py` | per-fold / per-year aggregator + PDSI check | 2 |
| `reports/ensemble_inference.py` | per-year 3-fold ensemble (incremental, resumable) | 2C |
| `reports/tta_ensemble_inference.py` | D4-flip TTA on the ensemble | 2C |
| `scripts/run_phase*` | idempotent sweep drivers | 1–2 |

The minimal Phase-1 diff against the upstream commit is saved at
`reports/phase1_changes.diff`.

## Acknowledgements

This work builds directly on WildfireSpreadTS. Please cite the original paper:

```bibtex
@inproceedings{gerard2023wildfirespreadts,
  title     = {WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction},
  author    = {Sebastian Gerard and Yu Zhao and Josephine Sullivan},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2023},
  url        = {https://openreview.net/forum?id=RgdGkPRQ03}
}
```

- Original code: <https://github.com/SebastianGer/WildfireSpreadTS>
- Dataset-creation code: <https://github.com/SebastianGer/WildfireSpreadTSCreateDataset>
- The UTAE model under `src/models/utae_paps_models/` is from
  [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps).

Two known upstream issues this fork addresses: the dataset-class bug fixed
upstream in `ab3c8f3`, and the angle-feature `sin`-only encoding (should use
`sin` **and** `cos`). See the original repo's notes for details.

## License

Same license as the upstream project — see [LICENSE](LICENSE).
