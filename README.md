# WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction

This repository contains the code for recreating the experiments in the WildfireSpreadTS paper. 

- [Link to main paper](https://openreview.net/pdf?id=RgdGkPRQ03)
- [Link to supplementary material](https://openreview.net/attachment?id=RgdGkPRQ03&name=supplementary_material)

## Updates 
- After publishing the paper, we discovered a bug in the dataset class. Based on initial experiments, the corrected dataset class leads to slightly higher performance, but the trends in the results are basically the same as those reported in the paper. The bug was fixed in commit `ab3c8f35c5ec8c52c306a4488eaeb71a5a13d0de`, in case you want to roll-back the change to compare with the results in the paper.

- **Feb 2026:** An observant researcher, who kindly reached out to me, pointed out that when dealing with angle features, the current code only transforms the features via sin, but should use both sin and cos, to not lose information. This happens [here](https://github.com/SebastianGer/WildfireSpreadTS/blob/main/src/dataloader/FireSpreadDataset.py#L339). I'm currently too occupied with deadlines in other projects to make sure that nothing breaks when I fix this, so if you're working with these features, be aware of this issue. 


## Setup the environment

``` pip3 install -r requirements.txt ```

## Preparing the dataset

The dataset is freely available at [https://doi.org/10.5281/zenodo.8006177](https://doi.org/10.5281/zenodo.8006177) under CC-BY-4.0. For training, you will need to convert them to HDF5 files, which take up twice as much space but allow for much faster training.

To convert the dataset to HDF5, run:
```python3 src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR```
 substituting the path to your local dataset and where you want the HDF5 version of the dataset to be created. 

You can skip this step, and simply pass `--data.load_from_hdf5=False` on the command line, but be aware that you won't be able to perform training at any reasonable speed. 

## Re-running the baseline experiments

We use wandb to log experimental results. This can be turned off by setting the environment variable `WANDB_MODE=disabled`. The results will then be logged to a local directory instead.

Experiments are parameterized via yaml files in the `cfgs` directory. Arguments are parsed via the [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

Grid searches and repetitions of experiments were done via WandB sweeps. Those are parameterized via yaml files in the `cfgs` directory prefixed with `wandb_`. For example, to run the experiments that Table 3 in the main paper is based on, you can run a wandb sweep with `cfgs/unet/wandb_table3.yaml`. For explanations on how to use wandb sweeps please refer to the [original documentation](https://docs.wandb.ai/guides/sweeps). To run the same experiments without WandB, the parameters specified in the WandB sweep configuration file can simply be passed via the command line. 

For example, to train the U-net architecture on one day of observations, you could pass arguments on the command line as follows:

```
python3 train.py --config=cfgs/unet/res18_monotemporal.yaml --trainer=cfgs/trainer_single_gpu.yaml --data=cfgs/data_monotemporal_full_features.yaml --seed_everything=0 --trainer.max_epochs=200 --do_test=True --data.data_dir YOUR_DATA_DIR
```
where you replace `YOUR_DATA_DIR` with the path to your local HDF5 dataset. Alternatively, you can permanently set the data directory in the respective data config files. Parameters defined in config files are overwritten by command-line arguments. Later arguments overwrite previously given arguments. 

## Re-creating the dataset

The code to create the dataset using Google Earth Engine is available at [https://github.com/SebastianGer/WildfireSpreadTSCreateDataset](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset).


## Using the dataset for your own experiments

To use the dataset outside of the baseline experiments, you can use the Lightning Datamodule at `src/dataloader/FireSpreadDataModule.py` which directly provides dataset loaders for train/val/test set. Alternatively, you can use the PyTorch dataset at `src/dataloader/FireSpreadDataset.py`. 

## Citation

```
@inproceedings{
    gerard2023wildfirespreadts,
    title={WildfireSpread{TS}: A dataset of multi-modal time series for wildfire spread prediction},
    author={Sebastian Gerard and Yu Zhao and Josephine Sullivan},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=RgdGkPRQ03}
}
```

---

# Our Improvements (ECE 228 fork)

This fork systematically extends the original WSTS baseline through **5 experimental phases**. The 12-fold mean test AP improved from **0.349 → 0.470**, matching the WSTS+ paper Res18-UNet SOTA (0.468 single-day / 0.472 multi-temporal). See `reports/phase2_paper_zh.tex` (and the rendered PDF in `paper_rewriting_output/final_paper/`) for the full progress report.

## Phase summary

| Phase | Configuration | 12-fold mean AP |
|-------|--------------|----------------|
| Phase 0 | Original repo baseline (T=1, Dice, val_loss) | 0.349 |
| Phase 1 | + angle sin/cos bug fix, + ImageNet pretrain, + Focal loss (alpha bug fixed), + val_AP early stopping | 0.397 (+14%) |
| Phase 2A | + T=5 multi-temporal, + cosine LR, + weight_decay 1e-4, + grad clip | 0.386 (−3%, fold 10 collapse) |
| Phase 2B | Phase 2A minus PDSI channel (raw idx 15) | 0.429 (+11%) |
| **Phase 2C** | Per-test-year 3-fold inference ensemble (no retrain) | **0.470 (+10%)** |

Key independent findings (not in WSTS+ paper):
- **PDSI ablation is a double-edged sword**: rescued fold 10 (+0.270) and fold 8 (+0.185), but regressed fold 3 (−0.162). Same test year (2019) splits both directions — PDSI is a train/val-composition-dependent dual-role feature.
- **Per-year inference ensemble (Phase 2C)** delivers +0.040 mean AP at zero additional training cost; the 2019 stratum (highest variance) gains the most (+0.077).

## Reproduction

### 1. Environment

```bash
# Use uv (recommended) or pip
uv venv /path/to/wsts_env
source /path/to/wsts_env/bin/activate
uv pip install -r requirements.txt
```

### 2. Data

Same as the original WSTS instructions above — download from Zenodo, convert to HDF5 via `src/preprocess/CreateHDF5Dataset.py`. The sweep scripts below assume the HDF5 data is at `/root/autodl-tmp/wsts/data_hdf5`; edit the `--data.data_dir` argument in the scripts to match your path.

### 3. Run the sweeps (sequential)

```bash
# Phase 1 single-day baseline (~6-10h GPU)
bash scripts/run_phase1_full.sh

# Phase 2A multi-temporal sweep (~12h GPU)
bash scripts/run_phase2a_full.sh

# Phase 2B no-PDSI sweep (~8h GPU, run A and B in parallel for ~4-5h wall time)
bash scripts/run_phase2b_A.sh  # folds 1, 4, 6, 8
bash scripts/run_phase2b_B.sh  # folds 3, 5, 7, 9
# Note: folds 0, 2, 10, 11 are completed by the orchestrator earlier; the A/B
# scripts use disjoint fold lists to avoid race conditions.
```

Each sweep script writes per-fold logs to `/root/autodl-tmp/wsts/logs/` (edit `LOG_DIR` in the scripts) and uses skip-if-done logic, so reruns are idempotent.

### 4. Analyze

```bash
# Phase 1 vs 2A vs 2B per-fold + per-year + 12-fold mean
python reports/analyze_phase2.py
# -> reports/phase2_summary.json

# Phase 2C per-year inference ensemble (~30-50 min GPU)
python reports/ensemble_inference.py
# -> reports/phase2c_ensemble.json
```

### 5. Generate figures (local, no GPU)

Sync the two JSON files to your local report directory, then:

```bash
python make_phase2_figures.py  # writes fig12-15*.png
```

## Key code changes vs upstream

| File | Change | Phase |
|------|--------|-------|
| `src/dataloader/FireSpreadDataset.py:337-350` | Angle features: sin → (sin, cos) | 1 |
| `src/models/SMPModel.py:31` | encoder_weights=imagenet | 1 |
| `src/models/BaseModel.py` | val_AP metric + Focal alpha bug fix | 1 |
| `cfgs/trainer_single_gpu.yaml` | gradient_clip_val=1.0, EarlyStopping monitor=val_AP | 1-2A |
| `cfgs/unet/res18_monotemporal.yaml` | AdamW + cosine LR + weight_decay 1e-4 | 2A |
| `cfgs/data_multitemporal_full_features.yaml` | T=5, bs=64, nw=8 | 2A |
| `cfgs/data_multitemporal_no_pdsi.yaml` | + features_to_keep excludes raw idx 15 (PDSI) | 2B |
| `src/dataloader/FireSpreadDataModule.py:83-92` | prefetch_factor=8, persistent_workers=True | 2A-2B |
| `reports/analyze_phase2.py` | Phase 1/2A/2B aggregator + PDSI hypothesis check | 2 |
| `reports/ensemble_inference.py` | Per-year 3-fold ensemble with incremental save + resume | 2C |
| `scripts/run_phase2*` | Sweep drivers with skip-if-done | 2 |

## Reference

The full progress report (Chinese, with all figures and ablation tables):

- `paper_rewriting_output/final_paper/main.tex` — LaTeX source
- `paper_rewriting_output/final_paper/figures/` — figures fig12-15
- `paper_rewriting_output/integrity_audit.md` — number-verification audit

Compile with XeLaTeX (Overleaf or local MiKTeX/TeX Live):

```bash
cd paper_rewriting_output/final_paper
xelatex main.tex
xelatex main.tex   # second pass for cross-references
```
