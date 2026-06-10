"""Train + test ONE model on ONE fold of the (subset) WildfireSpreadTS data.

Self-contained: avoids the repo's LightningCLI/wandb entry point, but replicates
Erkang's Phase-1/2B recipe exactly so baseline vs FNO is apples-to-apples:
  AdamW lr=1e-3 wd=1e-4 | CosineAnnealingLR T_max=100 | grad clip 1.0 |
  Focal loss | best checkpoint by val_AP | same DataModule / splits.

Usage:
  python run_fold.py --fold 5 --model baseline      --max_epochs 60
  python run_fold.py --fold 5 --model fno_bottleneck --max_epochs 60
  python run_fold.py --fold 5 --model fno_16x16      --max_epochs 60

Appends a row to tools/results.csv: model,fold,test_year,test_AP,test_loss,best_val_AP,epochs,seconds
"""
import os, sys, csv, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'ECE228-Project-WildfirePredict-main',
                   'ECE228-Project-WildfirePredict-main', 'src')
sys.path.insert(0, SRC)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["WANDB_MODE"] = "disabled"

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values
from models import SMPModel, FNOSMPModel

DATA_HDF5 = os.path.join(ROOT, 'data_hdf5')
RESULTS = os.path.join(os.path.dirname(__file__), 'results.csv')
TEST_YEAR = {0: 2021, 2: 2021, 6: 2021, 3: 2019, 5: 2019, 10: 2019}


def build_model(kind, n_channels, pos_class_weight):
    common = dict(encoder_name='resnet18', n_channels=n_channels,
                  flatten_temporal_dimension=True, pos_class_weight=pos_class_weight,
                  loss_function='Focal')
    if kind == 'baseline':
        m = SMPModel(**common)
    elif kind == 'fno_bottleneck':
        m = FNOSMPModel(fno_stage=5, fno_modes_h=2, fno_modes_w=3, n_fno_blocks=1, **common)
    elif kind == 'fno_16x16':
        m = FNOSMPModel(fno_stage=3, fno_modes_h=8, fno_modes_w=8, n_fno_blocks=1, **common)
    else:
        raise ValueError(kind)

    def configure_optimizers():
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return {"optimizer": opt, "lr_scheduler": sch}
    m.configure_optimizers = configure_optimizers
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--model', required=True,
                    choices=['baseline', 'fno_bottleneck', 'fno_16x16'])
    ap.add_argument('--max_epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    pl.seed_everything(0, workers=True)
    wandb.init(mode="disabled")          # satisfies BaseModel.on_test_epoch_end

    n_channels = FireSpreadDataset.get_n_features(1, None, False)
    train_years, _, _ = FireSpreadDataModule.split_fires(args.fold)
    _, _, mv = get_means_stds_missing_values(train_years)
    pos_class_weight = float(1.0 / (1.0 - mv[-1]))
    print(f"fold={args.fold} model={args.model} n_channels={n_channels} "
          f"pos_class_weight={pos_class_weight:.1f}", flush=True)

    dm = FireSpreadDataModule(
        data_dir=DATA_HDF5, batch_size=args.batch_size, n_leading_observations=1,
        n_leading_observations_test_adjustment=5, crop_side_length=128,
        load_from_hdf5=True, num_workers=args.num_workers,
        remove_duplicate_features=False, features_to_keep=None,
        return_doy=False, data_fold_id=args.fold)

    model = build_model(args.model, n_channels, pos_class_weight)

    run_name = f"{args.model}_fold{args.fold}"
    ckpt_cb = ModelCheckpoint(monitor='val_AP', mode='max', save_top_k=1,
                              filename='best-{epoch}-{val_AP:.4f}')
    es_cb = EarlyStopping(monitor='val_AP', mode='max', patience=15, verbose=True)
    trainer = pl.Trainer(
        accelerator='gpu', devices=1, precision='32-true',
        max_epochs=args.max_epochs, gradient_clip_val=1.0,
        deterministic='warn', num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=CSVLogger(os.path.join(ROOT, 'lightning_logs'), name=run_name),
        callbacks=[ckpt_cb, es_cb],
        default_root_dir=os.path.join(ROOT, 'lightning_logs'))

    t0 = time.time()
    trainer.fit(model, dm)
    test_metrics = trainer.test(model, dm, ckpt_path='best')[0]
    secs = time.time() - t0

    row = dict(model=args.model, fold=args.fold, test_year=TEST_YEAR.get(args.fold, '?'),
               test_AP=round(test_metrics.get('test_AP', float('nan')), 5),
               test_loss=round(test_metrics.get('test_loss', float('nan')), 5),
               best_val_AP=round(float(ckpt_cb.best_model_score), 5),
               epochs=trainer.current_epoch, seconds=round(secs, 1))
    write_header = not os.path.exists(RESULTS)
    with open(RESULTS, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print("RESULT", row, flush=True)


if __name__ == '__main__':
    main()
