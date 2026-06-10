"""Improvement 2 pilot: train one no-PDSI fold with Stochastic Weight Averaging
and test the SWA-AVERAGED weights (not the best/last checkpoint).

Why a dedicated script instead of train.py:
  Lightning's SWA callback only copies the averaged params into the module at
  on_train_end -- AFTER ModelCheckpoint has written best/last. So train.py's
  ckpt="best" test path would test a NON-averaged model. Here we call
  trainer.test(model, dm) with no ckpt_path right after fit(), so we test the
  in-memory model, which IS the SWA average + BN-updated.

wandb note: BaseModel.on_test_epoch_end unconditionally calls wandb.log() to
log the confusion matrix / PR curve. With no WandbLogger that raises
"You must call wandb.init() before wandb.log()" AFTER test_AP is already
computed -- killing the run before RESULT prints. We call wandb.init(mode=
"disabled") so those wandb.log calls become no-ops.

Usage: python train_swa.py <fold_id> [max_epochs] [swa_start_frac] [swa_lr]
"""
import os
import sys
import types

os.environ['WANDB_MODE'] = 'disabled'

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint

sys.path.insert(0, 'src')
from dataloader.FireSpreadDataModule import FireSpreadDataModule  # noqa: E402
from dataloader.FireSpreadDataset import FireSpreadDataset  # noqa: E402
from dataloader.utils import get_means_stds_missing_values  # noqa: E402
from models.SMPModel import SMPModel  # noqa: E402

DATA_DIR = '/root/autodl-tmp/wsts/data_hdf5'
NO_PDSI_KEEP = [i for i in range(43) if i != 15]

LR = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX = 100  # same cosine schedule as phase2b before SWA kicks in


def main() -> None:
    fold = int(sys.argv[1])
    max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    swa_start_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    swa_lr = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-4

    # Make BaseModel.on_test_epoch_end's wandb.log() calls no-ops.
    wandb.init(mode='disabled')

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(0, workers=True)

    n_obs = 5
    n_features = FireSpreadDataset.get_n_features(n_obs, NO_PDSI_KEEP, True)

    train_years, _, _ = FireSpreadDataModule.split_fires(fold)
    _, _, mvr = get_means_stds_missing_values(train_years)
    pos_class_weight = float(1.0 / (1.0 - mvr[-1]))

    print(f'[swa] fold={fold} n_features={n_features} pos_w={pos_class_weight:.2f} '
          f'max_epochs={max_epochs} swa_start={swa_start_frac} swa_lr={swa_lr}', flush=True)

    dm = FireSpreadDataModule(
        data_dir=DATA_DIR, batch_size=64, n_leading_observations=n_obs,
        n_leading_observations_test_adjustment=n_obs, crop_side_length=128,
        load_from_hdf5=True, num_workers=4, remove_duplicate_features=True,
        features_to_keep=NO_PDSI_KEEP, data_fold_id=fold, return_doy=False,
    )

    model = SMPModel(
        encoder_name='resnet18', n_channels=n_features,
        flatten_temporal_dimension=True, pos_class_weight=pos_class_weight,
        loss_function='Focal',
    )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
        return {'optimizer': opt, 'lr_scheduler': sch}

    model.configure_optimizers = types.MethodType(configure_optimizers, model)

    swa = StochasticWeightAveraging(
        swa_lrs=swa_lr, swa_epoch_start=swa_start_frac,
        annealing_epochs=5, annealing_strategy='cos',
    )
    ckpt = ModelCheckpoint(monitor='val_AP', mode='max', save_top_k=1,
                           save_last=True, filename='best-{epoch}-{val_AP:.4f}')

    trainer = pl.Trainer(
        accelerator='gpu', devices=1, precision='32-true',
        max_epochs=max_epochs, gradient_clip_val=1.0,
        deterministic='warn', logger=False,
        callbacks=[swa, ckpt],
        enable_progress_bar=False,
    )

    trainer.fit(model, dm)
    results = trainer.test(model, dm)  # in-memory model = SWA-averaged + BN-updated
    test_ap = float(results[0]['test_AP'])
    print(f'[swa] RESULT fold={fold} SWA_test_AP={test_ap:.4f}', flush=True)


if __name__ == '__main__':
    main()
