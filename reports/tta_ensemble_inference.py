"""Phase 2C + TTA: per-test-year ensemble with flip-group test-time augmentation.

For each test year we ensemble the 3 phase2b fold checkpoints (same as Phase 2C).
On top of that, each model is evaluated with flip-group TTA:
    {identity, hflip, vflip, rot180}
Directional channels (wind dir / aspect / forecast wind dir, raw indices [7,13,19])
are corrected in DEGREES using the exact formulas from FireSpreadDataset.augment(),
BEFORE the sin/cos encoding -- so the spatial flip stays physically consistent.

We deliberately skip 90/270 rotations: test crops are non-square (multiple of 32),
which would swap H/W and introduce a CW/CCW convention ambiguity. The flip group
({id,h,v,180}) is lossless and unambiguous on non-square images.

Speed: each of the 4 orientations is encoded ONCE per fire, then the 4 orientations
are stacked into a single batch and run through each model in ONE forward
-> 3 forwards/fire instead of 12.

Self-validation: also computes the no-TTA ensemble AP (identity only). That number
must reproduce reports/phase2c_ensemble.json (mean 0.4696) -- if it does, the inline
encoding replication is correct and the TTA number is trustworthy.
"""
import json
import re
import sys
from pathlib import Path

import torch
import torchmetrics

sys.path.insert(0, 'src')
from dataloader.FireSpreadDataModule import FireSpreadDataModule  # noqa: E402
from models.SMPModel import SMPModel  # noqa: E402

PROJECT_ROOT = Path('/root/ECE228-Project-WildfirePredict')
CKPT_ROOT = PROJECT_ROOT / 'lightning_logs' / 'wildfire_progression'
LOG_DIR = Path('/root/autodl-tmp/wsts/logs')
DATA_DIR = '/root/autodl-tmp/wsts/data_hdf5'
OUT_PATH = PROJECT_ROOT / 'reports' / 'phase2c_tta_ensemble.json'

FOLD_TO_TEST_YEAR = {
    0: 2021, 1: 2020, 2: 2021, 3: 2019, 4: 2020, 5: 2019,
    6: 2021, 7: 2018, 8: 2020, 9: 2018, 10: 2019, 11: 2018,
}
NO_PDSI_KEEP = [i for i in range(43) if i != 15]
DEVICE = 'cuda'


def find_best_ckpt(fold_id: int, prefix: str = 'phase2b') -> Path:
    text = (LOG_DIR / f'{prefix}_fold{fold_id}.log').read_text(errors='ignore')
    run_id = re.search(r'lightning_logs/wandb/(?:offline-)?run-[0-9_]+-([a-z0-9]+)', text).group(1)
    ckpts = sorted((CKPT_ROOT / run_id / 'checkpoints').glob('best-*.ckpt'))
    ckpts.sort(key=lambda p: float(re.search(r'val_AP=([0-9]+\.[0-9]+)', p.name).group(1)))
    return ckpts[-1]


def build_dataset(fold_id: int):
    dm = FireSpreadDataModule(
        data_dir=DATA_DIR, batch_size=1, n_leading_observations=5,
        n_leading_observations_test_adjustment=5, crop_side_length=128,
        load_from_hdf5=True, num_workers=0, remove_duplicate_features=True,
        features_to_keep=NO_PDSI_KEEP, data_fold_id=fold_id, return_doy=False,
    )
    dm.setup('test')
    return dm.test_dataset


def encode(ds, x: torch.Tensor) -> torch.Tensor:
    """Faithful copy of preprocess_and_augment's encoding (non-augment hdf5 branch).
    x: (T, 23, H, W) center-cropped raw tensor, degree channels still in DEGREES.
    Returns flattened/deduplicated model input (C_flat, H, W)."""
    idx = ds.indices_of_degree_features  # [7, 13, 19]
    ang = torch.deg2rad(x[:, idx, ...])
    x[:, idx, ...] = torch.sin(ang)
    cos_feat = torch.cos(ang) / ds.stds[:, idx, ...]
    binary_af = (x[:, -1:, ...] > 0).float()
    x = (x - ds.means) / ds.stds
    x = torch.cat([x, cos_feat, binary_af], dim=1)
    x = torch.nan_to_num(x, nan=0.0)
    new_shape = (x.shape[0], x.shape[2], x.shape[3], ds.one_hot_matrix.shape[0])
    lc = x[:, 16, ...].long().flatten() - 1
    lc_enc = ds.one_hot_matrix[lc].reshape(new_shape).permute(0, 3, 1, 2)
    x = torch.cat([x[:, :16, ...], lc_enc, x[:, 17:, ...]], dim=1)
    x = ds.flatten_and_remove_duplicate_features_(x)
    return x


def in_h(x, idx):
    x = torch.flip(x, dims=[-1])
    x[:, idx, ...] = 360.0 - x[:, idx, ...]
    return x


def in_v(x, idx):
    x = torch.flip(x, dims=[-2])
    x[:, idx, ...] = (180.0 - x[:, idx, ...]) % 360.0
    return x


# ordered: index 0 is identity (used for no-TTA self-validation)
TRANSFORM_OPS = [[], ['h'], ['v'], ['h', 'v']]


def apply_input(x, ops, idx):
    for op in ops:
        x = in_h(x, idx) if op == 'h' else in_v(x, idx)
    return x


def invert_output(p, ops):
    for op in ops:
        p = torch.flip(p, dims=[-1]) if op == 'h' else torch.flip(p, dims=[-2])
    return p


@torch.no_grad()
def predict_fire(models, ds, x_c, idx):
    """Return (id_ensemble_prob, tta_ensemble_prob), both (H,W) cpu, averaged over models."""
    enc = [encode(ds, apply_input(x_c.clone(), ops, idx)) for ops in TRANSFORM_OPS]
    batch = torch.stack(enc, dim=0).to(DEVICE)  # (4, C, H, W)

    id_sum, tta_sum = None, None
    for _, m in models:
        try:
            logits = m(batch).squeeze(1)            # (4, H, W)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logits = torch.cat([m(batch[k:k + 1]).squeeze(1) for k in range(4)], dim=0)
        probs = torch.sigmoid(logits).cpu()         # (4, H, W)
        inv = [invert_output(probs[k], ops) for k, ops in enumerate(TRANSFORM_OPS)]
        id_p = inv[0]
        tta_p = torch.stack(inv, dim=0).mean(dim=0)
        id_sum = id_p if id_sum is None else id_sum + id_p
        tta_sum = tta_p if tta_sum is None else tta_sum + tta_p
    n = len(models)
    return id_sum / n, tta_sum / n


def main() -> None:
    year_to_folds: dict[int, list[int]] = {}
    for fid, yr in FOLD_TO_TEST_YEAR.items():
        year_to_folds.setdefault(yr, []).append(fid)

    results: dict[str, dict] = {}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            results.pop('_summary', None)
            print(f'Resuming, found years: {list(results.keys())}', flush=True)
        except Exception:
            results = {}

    for year, fold_list in sorted(year_to_folds.items()):
        if str(year) in results:
            print(f'=== year {year} already done, skip ===', flush=True)
            continue
        print(f'=== test year {year} (folds {fold_list}) ===', flush=True)

        models = []
        for fid in fold_list:
            ckpt = find_best_ckpt(fid)
            print(f'  loading fold {fid}: {ckpt.name}', flush=True)
            m = SMPModel.load_from_checkpoint(str(ckpt), map_location=DEVICE)
            m.eval().to(DEVICE)
            models.append((fid, m))

        ds = build_dataset(fold_list[0])
        idx = ds.indices_of_degree_features

        ap_notta = torchmetrics.AveragePrecision('binary')
        ap_tta = torchmetrics.AveragePrecision('binary')

        n = len(ds)
        for i in range(n):
            yr, nm, ifi = ds.find_image_index_from_dataset_index(i)
            x_raw, y_raw = ds.load_imgs(yr, nm, ifi)
            x = torch.Tensor(x_raw)
            y = (torch.Tensor(y_raw) > 0).long()
            x_c, y_c = ds.center_crop_x32(x, y)
            target = y_c.long().flatten()

            id_ens, tta_ens = predict_fire(models, ds, x_c, idx)
            ap_notta.update(id_ens.flatten(), target)
            ap_tta.update(tta_ens.flatten(), target)

            if (i + 1) % 50 == 0:
                print(f'    {i + 1}/{n} fires', flush=True)

        v_notta = float(ap_notta.compute().item())
        v_tta = float(ap_tta.compute().item())
        print(f'  year {year}: no-TTA={v_notta:.4f}  TTA={v_tta:.4f}  (delta {v_tta - v_notta:+.4f})', flush=True)

        results[str(year)] = {
            'folds': fold_list, 'n_fires': n,
            'ensemble_AP_no_tta': v_notta, 'ensemble_AP_tta': v_tta,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

        for _, m in models:
            del m
        torch.cuda.empty_cache()

    mean_notta = sum(r['ensemble_AP_no_tta'] for r in results.values()) / len(results)
    mean_tta = sum(r['ensemble_AP_tta'] for r in results.values()) / len(results)
    results['_summary'] = {
        'mean_no_tta': mean_notta, 'mean_tta': mean_tta, 'delta': mean_tta - mean_notta,
        'note': 'mean across 4 test years; no_tta must match phase2c_ensemble.json (0.4696).',
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f'\nMEAN no-TTA={mean_notta:.4f}  TTA={mean_tta:.4f}  delta={mean_tta - mean_notta:+.4f}', flush=True)
    print(f'Saved to {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
