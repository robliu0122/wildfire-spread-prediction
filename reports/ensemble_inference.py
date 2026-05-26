"""Phase 2C: per-test-year ensemble of phase2b fold checkpoints."""
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
OUT_PATH = PROJECT_ROOT / 'reports' / 'phase2c_ensemble.json'

FOLD_TO_TEST_YEAR = {
    0: 2021, 1: 2020, 2: 2021, 3: 2019, 4: 2020, 5: 2019,
    6: 2021, 7: 2018, 8: 2020, 9: 2018, 10: 2019, 11: 2018,
}
NO_PDSI_KEEP = [i for i in range(43) if i != 15]


def find_best_ckpt(fold_id: int, prefix: str = 'phase2b') -> Path:
    log_path = LOG_DIR / f'{prefix}_fold{fold_id}.log'
    text = log_path.read_text(errors='ignore')
    m = re.search(r'lightning_logs/wandb/(?:offline-)?run-[0-9_]+-([a-z0-9]+)', text)
    run_id = m.group(1)
    ckpt_dir = CKPT_ROOT / run_id / 'checkpoints'
    ckpts = sorted(ckpt_dir.glob('best-*.ckpt'))
    ckpts.sort(key=lambda p: float(re.search(r'val_AP=([0-9]+\.[0-9]+)', p.name).group(1)))
    return ckpts[-1]


def build_datamodule(fold_id: int) -> FireSpreadDataModule:
    return FireSpreadDataModule(
        data_dir=DATA_DIR,
        batch_size=1,
        n_leading_observations=5,
        n_leading_observations_test_adjustment=5,
        crop_side_length=128,
        load_from_hdf5=True,
        num_workers=2,
        remove_duplicate_features=True,
        features_to_keep=NO_PDSI_KEEP,
        data_fold_id=fold_id,
        return_doy=False,
    )


def main() -> None:
    year_to_folds: dict[int, list[int]] = {}
    for fid, yr in FOLD_TO_TEST_YEAR.items():
        year_to_folds.setdefault(yr, []).append(fid)


    # Resume from existing JSON if present
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            results.pop('_summary', None)
            print(f'Resuming from existing JSON, found years: {list(results.keys())}')
        except Exception:
            results = {}
    else:
        results = {}
    for year, fold_list in sorted(year_to_folds.items()):
        if str(year) in results:
            print(f'=== test year {year} already done, skipping ===', flush=True)
            continue
        print(f'=== test year {year} (folds {fold_list}) ===', flush=True)

        # Load all 3 models for this year
        models = []
        for fid in fold_list:
            ckpt = find_best_ckpt(fid)
            print(f'  loading fold {fid}: {ckpt.name}', flush=True)
            m = SMPModel.load_from_checkpoint(str(ckpt), map_location='cuda')
            m.eval().cuda()
            models.append((fid, m))

        # Build datamodule using fold_list[0] (test set is identical across the 3 folds since they share test year)
        dm = build_datamodule(fold_list[0])
        dm.setup('test')

        individual_aps = {fid: torchmetrics.AveragePrecision('binary') for fid, _ in models}
        ensemble_ap = torchmetrics.AveragePrecision('binary')

        n_fires = 0
        for batch in dm.test_dataloader():
            x, y = batch
            x = x.cuda()
            fire_probs = []
            target = None
            for fid, m in models:
                with torch.no_grad():
                    y_hat, y_ret = m.get_pred_and_gt((x, y))
                probs = torch.sigmoid(y_hat).cpu()
                fire_probs.append(probs)
                if target is None:
                    target = y_ret.long().flatten()
                individual_aps[fid].update(probs.flatten(), target)
            ens_probs = torch.stack(fire_probs).mean(dim=0)
            ensemble_ap.update(ens_probs.flatten(), target)
            n_fires += 1
            if n_fires % 20 == 0:
                print(f'    processed {n_fires} fires', flush=True)

        ind_vals = {str(fid): float(metric.compute().item()) for fid, metric in individual_aps.items()}
        ens_val = float(ensemble_ap.compute().item())
        print(f'  individual APs: {ind_vals}', flush=True)
        print(f'  ENSEMBLE AP for {year}: {ens_val:.4f}', flush=True)

        results[str(year)] = {
            'folds': fold_list,
            'n_fires': n_fires,
            'individual_AP': ind_vals,
            'ensemble_AP': ens_val,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True); OUT_PATH.write_text(json.dumps(results, indent=2))  # save after each year

        # Free GPU memory before next year
        for _, m in models:
            del m
        torch.cuda.empty_cache()

    mean_ens = sum(r['ensemble_AP'] for r in results.values()) / len(results)
    results['_summary'] = {
        'mean_ensemble_AP_across_years': mean_ens,
        'note': 'Mean is across 4 test years.',
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))  # written after each year
    print(f'\nSaved to {OUT_PATH}')
    print(f'Mean ensemble AP across 4 test years: {mean_ens:.4f}')


if __name__ == '__main__':
    main()
