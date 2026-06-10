"""Driver: run baseline + FNO across chosen folds, sequentially, resumable.

Skips any (model, fold) already present in results.csv so it can be re-run
after an interruption. Calls run_fold.py as a subprocess per combo so a crash
in one run never kills the whole sweep.

Edit FOLDS / MODELS / MAX_EPOCHS below to control scope.
"""
import os, sys, csv, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results.csv')
RUN_FOLD = os.path.join(HERE, 'run_fold.py')

FOLDS = [5, 0, 3, 6]                       # 2019 hard: 5,3 | 2021 easy: 0,6
MODELS = ['baseline', 'fno_bottleneck', 'fno_16x16']  # resumable: skips done pairs
MAX_EPOCHS = 50
BATCH = 32
WORKERS = 2


def done_set():
    if not os.path.exists(RESULTS):
        return set()
    with open(RESULTS) as f:
        return {(r['model'], int(r['fold'])) for r in csv.DictReader(f)}


def main():
    combos = [(m, fo) for fo in FOLDS for m in MODELS]
    done = done_set()
    todo = [c for c in combos if c not in done]
    print(f"{len(done)} done, {len(todo)} to run: {todo}", flush=True)
    for i, (model, fold) in enumerate(todo, 1):
        print(f"\n===== [{i}/{len(todo)}] {model} fold {fold} =====", flush=True)
        cmd = [sys.executable, RUN_FOLD, '--fold', str(fold), '--model', model,
               '--max_epochs', str(MAX_EPOCHS), '--batch_size', str(BATCH),
               '--num_workers', str(WORKERS)]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! {model} fold {fold} exited rc={rc}; continuing", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == '__main__':
    main()
