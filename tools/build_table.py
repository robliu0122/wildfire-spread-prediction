"""Read results.csv and emit the UNet-vs-FNO comparison table (markdown) + means.
Handles any subset of models among baseline / fno_bottleneck / fno_16x16."""
import os, csv, collections

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, 'results.csv'))))
by = {(r['model'], int(r['fold'])): r for r in rows}
folds = sorted({int(r['fold']) for r in rows})
YEAR = {int(r['fold']): r['test_year'] for r in rows}

MODELS = [m for m in ['baseline', 'fno_bottleneck', 'fno_16x16']
          if any(k[0] == m for k in by)]
LABEL = {'baseline': 'UNet', 'fno_bottleneck': 'FNO-bottleneck(4x4)',
         'fno_16x16': 'FNO-16x16'}

header = '| fold | test year | ' + ' | '.join(LABEL[m] for m in MODELS) + ' |'
sep = '|' + '---|' * (len(MODELS) + 2)
lines = [header, sep]
acc = {m: [] for m in MODELS}
per_year = collections.defaultdict(lambda: {m: [] for m in MODELS})
for fo in folds:
    cells = []
    for m in MODELS:
        r = by.get((m, fo))
        v = float(r['test_AP']) if r else float('nan')
        acc[m].append(v)
        per_year[YEAR[fo]][m].append(v)
        cells.append(f'{v:.3f}')
    lines.append(f'| {fo} | {YEAR[fo]} | ' + ' | '.join(cells) + ' |')

mean_cells = [f'**{sum(acc[m])/len(acc[m]):.3f}**' for m in MODELS]
lines.append('| **mean** | -- | ' + ' | '.join(mean_cells) + ' |')

print('\n'.join(lines))
print('\nPer-test-year mean AP:')
for y in sorted(per_year):
    parts = [f'{LABEL[m]}={sum(per_year[y][m])/len(per_year[y][m]):.3f}' for m in MODELS]
    print(f'  {y}: ' + '  '.join(parts))
