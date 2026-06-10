"""
Resumable selective downloader for WildfireSpreadTS (Zenodo 8006177).

Downloads a reproducible ~20%-per-year stratified subset of fire folders from the
48 GB WildfireSpreadTS.zip WITHOUT downloading the whole zip, using HTTP range
requests (remotezip). Re-runnable: already-downloaded tifs of correct size are skipped.

Output layout (matches what CreateHDF5Dataset.py expects):
    <OUT>/2018/fire_xxxx/YYYY-MM-DD.tif
    ...

Manifest of selected fires is written to subset_manifest.json (reproducible, seed=0).
Progress is appended to subset_download_progress.txt so a killed session can be resumed.
"""
import os, sys, json, time, random, collections, threading
from concurrent.futures import ThreadPoolExecutor

URL = 'https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_raw')
MANIFEST = os.path.join(os.path.dirname(__file__), 'subset_manifest.json')
PROGRESS = os.path.join(os.path.dirname(__file__), 'subset_download_progress.txt')
FRACTION = 0.20
SEED = 0
N_WORKERS = 3           # parallel range-download workers (stays under 133 req/min)
PACE_SEC = 0.15         # small per-worker pace
RETRY_SLEEP = 65        # on rate-limit / transient error

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(PROGRESS, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def build_manifest(rz):
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    fires = collections.defaultdict(list)        # (year, fire) -> [member names]
    sizes = collections.defaultdict(int)
    for i in rz.infolist():
        p = i.filename.split('/')
        if len(p) >= 3 and p[2].endswith('.tif'):
            fires[(p[0], p[1])].append(i.filename)
            sizes[(p[0], p[1])] += i.file_size
    by_year = collections.defaultdict(list)
    for (y, fire) in fires:
        by_year[y].append(fire)
    rng = random.Random(SEED)
    selected = {}
    for y in sorted(by_year):
        flist = sorted(by_year[y])
        rng.shuffle(flist)
        k = max(1, round(len(flist) * FRACTION))
        chosen = sorted(flist[:k])
        selected[y] = chosen
    manifest = {'fraction': FRACTION, 'seed': SEED,
                'selected': selected,
                'members': {f'{y}/{fire}': sorted(fires[(y, fire)])
                            for y in selected for fire in selected[y]}}
    with open(MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=1)
    return manifest

def main():
    from remotezip import RemoteZip
    os.makedirs(OUT, exist_ok=True)
    log(f'Opening remote zip central directory...')
    rz = RemoteZip(URL)
    manifest = build_manifest(rz)
    sizes = {name: rz.getinfo(name).file_size for fire in manifest['members'].values() for name in fire}
    all_members = list(sizes)
    n_fires = sum(len(v) for v in manifest['selected'].values())
    log(f'Subset: {n_fires} fires, {len(all_members)} tifs across years '
        + ', '.join(f'{y}={len(v)}' for y, v in manifest["selected"].items()))

    counter = {'done': 0, 'n': 0}
    lock = threading.Lock()
    local = threading.local()

    def get_rz():
        if not hasattr(local, 'rz'):
            from remotezip import RemoteZip
            local.rz = RemoteZip(URL)
        return local.rz

    def fetch(name):
        target = os.path.join(OUT, *name.split('/'))
        if os.path.exists(target) and os.path.getsize(target) == sizes[name]:
            with lock:
                counter['done'] += 1; counter['n'] += 1
            return
        os.makedirs(os.path.dirname(target), exist_ok=True)
        for attempt in range(6):
            try:
                z = get_rz()
                with z.open(name) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                with lock:
                    counter['done'] += 1
                break
            except Exception as e:
                log(f'retry {attempt} on {name}: {type(e).__name__} {e}')
                time.sleep(RETRY_SLEEP)
        else:
            log(f'FAILED permanently: {name}')
        with lock:
            counter['n'] += 1
            if counter['n'] % 100 == 0:
                log(f"progress {counter['n']}/{len(all_members)} ({counter['done']} ok)")
        time.sleep(PACE_SEC)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        list(ex.map(fetch, all_members))
    log(f"DONE. {counter['done']}/{len(all_members)} tifs present in {OUT}")

if __name__ == '__main__':
    main()
