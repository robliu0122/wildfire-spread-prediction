"""Parse per-fold logs and report 12-fold mean / per-year breakdown.

Usage (on remote):
    cd ~/ECE228-Project-WildfirePredict
    /root/autodl-tmp/wsts/venv/bin/python reports/analyze_phase2.py

Reads /root/autodl-tmp/wsts/logs/{phase1,phase2a,phase2b}_fold{0..11}.log,
extracts final test_AP per fold, groups by test-year using the WSTS+
12-fold leave-year-out scheme, prints a comparison table, and writes
reports/phase2_summary.json.
"""
import json
import re
from pathlib import Path

LOG_DIR = Path("/root/autodl-tmp/wsts/logs")
OUT_PATH = Path("/root/ECE228-Project-WildfirePredict/reports/phase2_summary.json")
PHASES = ["phase1", "phase2a", "phase2b"]

FOLD_TO_TEST_YEAR = {
    0: 2021, 1: 2020, 2: 2021, 3: 2019, 4: 2020, 5: 2019,
    6: 2021, 7: 2018, 8: 2020, 9: 2018, 10: 2019, 11: 2018,
}

TEST_AP_RE = re.compile(r"test_AP\s*[│|]\s*([0-9.]+)")


def parse_fold_log(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    matches = TEST_AP_RE.findall(text)
    return float(matches[-1]) if matches else None


def collect_phase(phase: str) -> dict[int, float | None]:
    return {fid: parse_fold_log(LOG_DIR / f"{phase}_fold{fid}.log") for fid in range(12)}


def per_year_means(results: dict[int, float | None]) -> dict[int, float | None]:
    by_year: dict[int, list[float]] = {}
    for fid, ap in results.items():
        if ap is None:
            continue
        by_year.setdefault(FOLD_TO_TEST_YEAR[fid], []).append(ap)
    return {yr: (sum(v) / len(v) if v else None) for yr, v in sorted(by_year.items())}


def fmt(x: float | None, width: int = 6) -> str:
    return f"{x:.{width-2}f}" if x is not None else "  --  "


def main() -> None:
    all_results: dict[str, dict[int, float | None]] = {p: collect_phase(p) for p in PHASES}
    summary: dict[str, dict] = {}

    print(f"{'fold':>5} | {'test_year':>9} | " + " | ".join(f"{p:>7}" for p in PHASES))
    print("-" * (5 + 11 + 11 * len(PHASES)))
    for fid in range(12):
        row = [fmt(all_results[p][fid]) for p in PHASES]
        print(f"{fid:>5} | {FOLD_TO_TEST_YEAR[fid]:>9} | " + " | ".join(f"{r:>7}" for r in row))

    print()
    print(f"{'year':>5} | " + " | ".join(f"{p:>7}" for p in PHASES))
    print("-" * (7 + 11 * len(PHASES)))
    for yr in sorted({FOLD_TO_TEST_YEAR[f] for f in range(12)}):
        row = [fmt(per_year_means(all_results[p]).get(yr)) for p in PHASES]
        print(f"{yr:>5} | " + " | ".join(f"{r:>7}" for r in row))

    print()
    print(f"{'mean':>5} | " + " | ".join(f"{p:>7}" for p in PHASES))
    print("-" * (7 + 11 * len(PHASES)))
    means = {}
    row_strs = []
    for phase in PHASES:
        vals = [ap for ap in all_results[phase].values() if ap is not None]
        m = sum(vals) / len(vals) if vals else None
        means[phase] = m
        row_strs.append(fmt(m))
    print(f"{'12fold':>5} | " + " | ".join(f"{r:>7}" for r in row_strs))

    print()
    print("=== PDSI hypothesis check (Phase 2B - Phase 2A on 2019 stratum) ===")
    p2a_2019 = per_year_means(all_results["phase2a"]).get(2019)
    p2b_2019 = per_year_means(all_results["phase2b"]).get(2019)
    if p2a_2019 is not None and p2b_2019 is not None:
        delta = p2b_2019 - p2a_2019
        print(f"  2019 Phase 2A: {p2a_2019:.4f}")
        print(f"  2019 Phase 2B: {p2b_2019:.4f}")
        print(f"  Delta: {delta:+.4f}  (hypothesis confirmed if >= +0.05)")
    else:
        print("  Phase 2A or 2B 2019 stratum not yet complete.")

    summary = {
        "per_fold": {p: {str(f): all_results[p][f] for f in range(12)} for p in PHASES},
        "per_year_mean": {p: {str(y): v for y, v in per_year_means(all_results[p]).items()} for p in PHASES},
        "twelve_fold_mean": {p: means[p] for p in PHASES},
        "completed_folds": {p: sum(1 for v in all_results[p].values() if v is not None) for p in PHASES},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
