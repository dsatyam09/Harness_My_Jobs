"""CI regression gate: fail if any model/harness gets worse than the stored baseline.

    python -m jobharness.gate --summary results/summary.csv --baseline configs/gate_baseline.json
    python -m jobharness.gate --summary results/summary.csv --baseline configs/gate_baseline.json --update
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tolerance", type=float, default=0.05, help="allowed drop in success rate")
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    s = pd.read_csv(args.summary)
    s = s[s["family"] == "ALL"]
    current = {f"{r.model}|{r.harness}": float(r.success) for r in s.itertuples()}
    path = Path(args.baseline)
    if args.update or not path.exists():
        path.write_text(json.dumps(current, indent=2))
        print(f"baseline written: {path}")
        return
    base = json.loads(path.read_text())
    failures = []
    for key, old in base.items():
        new = current.get(key)
        if new is None:
            continue
        status = "OK"
        if new < old - args.tolerance:
            status = "REGRESSION"
            failures.append(key)
        print(f"{status:10s} {key:45s} baseline={old:.2f} now={new:.2f}")
    if failures:
        print(f"\nGate failed: {len(failures)} model/harness pair(s) regressed by more than "
              f"{args.tolerance:.0%}.")
        sys.exit(1)
    print("\nGate passed.")


if __name__ == "__main__":
    main()
