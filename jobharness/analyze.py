"""Turn run logs into report tables.

    python -m jobharness.analyze runs/pilot.jsonl runs/baselines.jsonl \
        --compare h0 h0_sensors --out results

Optional transfer analysis (once H_L and H_S exist):
    --large gpt-oss-120b --h0 h0 --hl hL --hs hS
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .latex import esc, write_table
from .tasks import FAMILIES

SCORE_NAME = {"search": "F1", "extraction": "Field acc.", "ranking": "nDCG@5",
              "application": "Field acc.", "tailoring": "Coverage"}


def load(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("metrics") or {}
            g = r.get("grade") or {}
            rows.append({
                "model": r["model"], "harness": r["harness"], "task_id": r["task_id"],
                "family": r["family"], "trial": r["trial"], "success": float(bool(g.get("success"))),
                "score": float(g.get("score", 0.0) or 0.0),
                "n_fabricated": g.get("n_fabricated"),
                "finish": m.get("finish_reason"), "steps": m.get("steps"), "turns": m.get("turns"),
                "malformed": m.get("malformed"), "invalid": m.get("invalid_answers"),
                "loops": m.get("loops"), "overflow": m.get("overflow"),
                "tokens": (m.get("prompt_tokens") or 0) + (m.get("completion_tokens") or 0),
                "seconds": m.get("seconds"), "verify_rounds": m.get("verify_rounds"),
            })
    return pd.DataFrame(rows)


def per_task(df: pd.DataFrame, col="success") -> pd.DataFrame:
    """Mean over trials for each (model, harness, task)."""
    return df.groupby(["model", "harness", "family", "task_id"], as_index=False)[col].mean()


def boot_ci(values, n=1000, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def label(model, harness):
    return esc(model) if harness == "none" else f"{esc(model)} / {esc(harness)}"


def success_table(df):
    pt = per_task(df)
    rows, csv = [], []
    for (m, h), g in pt.groupby(["model", "harness"], sort=False):
        cells = []
        for fam in FAMILIES:
            v = g[g["family"] == fam]["success"]
            cells.append(f"{100 * v.mean():.0f}" if len(v) else "--")
            if len(v):
                mu, lo, hi = boot_ci(v)
                csv.append({"model": m, "harness": h, "family": fam, "n_tasks": len(v),
                            "success": mu, "ci_lo": lo, "ci_hi": hi})
        if h == "none":
            overall = "--"
        else:
            mu, lo, hi = boot_ci(g["success"])
            overall = f"{100 * mu:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"
            csv.append({"model": m, "harness": h, "family": "ALL", "n_tasks": len(g),
                        "success": mu, "ci_lo": lo, "ci_hi": hi})
        rows.append([label(m, h)] + cells + [overall])
    tex = write_table(["Model / harness"] + [f.capitalize() for f in FAMILIES] + ["Overall [95\\% CI]"],
                      rows, "l" + "r" * len(FAMILIES) + "r",
                      note="Task success rate (\\%), averaged over trials; CIs bootstrap over tasks "
                           "(1,000 resamples). Baselines apply only to some families.")
    return tex, pd.DataFrame(csv)


def score_table(df):
    pt = per_task(df, "score")
    rows = []
    for (m, h), g in pt.groupby(["model", "harness"], sort=False):
        rows.append([label(m, h)] + [f"{g[g['family'] == f]['score'].mean():.2f}"
                                     if (g["family"] == f).any() else "--" for f in FAMILIES])
    heads = ["Model / harness"] + [f"{f.capitalize()} ({SCORE_NAME[f]})" for f in FAMILIES]
    return write_table(heads, rows, "l" + "r" * len(FAMILIES),
                       note="Continuous scores averaged over tasks and trials.")


def process_table(df):
    d = df[df["harness"] != "none"]
    rows = []
    for (m, h), g in d.groupby(["model", "harness"], sort=False):
        rows.append([label(m, h), len(g), f"{100 * (g['finish'] == 'submitted').mean():.0f}",
                     f"{g['malformed'].fillna(0).mean():.2f}", f"{g['invalid'].fillna(0).mean():.2f}",
                     f"{g['loops'].fillna(0).mean():.2f}", f"{100 * g['overflow'].fillna(False).astype(bool).mean():.0f}",
                     f"{g['steps'].fillna(0).mean():.1f}", f"{g['tokens'].mean() / 1000:.1f}",
                     f"{g['seconds'].fillna(0).mean():.0f}"])
    return write_table(["Model / harness", "Runs", "Submitted (\\%)", "Malformed / run",
                        "Invalid ans. / run", "Loops / run", "Overflow (\\%)", "Steps",
                        "Tokens (K)", "Seconds"], rows, "l" + "r" * 9)


def finish_table(df):
    d = df[df["harness"] != "none"]
    reasons = sorted(d["finish"].fillna("none").map(lambda s: s.split(":")[0]).unique())
    rows = []
    for (m, h), g in d.groupby(["model", "harness"], sort=False):
        fr = g["finish"].fillna("none").map(lambda s: s.split(":")[0])
        rows.append([label(m, h)] + [int((fr == r).sum()) for r in reasons])
    return write_table(["Model / harness"] + [esc(r) for r in reasons], rows, "l" + "r" * len(reasons),
                       note="How runs ended. Anything other than submitted is a harness-level failure.")


def mcnemar(a: dict, b: dict):
    common = sorted(set(a) & set(b))
    bb = sum(1 for t in common if a[t] >= 0.5 and b[t] < 0.5)
    cc = sum(1 for t in common if a[t] < 0.5 and b[t] >= 0.5)
    n = bb + cc
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(bb, cc) + 1)) / 2 ** n)
    return len(common), bb, cc, p


def compare_table(df, ha, hb):
    pt = per_task(df)
    rows = []
    for m in pt["model"].unique():
        a = pt[(pt.model == m) & (pt.harness == ha)].set_index("task_id")["success"].to_dict()
        b = pt[(pt.model == m) & (pt.harness == hb)].set_index("task_id")["success"].to_dict()
        if not a or not b:
            continue
        n, wins_a, wins_b, p = mcnemar(a, b)
        sa = np.mean([a[t] for t in set(a) & set(b)])
        sb = np.mean([b[t] for t in set(a) & set(b)])
        rows.append([esc(m), n, f"{100 * sa:.0f}", f"{100 * sb:.0f}", wins_a, wins_b, f"{p:.3f}"])
    return write_table(["Model", "Tasks", f"{esc(ha)} (\\%)", f"{esc(hb)} (\\%)",
                        f"Only {esc(ha)}", f"Only {esc(hb)}", "McNemar $p$"], rows, "l r r r r r r",
                       note="Paired comparison on the same tasks (exact McNemar test).")


def passk_table(df):
    d = df[df["harness"] != "none"]
    rows = []
    for (m, h), g in d.groupby(["model", "harness"], sort=False):
        k = g["trial"].nunique()
        if k < 2:
            continue
        by = g.groupby("task_id")["success"]
        rows.append([label(m, h), k, f"{100 * by.mean().mean():.0f}", f"{100 * (by.min() == 1).mean():.0f}"])
    if not rows:
        return None
    return write_table(["Model / harness", "$k$", "pass@1 (\\%)", "pass$^k$ (\\%)"], rows, "l r r r",
                       note="pass$^k$: share of tasks solved in every one of $k$ trials.")


def transfer_table(df, large, h0, hl, hs, n_boot=1000, seed=0):
    pt = per_task(df)
    S = {(m, h): g.set_index("task_id")["success"] for (m, h), g in pt.groupby(["model", "harness"])}
    rows = []
    for m in sorted(pt["model"].unique()):
        if m == large or m.startswith("baseline"):
            continue
        keys = [(m, hl), (m, h0), (large, hl), (large, h0)]
        if not all(k in S for k in keys):
            continue
        common = sorted(set.intersection(*[set(S[k].index) for k in keys]))
        arr = np.array([[S[k][t] for k in keys] for t in common])
        has_hs = (m, hs) in S
        hs_arr = np.array([S[(m, hs)].get(t, np.nan) for t in common]) if has_hs else None

        def stats(idx):
            a = arr[idx].mean(axis=0)
            den = a[2] - a[3]
            r = (a[0] - a[1]) / den if abs(den) > 1e-9 else np.nan
            c = np.nan
            if has_hs:
                hsm = np.nanmean(hs_arr[idx])
                den_c = a[2] - a[1]
                c = (hsm - a[1]) / den_c if abs(den_c) > 1e-9 else np.nan
            return r, c

        r0, c0 = stats(np.arange(len(common)))
        rng = np.random.default_rng(seed)
        bs = [stats(rng.integers(0, len(common), len(common))) for _ in range(n_boot)]
        rr = [b[0] for b in bs if b[0] == b[0]]
        cc = [b[1] for b in bs if b[1] == b[1]]
        fmt = lambda v, xs: "--" if v != v else f"{v:.2f} [{np.percentile(xs, 2.5):.2f}, {np.percentile(xs, 97.5):.2f}]"
        rows.append([esc(m), len(common), fmt(r0, rr), fmt(c0, cc) if has_hs else "--"])
    return write_table(["Small model", "Tasks", "Transfer ratio $R(m)$ [95\\% CI]",
                        "Gap closure $C(m)$ [95\\% CI]"], rows, "l r r r",
                       note=f"Large model: {esc(large)}. $R(m)$ uses {esc(hl)} vs. {esc(h0)}; "
                            f"$C(m)$ uses {esc(hs)}.")


def figure(df, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pt = per_task(df[df["harness"] != "none"])
    groups = list(pt.groupby(["model", "harness"], sort=False))
    if not groups:
        return
    x = np.arange(len(FAMILIES))
    w = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, ((m, h), g) in enumerate(groups):
        vals = [100 * g[g.family == f]["success"].mean() if (g.family == f).any() else 0 for f in FAMILIES]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=f"{m} / {h}")
    ax.set_xticks(x, [f.capitalize() for f in FAMILIES])
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)


def error_sample(paths, out_csv, n=40, seed=0):
    fails = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["harness"] == "none" or (r.get("grade") or {}).get("success"):
                continue
            last = (r.get("trajectory") or [{}])[-1]
            fails.append({"model": r["model"], "harness": r["harness"], "task_id": r["task_id"],
                          "family": r["family"], "trial": r["trial"],
                          "finish": (r.get("metrics") or {}).get("finish_reason"),
                          "answer": json.dumps(r.get("answer"))[:400],
                          "grade": json.dumps(r.get("grade"))[:400],
                          "last_turn": json.dumps(last)[:600], "error_category": "", "notes": ""})
    random.Random(seed).shuffle(fails)
    pd.DataFrame(fails[:n]).to_csv(out_csv, index=False)
    return min(n, len(fails))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare", nargs=2, metavar=("HARNESS_A", "HARNESS_B"))
    ap.add_argument("--large")
    ap.add_argument("--h0", default="h0")
    ap.add_argument("--hl", default="hL")
    ap.add_argument("--hs", default="hS")
    ap.add_argument("--error-sample", type=int, default=40)
    ap.add_argument("--include-mock", action="store_true", help="keep mock-* models (plumbing checks only)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = load(args.runs)
    if not args.include_mock:
        df = df[~df["model"].str.startswith("mock-")]

    tex, summary = success_table(df)
    (out / "tables" / "success.tex").write_text(tex)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "tables" / "scores.tex").write_text(score_table(df))
    if (df["harness"] != "none").any():
        (out / "tables" / "process.tex").write_text(process_table(df))
        (out / "tables" / "finish.tex").write_text(finish_table(df))
        figure(df, out / "figures" / "success_by_family.png")
        k = error_sample(args.runs, out / "error_sample.csv", args.error_sample)
        print(f"error sample: {k} failed runs -> {out / 'error_sample.csv'}")
    pk = passk_table(df)
    if pk:
        (out / "tables" / "passk.tex").write_text(pk)
    if args.compare:
        (out / "tables" / "compare.tex").write_text(compare_table(df, *args.compare))
    if args.large:
        (out / "tables" / "transfer.tex").write_text(
            transfer_table(df, args.large, args.h0, args.hl, args.hs))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
