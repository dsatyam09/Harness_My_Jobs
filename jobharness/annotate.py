"""Grader validation by human annotation.

1) Export blind sheets (each annotator fills in their own copy):
    python -m jobharness.annotate export-fit --tasks data/tasks_dev.jsonl --n 50
    python -m jobharness.annotate export-fab --runs runs/pilot.jsonl --n 50
2) Compute agreement:
    python -m jobharness.annotate kappa \
        --fit annotations/fit_A.csv annotations/fit_B.csv annotations/fit_key.csv \
        --fab annotations/fab_A.csv annotations/fab_B.csv annotations/fab_key.csv
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .latex import write_table
from .profiles import profile_card
from .run import load_board, load_tasks

FIT_GUIDE = ("0 = violates a hard requirement (on-site in another state and the candidate will "
             "not relocate; remote but candidate not open to remote; stated max salary below the "
             "candidate's minimum; no sponsorship and candidate needs it). Otherwise rate the share "
             "of the job's concrete required skills the candidate has: 1 = under a quarter, "
             "2 = a quarter to a half, 3 = half or more.")
FAB_GUIDE = ("yes = at least one bullet claims a skill, tool, employer, degree or number of years "
             "that the profile does not support; no = every claim is supported.")


def cohen_kappa(a, b, weights=None):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k < 2:
        return float("nan")
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    if weights == "quadratic":
        W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    else:
        W = 1 - np.eye(k)
    return float(1 - (W * O).sum() / (W * E).sum())


def export_fit(args):
    board = load_board(args.data)
    tasks = [t for t in load_tasks(args.tasks) if t["family"] == "ranking"]
    pairs = [(t, j) for t in tasks for j in t["job_ids"]]
    random.Random(args.seed).shuffle(pairs)
    rows, key = [], []
    for i, (t, jid) in enumerate(pairs[: args.n]):
        p, j = board.profiles[t["profile_id"]], board.get_job(jid)
        rows.append({"item_id": f"fit{i:03d}", "profile": profile_card(p), "job_title": j["title"],
                     "job_location": j["location"], "job_description": j["description"][:2500],
                     "label_0to3": "", "notes": ""})
        key.append({"item_id": f"fit{i:03d}", "auto_label": t["answer"]["labels"][jid]})
    _write(args.outdir, "fit", rows, key, FIT_GUIDE)


def export_fab(args):
    board = load_board(args.data)
    recs = [json.loads(l) for l in open(args.runs) if l.strip()]
    recs = [r for r in recs if r["family"] == "tailoring" and isinstance(r.get("answer"), dict)
            and r["answer"].get("bullets")]
    random.Random(args.seed).shuffle(recs)
    tasks = {t["task_id"]: t for t in load_tasks(args.tasks)} if args.tasks else {}
    rows, key = [], []
    for i, r in enumerate(recs[: args.n]):
        pid = tasks.get(r["task_id"], {}).get("profile_id") or r["task_id"]
        p = board.profiles.get(pid)
        rows.append({"item_id": f"fab{i:03d}", "profile": profile_card(p) if p else pid,
                     "bullets": "\n".join(map(str, r["answer"]["bullets"])),
                     "fabricated_yes_no": "", "notes": ""})
        key.append({"item_id": f"fab{i:03d}", "auto_label": "yes" if r["grade"].get("n_fabricated") else "no",
                    "model": r["model"], "harness": r["harness"]})
    _write(args.outdir, "fab", rows, key, FAB_GUIDE)


def _write(outdir, kind, rows, key, guide):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for who in ["A", "B"]:
        pd.DataFrame(rows).to_csv(out / f"{kind}_{who}.csv", index=False)
    pd.DataFrame(key).to_csv(out / f"{kind}_key.csv", index=False)
    (out / f"{kind}_GUIDELINES.txt").write_text(guide + "\n")
    print(f"wrote {len(rows)} items to {out}/{kind}_A.csv and {kind}_B.csv (key kept separately)")


def kappa(args):
    rows = []
    for kind, files, col, w in [("Fit label (0--3)", args.fit, "label_0to3", "quadratic"),
                                ("Fabrication (yes/no)", args.fab, "fabricated_yes_no", None)]:
        if not files:
            continue
        A, B, K = (pd.read_csv(f) for f in files)
        m = A[["item_id", col]].merge(B[["item_id", col]], on="item_id", suffixes=("_a", "_b")) \
                               .merge(K[["item_id", "auto_label"]], on="item_id")
        m = m.dropna(subset=[f"{col}_a", f"{col}_b"])
        norm = (lambda s: s.astype(int)) if w else (lambda s: s.astype(str).str.strip().str.lower())
        a, b, auto = norm(m[f"{col}_a"]), norm(m[f"{col}_b"]), norm(m["auto_label"])
        rows.append([kind, len(m), f"{cohen_kappa(a, b, w):.2f}", f"{cohen_kappa(a, auto, w):.2f}",
                     f"{cohen_kappa(b, auto, w):.2f}", f"{100 * (a == b).mean():.0f}"])
    tex = write_table(["Judgment", "Items", "$\\kappa$ A--B", "$\\kappa$ A--auto", "$\\kappa$ B--auto",
                       "Raw A--B agreement (\\%)"], rows, "l r r r r r",
                      note="Cohen's kappa; quadratic weights for the ordinal fit label.")
    Path(args.tables).mkdir(parents=True, exist_ok=True)
    (Path(args.tables) / "kappa.tex").write_text(tex)
    print(tex)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["export-fit", "export-fab"]:
        s = sub.add_parser(name)
        s.add_argument("--data", default="data")
        s.add_argument("--tasks", default="data/tasks_dev.jsonl")
        s.add_argument("--runs")
        s.add_argument("--n", type=int, default=50)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--outdir", default="annotations")
    k = sub.add_parser("kappa")
    k.add_argument("--fit", nargs=3)
    k.add_argument("--fab", nargs=3)
    k.add_argument("--tables", default="results/tables")
    args = ap.parse_args()
    {"export-fit": export_fit, "export-fab": export_fab, "kappa": kappa}[args.cmd](args)


if __name__ == "__main__":
    main()
