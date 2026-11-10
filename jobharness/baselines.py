"""Non-LLM reference points.

    python -m jobharness.baselines --tasks data/tasks_dev.jsonl --out runs/baselines.jsonl

Some baselines share rules with the answer key and are marked "label-aligned":
they show how far simple patterns go, not a fair competitor.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics

from . import vocab
from .graders import grade
from .run import load_board, load_tasks

LABEL_ALIGNED = {"baseline:regex", "baseline:skill-overlap"}


def _kw(instr):
    m = re.search(r'title contains "([^"]+)"', instr)
    return m.group(1) if m else ""


def bm25_top5(task, board):
    q = _kw(task["instruction"]) + (" remote" if "remote work" in task["instruction"] else "")
    return {"job_ids": [c["job_id"] for c in board.search(q, 5)]}


def title_only(task, board):
    return {"job_ids": [c["job_id"] for c in board.search_by_title(_kw(task["instruction"]), 50)]}


def regex_extract(task, board):
    j = board.get_job(task["job_ids"][0])
    amts = vocab.money_amounts(j["description"])
    text = j["description"].lower()
    period = "HOURLY" if re.search(r"per hour|/hr|hourly|an hour", text) else "YEARLY"
    return {"min_salary": min(amts) if amts else None, "max_salary": max(amts) if amts else None,
            "pay_period": period, "remote_allowed": vocab.remote_in_text(j["description"])}


def random_rank(task, board, seed=0):
    ids = list(task["job_ids"])
    random.Random(hash(task["task_id"]) + seed).shuffle(ids)
    return {"ranking": ids}


def overlap_rank(task, board):
    p = board.profiles[task["profile_id"]]
    def score(jid):
        s = vocab.skills_in_text(board.get_job(jid)["description"])
        return len(s & set(p["skills"]))
    return {"ranking": sorted(task["job_ids"], key=lambda j: -score(j))}


def majority_application(task, board):
    sal = statistics.median(p["min_salary"] for p in board.profiles.values())
    return {"needs_sponsorship": False, "years_with_skill": 0, "open_to_remote": True,
            "minimum_salary": int(sal), "highest_degree": "Bachelor's degree"}


def copy_profile(task, board):
    p = board.profiles[task["profile_id"]]
    hist = (p["work_history"] * 3)[:3]
    return {"bullets": [f"{h['title']} at {h['company']}, working with {', '.join(h['skills'])}"
                        for h in hist]}


BASELINES = {
    "search": {"baseline:bm25-top5": bm25_top5, "baseline:title-only": title_only},
    "extraction": {"baseline:regex": regex_extract},
    "ranking": {"baseline:random": random_rank, "baseline:skill-overlap": overlap_rank},
    "application": {"baseline:majority": majority_application},
    "tailoring": {"baseline:copy-profile": copy_profile},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    board = load_board(args.data)
    tasks = load_tasks(args.tasks)
    n = 0
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for t in tasks:
            for name, fn in BASELINES[t["family"]].items():
                ans = fn(t, board)
                prof = board.profiles.get(t.get("profile_id")) if t.get("profile_id") else None
                rec = {"model": name, "harness": "none", "task_id": t["task_id"],
                       "family": t["family"], "trial": 0, "grade": grade(t, ans, prof),
                       "answer": ans, "metrics": {"finish_reason": "baseline"}, "trajectory": []}
                f.write(json.dumps(rec, default=str) + "\n")
                n += 1
    print(f"wrote {n} baseline records to {args.out}")


if __name__ == "__main__":
    main()
