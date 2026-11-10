"""Run experiments.

Example (pilot, two small models, two harnesses, dev set, one trial):
    python -m jobharness.run --tasks data/tasks_dev.jsonl \
        --model gpt-oss-20b qwen3-30b-a3b \
        --harness configs/harness/h0.yaml configs/harness/h0_sensors.yaml \
        --trials 1 --concurrency 8 --out runs/pilot.jsonl

Re-running the same command resumes: finished (model, harness, task, trial) runs are skipped.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from .env import JobBoard
from .graders import grade
from .harness import HarnessConfig, run_episode
from .llm import OpenAIChat


def load_tasks(path, families=None, limit_per_family=None):
    tasks = [json.loads(l) for l in open(path) if l.strip()]
    if families:
        tasks = [t for t in tasks if t["family"] in families]
    if limit_per_family:
        out, seen = [], {}
        for t in tasks:
            seen[t["family"]] = seen.get(t["family"], 0) + 1
            if seen[t["family"]] <= limit_per_family:
                out.append(t)
        tasks = out
    return tasks


def load_board(data_dir):
    data = Path(data_dir)
    df = pd.read_parquet(data / "postings.parquet")
    profiles = json.loads((data / "profiles.json").read_text())
    return JobBoard(df, profiles)


def make_llm(models_cfg: dict, name: str):
    if name.startswith("mock-"):
        from .mock import mock_llm
        return mock_llm(name)
    c = models_cfg[name]
    return OpenAIChat(base_url=c["base_url"], model=c["model"], api_key=c.get("api_key", "EMPTY"),
                      temperature=c.get("temperature", 0.7), max_tokens=c.get("max_tokens", 1024),
                      extra_body=c.get("extra_body"))


def done_keys(out: Path) -> set:
    keys = set()
    if out.exists():
        for line in open(out):
            try:
                r = json.loads(line)
                keys.add((r["model"], r["harness"], r["task_id"], r["trial"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def run_one(llm, board, task, cfg, model_name, trial):
    if hasattr(llm, "bind"):  # mock models get a fresh per-task instance
        llm = llm.bind(task, board)
    try:
        res = run_episode(llm, board, task, cfg)
    except Exception as e:  # infrastructure error: recorded, counted as failure
        res = {"answer": None, "metrics": {"finish_reason": f"error: {type(e).__name__}: {e}"[:300]},
               "trajectory": [], "harness": {}}
    profile = board.profiles.get(task.get("profile_id")) if task.get("profile_id") else None
    g = grade(task, res["answer"], profile)
    return {"model": model_name, "harness": cfg.name, "task_id": task["task_id"],
            "family": task["family"], "trial": trial, "grade": g, "answer": res["answer"],
            "metrics": res["metrics"], "trajectory": res["trajectory"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--harness", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--limit-per-family", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    board = load_board(args.data)
    tasks = load_tasks(args.tasks, args.families, args.limit_per_family)
    models_cfg = yaml.safe_load(open(args.models)) if Path(args.models).exists() else {}
    cfgs = [HarnessConfig.load(p) for p in args.harness]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(out)

    jobs = []
    for m in args.model:
        llm = make_llm(models_cfg, m)
        for cfg in cfgs:
            for t in tasks:
                for k in range(args.trials):
                    if (m, cfg.name, t["task_id"], k) not in skip:
                        jobs.append((llm, t, cfg, m, k))
    print(f"{len(jobs)} runs to do ({len(skip)} already done)")
    lock = threading.Lock()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex, open(out, "a") as f:
        futs = [ex.submit(run_one, llm, board, t, cfg, m, k) for llm, t, cfg, m, k in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            n_ok += rec["grade"]["success"]
            with lock:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done, success so far {n_ok}/{i}")


if __name__ == "__main__":
    main()
