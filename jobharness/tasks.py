"""Task generation.

Usage:
    python -m jobharness.tasks --data data --per-family 12 --dev-frac 0.25

Each task has a visible instruction and a hidden answer. Answers come from the
structured columns (kept hidden from the agent) or from synthetic profile facts.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from . import vocab
from .latex import write_table
from .profiles import make_profiles, years_with_skill

FAMILIES = ["search", "extraction", "ranking", "application", "tailoring"]


def fit_label(job: dict, profile: dict) -> int:
    """0 = violates a hard constraint; 1-3 = increasing skill overlap.

    Uses only information stated in the job text (remote, salary, sponsorship,
    skills) plus the visible location, so the label is recoverable by a reader.
    """
    remote = bool(job["remote_text"])
    if not remote and job["state"] != profile["state"] and not profile["open_to_relocate"]:
        return 0
    if remote and not profile["open_to_remote"] and job["state"] != profile["state"]:
        return 0
    ymax = job.get("yearly_max")
    if ymax == ymax and ymax is not None and ymax < profile["min_salary"]:
        return 0
    if job["sponsorship_unavailable"] and profile["needs_sponsorship"]:
        return 0
    jskills = set(job["skills"])
    if not jskills:
        return 1
    overlap = len(jskills & set(profile["skills"])) / len(jskills)
    return 3 if overlap >= 0.5 else 2 if overlap >= 0.25 else 1


def _search_tasks(df, rng, n):
    tasks = []
    for _ in range(n * 20):
        if len(tasks) >= n:
            break
        cname = rng.choice([c for c in vocab.CLUSTERS if (df["cluster"] == c).sum() >= 5] or [None])
        if cname is None:
            break
        kw = rng.choice(vocab.CLUSTERS[cname]["titles"])
        need_remote = rng.random() < 0.6
        pool = df[df["title"].str.lower().str.contains(kw, regex=False) & df["yearly_min"].notna()]
        if need_remote:
            pool = pool[pool["remote_text"]]
        if len(pool) < 2:
            continue
        mins = sorted(pool["yearly_min"].tolist(), reverse=True)
        k = rng.randint(2, min(8, len(mins)))
        floor = int(mins[k - 1] // 5000 * 5000)
        ans = pool[pool["yearly_min"] >= floor]["job_id"].tolist()
        if not 1 <= len(ans) <= 10:
            continue
        remote_txt = " that allow remote work," if need_remote else ""
        instr = (f'Find every job posting whose title contains "{kw}",{remote_txt} and whose stated '
                 f"minimum salary is at least ${floor:,} per year. Convert hourly pay at 2,080 hours "
                 f"per year and monthly pay at 12 months. Only include postings that state a salary. "
                 f'Submit {{"job_ids": [...]}}.')
        sig = (kw, need_remote, floor)
        if any(t["meta"]["sig"] == list(sig) for t in tasks):
            continue
        tasks.append({"family": "search", "instruction": instr, "profile_id": None,
                      "job_ids": [], "answer": {"job_ids": ans},
                      "meta": {"sig": list(sig), "n_answers": len(ans)}})
    return tasks


def _extraction_tasks(df, rng, n):
    pool = df[df["salary_match"]]
    tasks = []
    for _, j in pool.sample(n=min(n, len(pool)), random_state=rng.randint(0, 10**6)).iterrows():
        instr = (f"For job {j['job_id']}, report the minimum and maximum salary exactly as stated, the "
                 f"pay period (YEARLY, HOURLY, MONTHLY, WEEKLY or BIWEEKLY), and whether remote work is "
                 f'allowed. Submit {{"min_salary": number or null, "max_salary": number, '
                 f'"pay_period": string, "remote_allowed": true/false}}.')
        mn = None if j["min_salary"] != j["min_salary"] else float(j["min_salary"])
        tasks.append({"family": "extraction", "instruction": instr, "profile_id": None,
                      "job_ids": [j["job_id"]],
                      "answer": {"min_salary": mn, "max_salary": float(j["max_salary"]),
                                 "pay_period": j["pay_period"], "remote_allowed": bool(j["remote_text"])},
                      "meta": {}})
    return tasks


def _ranking_tasks(df, profiles, rng, n, list_size):
    tasks = []
    for _ in range(n * 30):
        if len(tasks) >= n:
            break
        p = rng.choice(profiles)
        pool = df[df["cluster"] == p["cluster"]]
        if len(pool) < list_size:
            continue
        jobs = pool.sample(n=list_size, random_state=rng.randint(0, 10**6)).to_dict("records")
        labels = {j["job_id"]: fit_label(j, p) for j in jobs}
        if len(set(labels.values())) < 3:
            continue
        ids = [j["job_id"] for j in jobs]
        rng.shuffle(ids)
        instr = (f"Rank these {list_size} job postings from best to worst fit for the candidate "
                 f"(profile {p['profile_id']}). Consider location and remote preferences, minimum "
                 f"salary, visa sponsorship, and how many of the job's required skills the candidate "
                 f'has. Jobs: {", ".join(ids)}. Submit {{"ranking": [all {list_size} job IDs, best first]}}.')
        tasks.append({"family": "ranking", "instruction": instr, "profile_id": p["profile_id"],
                      "job_ids": ids, "answer": {"labels": labels}, "meta": {}})
    return tasks


def _application_tasks(profiles, rng, n):
    tasks = []
    for p in rng.sample(profiles, k=min(n, len(profiles))):
        hist_skills = sorted({s for h in p["work_history"] for s in h["skills"]})
        skill = rng.choice(hist_skills or p["skills"])
        instr = (f"Fill in the screening questions for candidate {p['profile_id']} using only the "
                 f"profile: (1) Will you now or in the future require visa sponsorship? (2) How many "
                 f"years of professional experience do you have using {skill}? (3) Are you open to "
                 f"remote work? (4) What is your minimum expected annual salary in USD? (5) What is "
                 f'your highest degree? Submit {{"needs_sponsorship": bool, "years_with_skill": int, '
                 f'"open_to_remote": bool, "minimum_salary": int, "highest_degree": string}}.')
        tasks.append({"family": "application", "instruction": instr, "profile_id": p["profile_id"],
                      "job_ids": [],
                      "answer": {"needs_sponsorship": p["needs_sponsorship"],
                                 "years_with_skill": years_with_skill(p, skill),
                                 "open_to_remote": p["open_to_remote"],
                                 "minimum_salary": p["min_salary"],
                                 "highest_degree": p["highest_degree"]},
                      "meta": {"skill": skill}})
    return tasks


def _tailoring_tasks(df, profiles, rng, n):
    tasks = []
    for _ in range(n * 30):
        if len(tasks) >= n:
            break
        p = rng.choice(profiles)
        pool = df[(df["cluster"] == p["cluster"])]
        pool = pool[pool["skills"].map(lambda s: len(set(s) & set(p["skills"])) >= 2)]
        if pool.empty:
            continue
        j = pool.sample(n=1, random_state=rng.randint(0, 10**6)).iloc[0]
        if any(t["job_ids"] == [j["job_id"]] and t["profile_id"] == p["profile_id"] for t in tasks):
            continue
        target = sorted(set(j["skills"]) & set(p["skills"]))
        instr = (f"Write exactly three resume bullet points for candidate {p['profile_id']}, tailored to "
                 f"job {j['job_id']}. Use only facts from the candidate profile: do not add skills, "
                 f'tools, employers or years of experience the candidate does not have. Submit '
                 f'{{"bullets": ["...", "...", "..."]}}.')
        tasks.append({"family": "tailoring", "instruction": instr, "profile_id": p["profile_id"],
                      "job_ids": [j["job_id"]], "answer": {"target_skills": target},
                      "meta": {}})
    return tasks


def build(df: pd.DataFrame, profiles: list[dict], per_family: int, seed: int, list_size: int):
    rng = random.Random(seed)
    tasks = (_search_tasks(df, rng, per_family) + _extraction_tasks(df, rng, per_family)
             + _ranking_tasks(df, profiles, rng, per_family, list_size)
             + _application_tasks(profiles, rng, per_family)
             + _tailoring_tasks(df, profiles, rng, per_family))
    for i, t in enumerate(tasks):
        t["task_id"] = f"{t['family'][:3]}-{i:04d}"
    return tasks


def split(tasks, dev_frac, seed):
    rng = random.Random(seed)
    dev, test = [], []
    for fam in FAMILIES:
        ft = [t for t in tasks if t["family"] == fam]
        rng.shuffle(ft)
        k = max(1, round(len(ft) * dev_frac)) if ft else 0
        dev += ft[:k]
        test += ft[k:]
    return dev, test


def stats_latex(dev, test) -> str:
    rows = []
    for fam in FAMILIES:
        d = [t for t in dev if t["family"] == fam]
        te = [t for t in test if t["family"] == fam]
        extra = ""
        if fam == "search":
            sizes = [t["meta"]["n_answers"] for t in d + te]
            extra = f"answer-set size {min(sizes)}--{max(sizes)}" if sizes else ""
        rows.append([fam, len(d), len(te), extra])
    return write_table(["Family", "Dev", "Test", "Notes"], rows, "l r r l")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--per-family", type=int, default=48)
    ap.add_argument("--profiles-per-cluster", type=int, default=8)
    ap.add_argument("--list-size", type=int, default=10)
    ap.add_argument("--dev-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()

    data = Path(args.data)
    df = pd.read_parquet(data / "postings.parquet")
    profiles = make_profiles(df, args.profiles_per_cluster, args.seed)
    tasks = build(df, profiles, args.per_family, args.seed, args.list_size)
    dev, test = split(tasks, args.dev_frac, args.seed)
    (data / "profiles.json").write_text(json.dumps(profiles, indent=1))
    for name, ts in [("tasks_dev.jsonl", dev), ("tasks_test.jsonl", test)]:
        with open(data / name, "w") as f:
            for t in ts:
                f.write(json.dumps(t) + "\n")
    Path(args.tables).mkdir(parents=True, exist_ok=True)
    (Path(args.tables) / "tasks.tex").write_text(stats_latex(dev, test))
    print(f"profiles={len(profiles)} dev={len(dev)} test={len(test)}")
    for fam in FAMILIES:
        print(f"  {fam:12s} dev={sum(t['family'] == fam for t in dev):3d} "
              f"test={sum(t['family'] == fam for t in test):3d}")


if __name__ == "__main__":
    main()
