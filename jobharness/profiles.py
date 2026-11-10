"""Synthetic candidate profiles. Every fact is known, so answers are gradable."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from . import vocab

FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery",
         "Quinn", "Rowan", "Skyler", "Drew", "Reese", "Emerson", "Hayden", "Parker"]
LAST = ["Lee", "Patel", "Garcia", "Kim", "Nguyen", "Smith", "Okafor", "Rossi",
        "Cohen", "Silva", "Novak", "Haddad", "Larsen", "Mendez", "Ito", "Brown"]


def _level(years: int) -> str:
    if years <= 1:
        return "Entry level"
    if years <= 3:
        return "Associate"
    return "Mid-Senior level"


def make_profiles(postings: pd.DataFrame, n_per_cluster: int = 8, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    states = [s for s in postings["state"].dropna().unique() if 1 < len(s) <= 3] or ["IN"]
    profiles = []
    for cname, c in vocab.CLUSTERS.items():
        pool = postings[postings["cluster"] == cname]
        sal = pool["yearly_min"].dropna()
        lo, hi = (np.percentile(sal, 30), np.percentile(sal, 70)) if len(sal) >= 5 else (50000, 90000)
        for i in range(n_per_cluster):
            years = rng.randint(0, 12)
            skills = rng.sample(c["skills"], k=min(len(c["skills"]), rng.randint(4, 7)))
            if cname != "pm" and rng.random() < 0.4:
                skills.append(rng.choice(vocab.CLUSTERS["pm"]["skills"]))
            # Work history: 1-3 jobs whose years sum to total experience.
            n_jobs = max(1, min(3, years // 3 + 1)) if years else 1
            cuts = sorted(rng.sample(range(1, years), k=n_jobs - 1)) if years > n_jobs else []
            spans = [b - a for a, b in zip([0] + cuts, cuts + [years])] if years else [0]
            history = []
            for span in spans:
                history.append({
                    "title": rng.choice(c["titles"]).title(),
                    "company": f"{rng.choice(LAST)} {rng.choice(['Labs', 'Group', 'Health', 'Systems', 'Partners'])}",
                    "years": span,
                    "skills": sorted(rng.sample(skills, k=min(len(skills), rng.randint(2, 4)))),
                })
            profiles.append({
                "profile_id": f"P{cname[:3].upper()}{i:02d}",
                "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "cluster": cname,
                "target_titles": c["titles"][:2],
                "skills": sorted(set(skills)),
                "years_experience": years,
                "experience_level": _level(years),
                "state": rng.choice(states),
                "open_to_remote": rng.random() < 0.7,
                "open_to_relocate": rng.random() < 0.3,
                "needs_sponsorship": rng.random() < 0.25,
                "min_salary": int(round(rng.uniform(lo, hi) / 5000) * 5000),
                "highest_degree": rng.choice(vocab.DEGREES[1:]),
                "work_history": history,
            })
    return profiles


def years_with_skill(profile: dict, skill: str) -> int:
    return sum(h["years"] for h in profile["work_history"] if skill in h["skills"])


def profile_card(profile: dict) -> str:
    """Compact structured summary pinned in context when the profile-card guide is on."""
    hist = "; ".join(f"{h['title']} at {h['company']} ({h['years']} yrs; {', '.join(h['skills'])})"
                     for h in profile["work_history"])
    return (f"CANDIDATE PROFILE CARD\n"
            f"- Skills: {', '.join(profile['skills'])}\n"
            f"- Experience: {profile['years_experience']} years ({profile['experience_level']})\n"
            f"- Location: {profile['state']}; open to remote: {profile['open_to_remote']}; "
            f"open to relocation: {profile['open_to_relocate']}\n"
            f"- Needs visa sponsorship: {profile['needs_sponsorship']}\n"
            f"- Minimum salary: ${profile['min_salary']:,} per year\n"
            f"- Highest degree: {profile['highest_degree']}\n"
            f"- Work history: {hist}")
