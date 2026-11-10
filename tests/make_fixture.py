"""Generate a SYNTHETIC postings CSV with the LinkedIn schema, for tests only.
These rows are invented; never report numbers computed from them."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from jobharness import vocab

STATES = ["IN", "IL", "CA", "NY", "TX", "WA"]
CITIES = {"IN": "Indianapolis", "IL": "Chicago", "CA": "San Jose", "NY": "New York",
          "TX": "Austin", "WA": "Seattle"}
LEVELS = ["Entry level", "Associate", "Mid-Senior level"]


def make(n=360, seed=1) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    clusters = list(vocab.CLUSTERS)
    for i in range(n):
        c = clusters[i % len(clusters)]
        cl = vocab.CLUSTERS[c]
        title = rng.choice(cl["titles"]).title() + rng.choice(["", " II", " (Contract)"])
        st = rng.choice(STATES)
        skills = rng.sample(cl["skills"], k=min(len(cl["skills"]), rng.randint(2, 5)))
        remote = rng.random() < 0.4
        hourly = rng.random() < 0.25
        has_sal = rng.random() < 0.6
        lo = rng.randrange(25, 45) if hourly else rng.randrange(50, 130) * 1000
        hi = lo + (rng.randrange(5, 15) if hourly else rng.randrange(10, 40) * 1000)
        parts = [f"We are hiring a {title} to join our team in {CITIES[st]}, {st}.",
                 f"You will work with {', '.join(skills)} every day."]
        if remote:
            parts.append("This role is fully remote within the US.")
        else:
            parts.append("This is an on-site role.")
        sal_in_text = has_sal and rng.random() < 0.85  # some columns are NOT stated in text
        if sal_in_text:
            parts.append(f"The pay range is ${lo:,}-${hi:,} {'per hour' if hourly else 'per year'}.")
        if rng.random() < 0.2:
            parts.append("We are unable to sponsor visas for this position.")
        if rng.random() < 0.5:
            parts.append(f"Requires {rng.randint(1, 6)}+ years of relevant experience.")
        parts.append("We offer health benefits, paid time off and a collaborative culture. " * 2)
        rows.append({
            "job_id": str(3900000000 + i), "company_name": f"Company {i % 37}", "title": title,
            "description": " ".join(parts), "location": f"{CITIES[st]}, {st}",
            "min_salary": lo if has_sal else None, "med_salary": None,
            "max_salary": hi if has_sal else None,
            "pay_period": ("HOURLY" if hourly else "YEARLY") if has_sal else None,
            "currency": "USD" if has_sal else None,
            "remote_allowed": 1.0 if (remote if rng.random() < 0.9 else not remote) else None,
            "formatted_work_type": rng.choice(["Full-time", "Contract", "Part-time"]),
            "formatted_experience_level": rng.choice(LEVELS),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures" / "postings_synthetic.csv"
    make().to_csv(out, index=False)
    print(f"wrote {out}")
