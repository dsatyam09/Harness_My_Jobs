"""Data preparation and dataset audit.

Usage:
    python -m jobharness.data --csv path/to/postings.csv --n 8000 --out data

Steps
1. Load the LinkedIn Job Postings CSV with DuckDB and sample N postings.
2. Annotate each posting: does the description text state the same salary,
   remote status, work type and experience level as the structured columns?
3. Write the audit table (Pilot A) and keep only "text-consistent" postings
   for the sandbox, so every hidden answer can be recovered from visible text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import vocab
from .latex import write_table

COLUMNS = ["job_id", "company_name", "title", "description", "location",
           "min_salary", "med_salary", "max_salary", "pay_period", "currency",
           "remote_allowed", "formatted_work_type", "formatted_experience_level"]


def load_postings(csv_path: str, n: int, seed: int = 13) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(f'"{c}"' for c in COLUMNS)
    df = con.execute(
        f"SELECT {cols} FROM read_csv_auto(?, all_varchar=true, ignore_errors=true) "
        "WHERE description IS NOT NULL AND length(description) > 200 "
        "AND title IS NOT NULL",
        [csv_path],
    ).df()
    for c in ["min_salary", "med_salary", "max_salary", "remote_allowed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["job_id"] = df["job_id"].astype(str)
    df = df.drop_duplicates("job_id")
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    desc = df["description"].fillna("")
    amounts = desc.map(vocab.money_amounts)

    mult = df["pay_period"].map(vocab.PAY_MULTIPLIER)
    df["yearly_min"] = df["min_salary"] * mult
    df["yearly_max"] = df["max_salary"] * mult
    df["salary_col"] = df["max_salary"].notna() | df["min_salary"].notna()
    df["salary_text"] = amounts.map(bool)
    min_ok = [vocab.amount_in_text(v, a) if v == v else True
              for v, a in zip(df["min_salary"], amounts)]
    max_ok = [vocab.amount_in_text(v, a) for v, a in zip(df["max_salary"], amounts)]
    df["salary_match"] = df["salary_col"] & np.array(min_ok) & np.array(max_ok)
    # Consistent = text states the column's salary, or neither side has a salary.
    df["salary_consistent"] = df["salary_match"] | (~df["salary_col"] & ~df["salary_text"])

    df["remote_col"] = df["remote_allowed"].fillna(0).astype(float) == 1.0
    df["remote_text"] = desc.map(vocab.remote_in_text)
    df["remote_consistent"] = df["remote_col"] == df["remote_text"]

    wt_text = desc.map(vocab.worktypes_in_text)
    df["worktype_text_match"] = [w in s for w, s in zip(df["formatted_work_type"], wt_text)]
    df["level_text_match"] = [vocab.level_in_text(l, d)
                              for l, d in zip(df["formatted_experience_level"], desc)]
    df["years_required"] = desc.map(vocab.years_required)
    df["sponsorship_unavailable"] = desc.map(vocab.sponsorship_unavailable)
    df["skills"] = desc.map(lambda t: sorted(vocab.skills_in_text(t)))
    df["cluster"] = df["title"].map(vocab.cluster_of_title)
    df["state"] = df["location"].fillna("").map(
        lambda s: s.split(",")[-1].strip() if "," in s else s.strip())
    df["sandbox_ok"] = df["salary_consistent"] & df["remote_consistent"]
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    def pct(x):
        return 100.0 * float(np.mean(x)) if len(x) else float("nan")

    rows = []
    sal = df[df["salary_col"]]
    rows.append(("Salary", pct(df["salary_col"]), pct(sal["salary_match"]),
                 pct(df["salary_consistent"])))
    rem = df[df["remote_col"]]
    rows.append(("Remote allowed", pct(df["remote_col"]), pct(rem["remote_text"]),
                 pct(df["remote_consistent"])))
    wt = df[df["formatted_work_type"].notna()]
    rows.append(("Work type", pct(df["formatted_work_type"].notna()),
                 pct(wt["worktype_text_match"]), float("nan")))
    lv = df[df["formatted_experience_level"].notna()]
    rows.append(("Experience level", pct(df["formatted_experience_level"].notna()),
                 pct(lv["level_text_match"]), float("nan")))
    out = pd.DataFrame(rows, columns=["field", "column_filled_pct",
                                      "stated_in_text_when_filled_pct", "consistent_pct"])
    out.attrs["n"] = n
    out.attrs["sandbox_kept"] = int(df["sandbox_ok"].sum())
    out.attrs["with_cluster"] = int(df["cluster"].notna().sum())
    out.attrs["years_stated_pct"] = pct(df["years_required"].notna())
    out.attrs["sponsor_neg_pct"] = pct(df["sponsorship_unavailable"])
    return out


def audit_latex(a: pd.DataFrame) -> str:
    def f(v):
        return "--" if v != v else f"{v:.1f}"
    rows = [[r.field, f(r.column_filled_pct), f(r.stated_in_text_when_filled_pct),
             f(r.consistent_pct)] for r in a.itertuples()]
    return write_table(
        ["Field", "Column filled (\\%)", "Stated in text when filled (\\%)",
         "Text--column consistent (\\%)"],
        rows, colspec="l r r r",
        note=(f"$N={a.attrs['n']:,}$ sampled postings; {a.attrs['sandbox_kept']:,} kept for the "
              f"sandbox (salary and remote consistent); {a.attrs['with_cluster']:,} match a role "
              f"cluster; years of experience stated in {a.attrs['years_stated_pct']:.1f}\\%."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data")
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()

    out, tables = Path(args.out), Path(args.tables)
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = annotate(load_postings(args.csv, args.n, args.seed))
    a = audit(df)
    a.to_csv(out / "audit.csv", index=False)
    (tables / "audit.tex").write_text(audit_latex(a))
    df.to_parquet(out / "postings_all.parquet", index=False)
    df[df["sandbox_ok"]].to_parquet(out / "postings.parquet", index=False)
    summary = {"sampled": len(df), **{k: (v if isinstance(v, (int, float)) else str(v))
                                      for k, v in a.attrs.items()}}
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(a.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
