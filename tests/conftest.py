import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from jobharness import data, tasks as T  # noqa: E402
from jobharness.env import JobBoard  # noqa: E402
from jobharness.profiles import make_profiles  # noqa: E402
import make_fixture  # noqa: E402


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    csv = d / "postings_synthetic.csv"
    make_fixture.make().to_csv(csv, index=False)
    df = data.annotate(data.load_postings(str(csv), n=0))
    sandbox = df[df["sandbox_ok"]].reset_index(drop=True)
    sandbox.to_parquet(d / "postings.parquet", index=False)
    import pandas as pd
    sandbox = pd.read_parquet(d / "postings.parquet")  # same round-trip as production
    profiles = make_profiles(sandbox, n_per_cluster=4, seed=3)
    (d / "profiles.json").write_text(json.dumps(profiles))
    tasks = T.build(sandbox, profiles, per_family=6, seed=5, list_size=10)
    with open(d / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    return {"dir": d, "df": df, "sandbox": sandbox, "profiles": profiles, "tasks": tasks,
            "board": JobBoard(sandbox, profiles)}
