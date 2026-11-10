import json

import pytest

from jobharness import analyze, data
from jobharness.env import VISIBLE
from jobharness.graders import grade
from jobharness.harness import HarnessConfig, run_episode
from jobharness.mock import mock_llm
from jobharness.tasks import FAMILIES

HIDDEN = ["min_salary", "max_salary", "med_salary", "pay_period", "remote_allowed",
          "formatted_work_type", "formatted_experience_level", "yearly_min", "skills"]


def test_audit_finds_inconsistencies(built):
    a = data.audit(built["df"])
    assert set(a["field"]) == {"Salary", "Remote allowed", "Work type", "Experience level"}
    sal = a[a.field == "Salary"].iloc[0]
    assert 0 < sal.stated_in_text_when_filled_pct < 100  # fixture hides some salaries
    assert 0 < len(built["sandbox"]) < len(built["df"])


def test_all_families_generated(built):
    fams = {t["family"] for t in built["tasks"]}
    assert fams == set(FAMILIES)
    for t in built["tasks"]:
        if t["family"] == "ranking":
            assert len(set(t["answer"]["labels"].values())) >= 3


def test_tools_never_leak_hidden_fields(built):
    b = built["board"]
    jid = b.ids[0]
    outs = [b.get_job(jid), b.get_job(jid, "key_sections"), b.evidence(jid, "salary")] + b.search("analyst", 5)
    for o in outs:
        for k in o:
            assert k not in HIDDEN, k


def _run(built, model, cfg, family=None):
    recs = []
    for t in built["tasks"]:
        if family and t["family"] != family:
            continue
        llm = mock_llm(model).bind(t, built["board"])
        res = run_episode(llm, built["board"], t, cfg)
        prof = built["board"].profiles.get(t.get("profile_id")) if t.get("profile_id") else None
        recs.append((t, res, grade(t, res["answer"], prof)))
    return recs


@pytest.mark.parametrize("fmt", ["native", "text"])
@pytest.mark.parametrize("tools", ["broad", "narrow"])
def test_oracle_solves_everything(built, fmt, tools):
    cfg = HarnessConfig(tool_format=fmt, tools=tools)
    for t, res, g in _run(built, "mock-oracle", cfg):
        assert res["metrics"]["finish_reason"] == "submitted", (t["task_id"], res["metrics"])
        assert g["success"], (t["task_id"], g)


def test_verify_round_then_accept(built):
    cfg = HarnessConfig(verify_before_submit=True, claim_audit=True)
    for t, res, g in _run(built, "mock-oracle", cfg, "tailoring"):
        assert res["metrics"]["verify_rounds"] == 1 and g["success"]


def test_no_tool_call_without_and_with_recovery(built):
    t = built["tasks"][0]
    res = run_episode(mock_llm("mock-notool").bind(t, built["board"]), built["board"], t, HarnessConfig())
    assert res["metrics"]["finish_reason"] == "no_tool_call" and res["metrics"]["malformed"] == 1
    res = run_episode(mock_llm("mock-notool").bind(t, built["board"]), built["board"], t,
                      HarnessConfig(recovery="retry"))
    assert res["metrics"]["malformed"] == 4 and res["answer"] is None


def test_invalid_answers_counted(built):
    t = built["tasks"][0]
    res = run_episode(mock_llm("mock-badjson").bind(t, built["board"]), built["board"], t,
                      HarnessConfig(recovery="retry"))
    assert res["metrics"]["invalid_answers"] == 4 and res["metrics"]["finish_reason"] == "invalid_answer"


def test_analysis_tables(built, tmp_path):
    out = tmp_path / "runs.jsonl"
    with open(out, "w") as f:
        for name, cfg in [("h0", HarnessConfig(name="h0")), ("hx", HarnessConfig(name="hx", recovery="retry"))]:
            for model in ["mock-oracle", "mock-notool"]:
                for t, res, g in _run(built, model, cfg):
                    f.write(json.dumps({"model": model, "harness": name, "task_id": t["task_id"],
                                        "family": t["family"], "trial": 0, "grade": g,
                                        "answer": res["answer"], "metrics": res["metrics"],
                                        "trajectory": res["trajectory"]}, default=str) + "\n")
    df = analyze.load([out])
    tex, summary = analyze.success_table(df)
    assert "mock-oracle" in tex and summary[summary.family == "ALL"]["success"].max() == 1.0
    assert "McNemar" in analyze.compare_table(df, "h0", "hx")
    assert "Transfer" in analyze.transfer_table(df, "mock-oracle", "h0", "hx", "hx", n_boot=50)
