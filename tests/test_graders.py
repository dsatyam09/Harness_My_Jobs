from jobharness import graders as G
from jobharness.annotate import cohen_kappa


def test_f1():
    assert G.f1({"a", "b"}, {"a", "b"})[2] == 1.0
    assert G.f1({"a"}, {"a", "b"})[2] == 2 * 1 * 0.5 / 1.5
    assert G.f1(set(), {"a"})[2] == 0.0


def test_ndcg_perfect_and_reversed():
    labels = {"a": 3, "b": 2, "c": 0}
    assert abs(G.ndcg_at_k(["a", "b", "c"], labels, 3) - 1.0) < 1e-9
    assert G.ndcg_at_k(["c", "b", "a"], labels, 3) < 0.7


def test_extraction_tolerance_and_types():
    t = {"family": "extraction", "answer": {"min_salary": 50000.0, "max_salary": 70000.0,
                                            "pay_period": "YEARLY", "remote_allowed": True}}
    ok = G.grade(t, {"min_salary": "50000", "max_salary": 70000, "pay_period": "yearly",
                     "remote_allowed": "true"})
    assert ok["success"]
    bad = G.grade(t, {"min_salary": 50000, "max_salary": 75000, "pay_period": "YEARLY",
                      "remote_allowed": True})
    assert not bad["success"] and bad["score"] == 0.75


def test_fabrication_detects_unsupported_skill_and_years():
    profile = {"skills": ["python", "sql"], "years_experience": 3}
    f = G.fabrication_check(["Built SQL pipelines in Python", "Led Kubernetes rollout", "8 years of analytics"], profile)
    assert "kubernetes" in f["fabricated_skills"] and f["inflated_years"] == [8]


def test_kappa():
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3], "quadratic") == 1.0
    assert cohen_kappa(["yes", "no", "yes", "no"], ["yes", "no", "no", "yes"]) == 0.0


def test_tailoring_requires_relevance_not_just_coverage():
    from jobharness.graders import grade_tailoring
    prof = {"skills": ["python", "sql", "excel", "tableau", "java"], "years_experience": 5}
    task = {"answer": {"target_skills": ["python", "sql"]}}
    focused = {"bullets": ["Built python pipelines", "Tuned sql queries", "Shipped python and sql tooling"]}
    dump = {"bullets": ["python sql excel", "tableau java", "excel tableau"]}
    assert grade_tailoring(task, focused, prof)["success"]
    g = grade_tailoring(task, dump, prof)
    assert g["coverage"] == 1.0 and g["relevance"] < 0.6 and not g["success"]
